//! Steady-state tracking pipeline.
//!
//! [`OpticalFlowBuilder`] gathers the global parameters and resolution for the
//! pipeline. [`OpticalFlowBuilder::build`] returns an [`OpticalFlowBuffer`]
//! with every internal buffer pre-allocated. After warm-up,
//! [`OpticalFlowBuffer::push_frame`] performs zero heap allocations in the
//! library.

use image::flat::FlatSamples;

use crate::error::TrackError;
use crate::features::FeaturesBuffer;
use crate::lk::LkBuffer;
use crate::pyramid::PyramidBuffer;

const DEFAULT_PYRAMID_LEVELS: usize = 3;
const DEFAULT_WINDOW_SIZE: usize = 21;
const DEFAULT_MAX_ITERATIONS: usize = 30;
const DEFAULT_FEATURE_QUALITY_LEVEL: f32 = 0.4;
const DEFAULT_FEATURE_MIN_DISTANCE: u32 = 10;

/// A tracked feature point with spatial position and detection strength.
///
/// `strength` is the Shi-Tomasi min-eigenvalue from feature detection
/// (always >= 0). Features produced by
/// [`calculate_flow`](OpticalFlowBuffer::calculate_flow) inherit their input
/// feature's strength unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feature {
    pub x: f32,
    pub y: f32,
    pub strength: f32,
}

/// Pyramid image data plus the features associated with that frame.
///
/// `features` starts empty after construction or after a `push_frame` rotation;
/// it is populated by `good_features_to_track` or `calculate_flow`.
pub(crate) struct FrameBuffer {
    pyramid: PyramidBuffer,
    features: Vec<Feature>,
}

impl FrameBuffer {
    pub(crate) fn with_capacity(width: u32, height: u32, levels: usize) -> Self {
        Self {
            pyramid: PyramidBuffer::with_capacity(width, height, levels),
            features: Vec::new(),
        }
    }
}

/// Builder for [`OpticalFlowBuffer`].
pub struct OpticalFlowBuilder {
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,
}

impl OpticalFlowBuilder {
    /// Start a builder for a fixed-resolution pipeline.
    ///
    /// Defaults: 3 pyramid levels, 21x21 window, 30 max iterations,
    /// `quality_level = 0.4`, `min_distance = 10`.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pyramid_levels: DEFAULT_PYRAMID_LEVELS,
            window_size: DEFAULT_WINDOW_SIZE,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            feature_quality_level: DEFAULT_FEATURE_QUALITY_LEVEL,
            feature_min_distance: DEFAULT_FEATURE_MIN_DISTANCE,
        }
    }

    pub fn pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn feature_quality_level(mut self, q: f32) -> Self {
        self.feature_quality_level = q;
        self
    }

    pub fn feature_min_distance(mut self, d: u32) -> Self {
        self.feature_min_distance = d;
        self
    }

    /// Construct the buffer, pre-allocating all internal storage.
    ///
    /// Panics if `width == 0`, `height == 0`, `pyramid_levels == 0`, or
    /// `window_size` is even.
    pub fn build(self) -> OpticalFlowBuffer {
        assert!(self.width > 0, "width must be > 0");
        assert!(self.height > 0, "height must be > 0");
        assert!(self.pyramid_levels > 0, "pyramid_levels must be > 0");
        assert!(self.window_size % 2 == 1, "window_size must be odd");

        let prev_frame = FrameBuffer::with_capacity(self.width, self.height, self.pyramid_levels);
        let curr_frame = FrameBuffer::with_capacity(self.width, self.height, self.pyramid_levels);
        let lk_buffer = LkBuffer::with_capacity(
            self.width,
            self.height,
            self.pyramid_levels,
            self.window_size,
        );
        let features_buffer = FeaturesBuffer::with_capacity(
            self.width,
            self.height,
            self.feature_min_distance,
        );

        OpticalFlowBuffer {
            width: self.width,
            height: self.height,
            pyramid_levels: self.pyramid_levels,
            window_size: self.window_size,
            max_iterations: self.max_iterations,
            feature_quality_level: self.feature_quality_level,
            feature_min_distance: self.feature_min_distance,
            prev_frame,
            curr_frame,
            lk_buffer,
            features_buffer,
            staging_positions: Vec::new(),
            has_curr: false,
            has_prev: false,
        }
    }
}

/// Steady-state tracking pipeline.
///
/// All state is private. Read access to detected/tracked features goes through
/// [`current_features`](Self::current_features) and
/// [`previous_features`](Self::previous_features); configuration is exposed
/// through the remaining read-only accessors.
pub struct OpticalFlowBuffer {
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,

    prev_frame: FrameBuffer,
    curr_frame: FrameBuffer,
    lk_buffer: LkBuffer,
    features_buffer: FeaturesBuffer,
    /// Reusable staging Vec for the LK call (LkBuffer takes `Vec<(f32, f32)>`).
    /// Grows during warm-up to the steady-state feature count, then reused.
    staging_positions: Vec<(f32, f32)>,

    has_curr: bool,
    has_prev: bool,
}

impl OpticalFlowBuffer {
    /// Rotate prev/curr and build the new frame's pyramid into curr.
    /// Clears `curr_frame.features` (the new frame has no features yet).
    pub fn push_frame<B: AsRef<[u8]>>(
        &mut self,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError> {
        validate_dimensions(image, self.width, self.height)?;
        validate(image)?;
        let thinned = thin(image);

        std::mem::swap(&mut self.prev_frame, &mut self.curr_frame);
        self.curr_frame.pyramid.build_into(&thinned);
        self.curr_frame.features.clear();

        if self.has_curr {
            self.has_prev = true;
        }
        self.has_curr = true;
        Ok(())
    }

    /// Detect Shi-Tomasi features on the current frame and write them into
    /// `current_features()` (cleared first). Errors `NoCurrentFrame` if no
    /// frame has been pushed.
    pub fn good_features_to_track(&mut self) -> Result<(), TrackError> {
        if !self.has_curr {
            return Err(TrackError::NoCurrentFrame);
        }
        let level0 = self.curr_frame.pyramid.levels()[0].as_flat_samples();
        let detected = self.features_buffer.detect_into(
            &level0,
            self.feature_quality_level,
            self.feature_min_distance,
        );
        self.curr_frame.features.clear();
        self.curr_frame.features.extend(
            detected.iter().map(|&(x, y, q)| Feature {
                x: x as f32,
                y: y as f32,
                strength: q,
            }),
        );
        Ok(())
    }

    /// Track `previous_features()` from the previous frame into the current
    /// frame using Lucas-Kanade. Writes results into `current_features()`
    /// (cleared first). Strength is preserved from each input feature.
    /// Errors `NoPreviousFrame` if fewer than two frames have been pushed.
    pub fn calculate_flow(&mut self) -> Result<(), TrackError> {
        if !self.has_prev {
            return Err(TrackError::NoPreviousFrame);
        }

        // Stage previous positions for LK.
        self.staging_positions.clear();
        self.staging_positions
            .extend(self.prev_frame.features.iter().map(|f| (f.x, f.y)));

        self.lk_buffer.calc_into(
            self.prev_frame.pyramid.levels(),
            self.curr_frame.pyramid.levels(),
            &mut self.staging_positions,
            self.max_iterations,
        );

        self.curr_frame.features.clear();
        self.curr_frame.features.extend(
            self.prev_frame
                .features
                .iter()
                .zip(self.staging_positions.iter())
                .map(|(prev, &(x, y))| Feature {
                    x,
                    y,
                    strength: prev.strength,
                }),
        );
        Ok(())
    }

    /// Returns the features for the most recently pushed frame.
    ///
    /// Empty until [`good_features_to_track`](Self::good_features_to_track) or
    /// [`calculate_flow`](Self::calculate_flow) has been called after the last
    /// [`push_frame`](Self::push_frame).
    pub fn current_features(&self) -> &[Feature] {
        &self.curr_frame.features
    }

    /// Returns the features for the frame prior to the most recently pushed one.
    ///
    /// Empty until a second [`push_frame`](Self::push_frame) has been called.
    pub fn previous_features(&self) -> &[Feature] {
        &self.prev_frame.features
    }

    /// Mutable view of the current frame's feature list. Callers may push,
    /// pop, retain, or modify individual entries. The next
    /// [`good_features_to_track`](Self::good_features_to_track) or
    /// [`calculate_flow`](Self::calculate_flow) call will overwrite the list
    /// in place — manual edits made after one of those calls are preserved
    /// until the next call that writes the list.
    pub fn current_features_mut(&mut self) -> &mut Vec<Feature> {
        &mut self.curr_frame.features
    }

    /// Mutable view of the previous frame's feature list. The next
    /// [`calculate_flow`](Self::calculate_flow) reads from this list, so
    /// manual edits made before that call drive what gets tracked. Note
    /// that the next [`push_frame`](Self::push_frame) will swap this list
    /// into the current frame's slot.
    pub fn previous_features_mut(&mut self) -> &mut Vec<Feature> {
        &mut self.prev_frame.features
    }

    // --- read-only accessors ---

    /// True after at least one `push_frame` call.
    pub fn has_current_frame(&self) -> bool {
        self.has_curr
    }

    /// True after at least two `push_frame` calls — i.e., `calculate_flow`
    /// has the data it needs.
    pub fn has_previous_frame(&self) -> bool {
        self.has_prev
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn pyramid_levels(&self) -> usize {
        self.pyramid_levels
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn feature_quality_level(&self) -> f32 {
        self.feature_quality_level
    }

    pub fn feature_min_distance(&self) -> u32 {
        self.feature_min_distance
    }
}

fn validate_dimensions<B: AsRef<[u8]>>(
    image: &FlatSamples<B>,
    expected_width: u32,
    expected_height: u32,
) -> Result<(), TrackError> {
    let actual = (image.layout.width, image.layout.height);
    let expected = (expected_width, expected_height);
    if actual != expected {
        return Err(TrackError::DimensionMismatch { expected, actual });
    }
    Ok(())
}

fn validate<B: AsRef<[u8]>>(fs: &FlatSamples<B>) -> Result<(), crate::LayoutError> {
    use crate::LayoutError;
    if fs.layout.channels != 1 {
        return Err(LayoutError::UnsupportedChannels(fs.layout.channels));
    }
    if fs.layout.width_stride != 1 {
        return Err(LayoutError::UnsupportedWidthStride(fs.layout.width_stride));
    }
    let width = fs.layout.width;
    let height = fs.layout.height;
    if fs.layout.height_stride < width as usize {
        return Err(LayoutError::OverlappingRows {
            height_stride: fs.layout.height_stride,
            width,
        });
    }
    if width == 0 || height == 0 {
        return Ok(());
    }
    let required = (height as usize - 1) * fs.layout.height_stride + width as usize;
    let actual = fs.samples.as_ref().len();
    if actual < required {
        return Err(LayoutError::BufferTooSmall { required, actual });
    }
    Ok(())
}

fn thin<B: AsRef<[u8]>>(fs: &FlatSamples<B>) -> FlatSamples<&[u8]> {
    FlatSamples {
        samples: fs.samples.as_ref(),
        layout: fs.layout,
        color_hint: fs.color_hint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LayoutError;
    use image::flat::SampleLayout;

    fn make_view<'a>(buf: &'a [u8], width: u32, height: u32) -> FlatSamples<&'a [u8]> {
        FlatSamples {
            samples: buf,
            layout: SampleLayout {
                channels: 1,
                channel_stride: 1,
                width,
                width_stride: 1,
                height,
                height_stride: width as usize,
            },
            color_hint: None,
        }
    }

    fn checkerboard(width: u32, height: u32, cell: u32) -> Vec<u8> {
        let mut out = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let on = ((x / cell) + (y / cell)) % 2 == 0;
                out[(y * width + x) as usize] = if on { 255 } else { 0 };
            }
        }
        out
    }

    #[test]
    fn lifecycle_push_detect_push_track() {
        const W: u32 = 64;
        const H: u32 = 64;
        let frame_a = checkerboard(W, H, 8);
        let frame_b: Vec<u8> = frame_a.iter().map(|p| p.wrapping_add(1)).collect();

        let mut buf = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .window_size(5)
            .max_iterations(5)
            .feature_min_distance(4)
            .feature_quality_level(0.1)
            .build();

        assert!(!buf.has_current_frame());
        assert!(!buf.has_previous_frame());
        assert!(buf.current_features().is_empty());
        assert!(buf.previous_features().is_empty());

        buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
        assert!(buf.has_current_frame());
        assert!(buf.current_features().is_empty(), "push clears current_features");

        buf.good_features_to_track().unwrap();
        let n = buf.current_features().len();
        assert!(n > 0, "checkerboard yields features");
        // Strength should be non-negative.
        assert!(buf.current_features().iter().all(|f| f.strength >= 0.0));

        buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
        assert!(buf.has_previous_frame());
        assert_eq!(
            buf.previous_features().len(),
            n,
            "swap preserves features attached to old curr"
        );
        assert!(buf.current_features().is_empty(), "new curr starts empty");

        buf.calculate_flow().unwrap();
        assert_eq!(buf.current_features().len(), n);

        // Strength is preserved through tracking.
        for (prev, curr) in buf
            .previous_features()
            .iter()
            .zip(buf.current_features().iter())
        {
            assert_eq!(prev.strength, curr.strength);
        }
    }

    #[test]
    fn push_frame_clears_stale_curr_features() {
        const W: u32 = 16;
        const H: u32 = 16;
        let frame = vec![0u8; (W * H) as usize];

        let mut buf = OpticalFlowBuilder::new(W, H).build();
        buf.push_frame(&make_view(&frame, W, H)).unwrap();
        buf.push_frame(&make_view(&frame, W, H)).unwrap();
        // After two pushes with no detect/track, both feature lists must be empty.
        assert!(buf.current_features().is_empty());
        assert!(buf.previous_features().is_empty());
    }

    #[test]
    fn good_features_errors_before_any_push() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        assert_eq!(
            buf.good_features_to_track(),
            Err(TrackError::NoCurrentFrame)
        );
    }

    #[test]
    fn calculate_flow_errors_before_two_pushes() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        let frame = vec![0u8; 16 * 16];
        assert_eq!(buf.calculate_flow(), Err(TrackError::NoPreviousFrame));

        buf.push_frame(&make_view(&frame, 16, 16)).unwrap();
        assert_eq!(buf.calculate_flow(), Err(TrackError::NoPreviousFrame));

        buf.push_frame(&make_view(&frame, 16, 16)).unwrap();
        assert!(buf.calculate_flow().is_ok());
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut buf = OpticalFlowBuilder::new(100, 50).build();
        let img = vec![0u8; 200 * 50];
        let err = buf.push_frame(&make_view(&img, 200, 50)).unwrap_err();
        assert_eq!(
            err,
            TrackError::DimensionMismatch {
                expected: (100, 50),
                actual: (200, 50),
            }
        );
    }

    #[test]
    fn layout_error_unsupported_channels() {
        let mut buf = OpticalFlowBuilder::new(8, 8).build();
        let data = vec![0u8; 8 * 8 * 3];
        let view = FlatSamples {
            samples: &data[..],
            layout: SampleLayout {
                channels: 3,
                channel_stride: 1,
                width: 8,
                width_stride: 1,
                height: 8,
                height_stride: 8,
            },
            color_hint: None,
        };
        let err = buf.push_frame(&view).unwrap_err();
        assert_eq!(err, TrackError::Layout(LayoutError::UnsupportedChannels(3)));
    }

    #[test]
    fn layout_error_unsupported_width_stride() {
        let mut buf = OpticalFlowBuilder::new(8, 8).build();
        let data = vec![0u8; 8 * 8 * 2];
        let view = FlatSamples {
            samples: &data[..],
            layout: SampleLayout {
                channels: 1,
                channel_stride: 1,
                width: 8,
                width_stride: 2,
                height: 8,
                height_stride: 16,
            },
            color_hint: None,
        };
        let err = buf.push_frame(&view).unwrap_err();
        assert_eq!(
            err,
            TrackError::Layout(LayoutError::UnsupportedWidthStride(2))
        );
    }

    #[test]
    fn layout_error_overlapping_rows() {
        let mut buf = OpticalFlowBuilder::new(8, 8).build();
        let data = vec![0u8; 8 * 8];
        let view = FlatSamples {
            samples: &data[..],
            layout: SampleLayout {
                channels: 1,
                channel_stride: 1,
                width: 8,
                width_stride: 1,
                height: 8,
                height_stride: 4,
            },
            color_hint: None,
        };
        let err = buf.push_frame(&view).unwrap_err();
        assert_eq!(
            err,
            TrackError::Layout(LayoutError::OverlappingRows {
                height_stride: 4,
                width: 8,
            })
        );
    }

    #[test]
    fn layout_error_buffer_too_small() {
        let mut buf = OpticalFlowBuilder::new(8, 8).build();
        let data = vec![0u8; 10];
        let view = FlatSamples {
            samples: &data[..],
            layout: SampleLayout {
                channels: 1,
                channel_stride: 1,
                width: 8,
                width_stride: 1,
                height: 8,
                height_stride: 8,
            },
            color_hint: None,
        };
        let err = buf.push_frame(&view).unwrap_err();
        assert_eq!(
            err,
            TrackError::Layout(LayoutError::BufferTooSmall {
                required: 64,
                actual: 10
            })
        );
    }

    #[test]
    fn accessors_return_builder_values() {
        let buf = OpticalFlowBuilder::new(320, 240)
            .pyramid_levels(4)
            .window_size(11)
            .max_iterations(20)
            .feature_quality_level(0.25)
            .feature_min_distance(7)
            .build();

        assert_eq!(buf.dimensions(), (320, 240));
        assert_eq!(buf.pyramid_levels(), 4);
        assert_eq!(buf.window_size(), 11);
        assert_eq!(buf.max_iterations(), 20);
        assert!((buf.feature_quality_level() - 0.25).abs() < 1e-6);
        assert_eq!(buf.feature_min_distance(), 7);
    }

    #[test]
    fn current_features_mut_allows_caller_edits() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        let frame = vec![0u8; 16 * 16];
        buf.push_frame(&make_view(&frame, 16, 16)).unwrap();

        buf.current_features_mut().push(Feature {
            x: 1.0,
            y: 2.0,
            strength: 3.0,
        });
        buf.current_features_mut().push(Feature {
            x: 4.0,
            y: 5.0,
            strength: 6.0,
        });
        assert_eq!(buf.current_features().len(), 2);

        buf.current_features_mut().retain(|f| f.x > 2.0);
        assert_eq!(buf.current_features().len(), 1);
        assert_eq!(buf.current_features()[0].x, 4.0);
    }

    #[test]
    fn previous_features_mut_seeds_what_calculate_flow_tracks() {
        const W: u32 = 32;
        const H: u32 = 32;
        let frame_a = vec![128u8; (W * H) as usize];
        let frame_b: Vec<u8> = frame_a.iter().map(|p| p.wrapping_add(1)).collect();

        let mut buf = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .window_size(5)
            .max_iterations(5)
            .build();

        buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
        buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
        // After two pushes both feature lists are empty.
        assert!(buf.previous_features().is_empty());

        // Manually seed previous_features so calculate_flow has something to track.
        buf.previous_features_mut().push(Feature {
            x: 10.0,
            y: 10.0,
            strength: 1.0,
        });
        buf.previous_features_mut().push(Feature {
            x: 20.0,
            y: 20.0,
            strength: 2.0,
        });

        buf.calculate_flow().unwrap();
        assert_eq!(buf.current_features().len(), 2);
        // Strength preserved from the manually seeded inputs.
        assert_eq!(buf.current_features()[0].strength, 1.0);
        assert_eq!(buf.current_features()[1].strength, 2.0);
    }
}
