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

        let prev_pyramid = PyramidBuffer::with_capacity(self.width, self.height, self.pyramid_levels);
        let curr_pyramid = PyramidBuffer::with_capacity(self.width, self.height, self.pyramid_levels);
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
            prev_pyramid,
            curr_pyramid,
            lk_buffer,
            features_buffer,
            has_curr: false,
            has_prev: false,
        }
    }
}

/// Steady-state tracking pipeline.
///
/// All state is private. Read access goes through the accessors below; callers
/// manage their own feature-point buffers and pass them into
/// [`good_features_to_track`](Self::good_features_to_track) and
/// [`calculate_flow`](Self::calculate_flow).
pub struct OpticalFlowBuffer {
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,

    prev_pyramid: PyramidBuffer,
    curr_pyramid: PyramidBuffer,
    lk_buffer: LkBuffer,
    features_buffer: FeaturesBuffer,

    has_curr: bool,
    has_prev: bool,
}

impl OpticalFlowBuffer {
    /// Build a pyramid from `image` and rotate the buffer's pyramid storage
    /// so that `curr_pyramid` holds the just-pushed frame and `prev_pyramid`
    /// holds the previously-pushed frame (if any).
    ///
    /// Does not run optical flow — call [`calculate_flow`](Self::calculate_flow)
    /// after at least two `push_frame` calls. Does not touch any feature list —
    /// the caller manages those.
    pub fn push_frame<B: AsRef<[u8]>>(
        &mut self,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError> {
        validate_dimensions(image, self.width, self.height)?;
        validate(image)?;
        let thinned = thin(image);

        // Rotate: previous curr becomes prev, then build the new frame into curr.
        std::mem::swap(&mut self.prev_pyramid, &mut self.curr_pyramid);
        self.curr_pyramid.build_into(&thinned);

        if self.has_curr {
            self.has_prev = true;
        }
        self.has_curr = true;
        Ok(())
    }

    /// Detect Shi-Tomasi features on the most recently pushed frame and
    /// write their positions into `out` (cleared first). The output is in
    /// the same coordinate system the next [`calculate_flow`](Self::calculate_flow)
    /// will use.
    ///
    /// Quality scores are dropped — if you need them, use the legacy
    /// `good_features_to_track` free function.
    ///
    /// Errors with [`TrackError::NoCurrentFrame`] if no frame has been pushed.
    pub fn good_features_to_track(
        &mut self,
        out: &mut Vec<(f32, f32)>,
    ) -> Result<(), TrackError> {
        if !self.has_curr {
            return Err(TrackError::NoCurrentFrame);
        }
        let level0 = self.curr_pyramid.levels()[0].as_flat_samples();
        let detected = self.features_buffer.detect_into(
            &level0,
            self.feature_quality_level,
            self.feature_min_distance,
        );
        out.clear();
        out.extend(detected.iter().map(|&(x, y, _)| (x as f32, y as f32)));
        Ok(())
    }

    /// Track `features` from the previous frame into the current frame using
    /// Lucas-Kanade. Writes the new positions into `out_features` (cleared
    /// first; sized to `features.len()`). The caller typically calls
    /// `std::mem::swap(&mut features, &mut out_features)` afterwards so the
    /// fresh positions become the input for the next iteration.
    ///
    /// Errors with [`TrackError::NoPreviousFrame`] if fewer than two frames
    /// have been pushed.
    pub fn calculate_flow(
        &mut self,
        features: &[(f32, f32)],
        out_features: &mut Vec<(f32, f32)>,
    ) -> Result<(), TrackError> {
        if !self.has_prev {
            return Err(TrackError::NoPreviousFrame);
        }
        out_features.clear();
        out_features.extend_from_slice(features);
        self.lk_buffer.calc_into(
            self.prev_pyramid.levels(),
            self.curr_pyramid.levels(),
            out_features,
            self.max_iterations,
        );
        Ok(())
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

        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut tracked: Vec<(f32, f32)> = Vec::new();

        // First push primes curr.
        buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
        assert!(buf.has_current_frame());
        assert!(!buf.has_previous_frame());

        // Detect on curr (= frame_a).
        buf.good_features_to_track(&mut points).unwrap();
        assert!(!points.is_empty(), "checkerboard should yield features");

        // Second push rotates: prev=frame_a, curr=frame_b.
        buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
        assert!(buf.has_previous_frame());

        // Track.
        let n = points.len();
        buf.calculate_flow(&points, &mut tracked).unwrap();
        assert_eq!(tracked.len(), n, "tracked count matches feature count");
    }

    #[test]
    fn good_features_errors_before_any_push() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        let mut points = Vec::new();
        assert_eq!(
            buf.good_features_to_track(&mut points),
            Err(TrackError::NoCurrentFrame)
        );
        assert!(!buf.has_current_frame());
    }

    #[test]
    fn calculate_flow_errors_before_two_pushes() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        let frame = vec![0u8; 16 * 16];

        // Zero pushes.
        let features: Vec<(f32, f32)> = vec![(0.0, 0.0)];
        let mut out = Vec::new();
        assert_eq!(
            buf.calculate_flow(&features, &mut out),
            Err(TrackError::NoPreviousFrame)
        );

        // One push — still no prev.
        buf.push_frame(&make_view(&frame, 16, 16)).unwrap();
        assert_eq!(
            buf.calculate_flow(&features, &mut out),
            Err(TrackError::NoPreviousFrame)
        );
        assert!(!buf.has_previous_frame());

        // Two pushes — now ready.
        buf.push_frame(&make_view(&frame, 16, 16)).unwrap();
        assert!(buf.has_previous_frame());
        assert!(buf.calculate_flow(&features, &mut out).is_ok());
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
        assert_eq!(err, TrackError::Layout(LayoutError::UnsupportedWidthStride(2)));
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
            TrackError::Layout(LayoutError::BufferTooSmall { required: 64, actual: 10 })
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
}
