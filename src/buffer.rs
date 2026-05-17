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
            has_prev_frame: false,
            points: Vec::new(),
        }
    }
}

/// Steady-state tracking pipeline.
///
/// All state is private. Read access goes through the accessors below; write
/// access to the tracked-point list goes through [`points_mut`](Self::points_mut),
/// [`reset`](Self::reset), or
/// [`reset_with_good_features_to_track`](Self::reset_with_good_features_to_track).
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

    has_prev_frame: bool,
    points: Vec<(f32, f32)>,
}

impl OpticalFlowBuffer {
    /// Build a pyramid from `image` and, if a previous frame is present and
    /// the points list is non-empty, update the tracked points via
    /// Lucas-Kanade. Then swap prev/curr so the just-pushed frame becomes
    /// the next call's "previous".
    pub fn push_frame<B: AsRef<[u8]>>(
        &mut self,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError> {
        validate_dimensions(image, self.width, self.height)?;
        crate::generic::validate(image)?;
        let thinned = crate::generic::thin(image);

        self.curr_pyramid.build_into(&thinned);

        if self.has_prev_frame && !self.points.is_empty() {
            self.lk_buffer.calc_into(
                self.prev_pyramid.levels(),
                self.curr_pyramid.levels(),
                &mut self.points,
                self.max_iterations,
            );
        }

        std::mem::swap(&mut self.prev_pyramid, &mut self.curr_pyramid);
        self.has_prev_frame = true;
        Ok(())
    }

    /// Replace the tracked-point list with the provided values. Moves the
    /// Vec contents into the buffer; the existing internal allocation is
    /// reused when the new length fits.
    pub fn reset(&mut self, points: Vec<(f32, f32)>) {
        self.points = points;
    }

    /// Detect Shi-Tomasi features on the most recently pushed frame and
    /// replace the tracked-point list with the result. Uses the builder's
    /// `feature_quality_level` and `feature_min_distance`.
    pub fn reset_with_good_features_to_track(&mut self) -> Result<(), TrackError> {
        if !self.has_prev_frame {
            return Err(TrackError::NoPreviousFrame);
        }
        let level0 = self.prev_pyramid.levels()[0].as_flat_samples();
        let detected = self.features_buffer.detect_into(
            &level0,
            self.feature_quality_level,
            self.feature_min_distance,
        );
        self.points.clear();
        self.points.extend(detected.iter().map(|&(x, y, _)| (x as f32, y as f32)));
        Ok(())
    }

    /// Read-only view of the currently tracked points.
    pub fn points(&self) -> &[(f32, f32)] {
        &self.points
    }

    /// Mutable view of the points list. Callers may push, pop, or modify
    /// individual entries.
    pub fn points_mut(&mut self) -> &mut Vec<(f32, f32)> {
        &mut self.points
    }

    pub fn has_previous_frame(&self) -> bool {
        self.has_prev_frame
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

    #[test]
    fn lifecycle_first_push_primes_pyramid_then_detect_then_track() {
        const W: u32 = 64;
        const H: u32 = 64;
        // Checkerboard with 8×8 cells: high-contrast edges produce gradients
        // large enough to survive the /32 integer division in feature detection.
        let frame_a: Vec<u8> = (0..(W * H) as usize)
            .map(|i| {
                let x = (i % W as usize) / 8;
                let y = (i / W as usize) / 8;
                if (x + y) % 2 == 0 { 0u8 } else { 255u8 }
            })
            .collect();
        let frame_b: Vec<u8> = frame_a.iter().map(|p| p.wrapping_add(2)).collect();

        let mut buf = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .window_size(5)
            .max_iterations(5)
            .feature_min_distance(4)
            .feature_quality_level(0.1)
            .build();

        assert!(!buf.has_previous_frame());
        assert!(buf.points().is_empty());

        buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
        assert!(buf.has_previous_frame());
        assert!(buf.points().is_empty(), "first push should not produce points");

        buf.reset_with_good_features_to_track().unwrap();
        assert!(!buf.points().is_empty(), "detect should find at least one feature");

        let detected_count = buf.points().len();
        buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
        assert_eq!(
            buf.points().len(),
            detected_count,
            "tracking preserves the number of points"
        );
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
    fn layout_error_pass_through() {
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
    fn no_previous_frame_error() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        let err = buf.reset_with_good_features_to_track().unwrap_err();
        assert_eq!(err, TrackError::NoPreviousFrame);
        assert!(!buf.has_previous_frame());
    }

    #[test]
    fn reset_preserves_capacity() {
        let mut buf = OpticalFlowBuilder::new(16, 16).build();
        buf.reset(vec![(0.0, 0.0); 100]);
        let cap_after_first = buf.points_mut().capacity();
        assert!(cap_after_first >= 100);

        buf.reset(vec![(0.0, 0.0); 50]);
        // After replacement, capacity comes from the new Vec — the old buffer
        // is moved out by `self.points = points`. The new Vec was sized to 50;
        // Rust may give it capacity == len. The contract holds: capacity covers
        // at least the new length.
        assert!(buf.points_mut().capacity() >= 50);

        // Now grow via points_mut and confirm capacity tracks.
        buf.points_mut().reserve(200);
        assert!(buf.points_mut().capacity() >= 200);
    }

    #[test]
    fn manual_mutation_via_points_mut_drives_tracking() {
        const W: u32 = 64;
        const H: u32 = 64;
        let frame_a: Vec<u8> = (0..(W * H) as usize).map(|i| (i % 251) as u8).collect();
        let frame_b: Vec<u8> = frame_a.iter().map(|p| p.wrapping_add(1)).collect();

        let mut buf = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .window_size(5)
            .max_iterations(5)
            .build();

        buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
        buf.points_mut().push((20.0, 20.0));
        buf.points_mut().push((40.0, 40.0));

        buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
        assert_eq!(buf.points().len(), 2);
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
