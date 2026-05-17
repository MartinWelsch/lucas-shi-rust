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
        let thinned = thin_flat_samples(image);

        self.curr_pyramid.build_into(&thinned);

        if self.has_prev_frame && !self.points.is_empty() {
            self.lk_buffer.calc_into(
                self.prev_pyramid.levels(),
                self.curr_pyramid.levels(),
                &mut self.points,
                self.window_size,
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

fn thin_flat_samples<B: AsRef<[u8]>>(fs: &FlatSamples<B>) -> FlatSamples<&[u8]> {
    FlatSamples {
        samples: fs.samples.as_ref(),
        layout: fs.layout,
        color_hint: fs.color_hint,
    }
}
