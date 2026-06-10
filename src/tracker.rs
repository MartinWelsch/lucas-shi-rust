//! Steady-state tracking pipeline.
//!
//! [`OpticalFlowBuilder`] gathers the global parameters and resolution.
//! [`OpticalFlowBuilder::build`] returns an [`OpticalFlowTracker`] with
//! every internal buffer pre-allocated. After warm-up,
//! [`OpticalFlowTracker::push_frame`] performs zero heap allocations.
//!
//! For manual pipeline assembly (without the convenience wrapper), see the
//! [`buffers`](crate::buffers) module.

use image::flat::FlatSamples;

use crate::error::TrackError;
use crate::feature::Feature;
use crate::features::FeaturesBuffer;
use crate::lk::LkBuffer;
use crate::pyramid::PyramidBuffer;
use crate::validate::{as_byte_view, validate_dimensions, validate_layout};

const DEFAULT_PYRAMID_LEVELS: usize = 3;
const DEFAULT_WINDOW_SIZE: usize = 21;
const DEFAULT_MAX_ITERATIONS: usize = 30;
const DEFAULT_FEATURE_QUALITY_LEVEL: f32 = 0.4;
const DEFAULT_FEATURE_MIN_DISTANCE: u32 = 10;
const DEFAULT_MAX_FEATURES: usize = 500;

/// Builder for [`OpticalFlowTracker`].
pub struct OpticalFlowBuilder {
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,
    max_features: usize,
}

impl OpticalFlowBuilder {
    /// Start a builder for a fixed-resolution pipeline.
    ///
    /// Defaults: 3 pyramid levels, 21x21 window, 30 max iterations,
    /// `quality_level = 0.4`, `min_distance = 10`, `max_features = 500`.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pyramid_levels: DEFAULT_PYRAMID_LEVELS,
            window_size: DEFAULT_WINDOW_SIZE,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            feature_quality_level: DEFAULT_FEATURE_QUALITY_LEVEL,
            feature_min_distance: DEFAULT_FEATURE_MIN_DISTANCE,
            max_features: DEFAULT_MAX_FEATURES,
        }
    }

    /// Number of pyramid levels for both detection and LK tracking.
    /// Level 0 is full resolution; each subsequent level halves both dimensions.
    /// Must be > 0; `build()` panics otherwise. Default: 3.
    pub fn pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    /// Side length of the Lucas-Kanade search window, in pixels. Must be odd;
    /// `build()` panics otherwise. Larger windows are more robust against
    /// noise but more expensive; typical values are 11–31. Default: 21.
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Maximum refinement iterations per pyramid level inside the LK
    /// inner loop. Larger values converge more accurately on hard motion
    /// but cost more per frame. Default: 30.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Shi-Tomasi quality threshold, expressed as a fraction (0.0..=1.0) of
    /// the strongest corner's score in each frame. A candidate is kept iff
    /// its min-eigenvalue ≥ `quality_level × strongest_score`. Lower → more
    /// features kept. Default: 0.4.
    pub fn feature_quality_level(mut self, q: f32) -> Self {
        self.feature_quality_level = q;
        self
    }

    /// Minimum spatial separation between accepted features, in pixels.
    /// Larger → sparser, more spread-out detections. A value of 0 is
    /// internally clamped to 1 to avoid division by zero in the grid
    /// filter. Default: 10.
    pub fn feature_min_distance(mut self, d: u32) -> Self {
        self.feature_min_distance = d;
        self
    }

    /// Hard upper bound on the number of features `detect_features` will
    /// produce, and the pre-allocated capacity of the features Vec. Cuts
    /// steady-state memory dramatically — for 1920×1080, going from
    /// `width × height` to a few hundred saves tens of MB. Default: 500.
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = n;
        self
    }

    /// Construct the tracker, pre-allocating all internal storage.
    ///
    /// Panics if `width == 0`, `height == 0`, `pyramid_levels == 0`, or
    /// `window_size` is even.
    pub fn build(self) -> OpticalFlowTracker {
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
            self.max_features,
        );
        let features_buffer = FeaturesBuffer::with_capacity(
            self.width,
            self.height,
            self.feature_min_distance,
            self.max_features,
        );

        OpticalFlowTracker {
            width: self.width,
            height: self.height,
            pyramid_levels: self.pyramid_levels,
            window_size: self.window_size,
            max_iterations: self.max_iterations,
            feature_quality_level: self.feature_quality_level,
            feature_min_distance: self.feature_min_distance,
            max_features: self.max_features,
            prev_pyramid,
            curr_pyramid,
            features: Vec::with_capacity(self.max_features),
            lk_buffer,
            features_buffer,
            fb_origin: Vec::with_capacity(self.max_features),
            fb_back: Vec::with_capacity(self.max_features),
            fb_fwd_valid: Vec::with_capacity(self.max_features),
            fb_back_valid: Vec::with_capacity(self.max_features),
            frame_count: 0,
        }
    }
}

/// Steady-state tracking pipeline.
///
/// Owns a single feature list (`features`) that describes positions in
/// whichever frame was most recently consumed by [`detect_features`] or
/// [`calculate_flow`]. [`push_frame`] does not touch `features`; it rotates
/// the internal pyramids so the next [`calculate_flow`] tracks the current
/// `features` from the previous frame into the just-pushed one.
///
/// [`detect_features`]: Self::detect_features
/// [`calculate_flow`]: Self::calculate_flow
/// [`push_frame`]: Self::push_frame
pub struct OpticalFlowTracker {
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,
    max_features: usize,

    prev_pyramid: PyramidBuffer,
    curr_pyramid: PyramidBuffer,
    features: Vec<Feature>,
    lk_buffer: LkBuffer,
    features_buffer: FeaturesBuffer,

    // Reusable scratch for forward-backward validation. All pre-grown to
    // `max_features` so `calculate_flow_fb` allocates nothing after warm-up.
    fb_origin: Vec<(f32, f32)>,
    fb_back: Vec<Feature>,
    fb_fwd_valid: Vec<bool>,
    fb_back_valid: Vec<bool>,

    frame_count: u64,
}

impl OpticalFlowTracker {
    /// Rotate prev/curr pyramids and build the new frame's pyramid into curr.
    /// Does not touch `features` — they keep their existing positions, which
    /// now describe locations in the previous frame (the one rotating out of
    /// curr). Call [`calculate_flow`](Self::calculate_flow) next to move
    /// them forward into the just-pushed frame.
    pub fn push_frame<B: AsRef<[u8]>>(
        &mut self,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError> {
        validate_dimensions(image, self.width, self.height)?;
        validate_layout(image)?;
        let thinned = as_byte_view(image);

        std::mem::swap(&mut self.prev_pyramid, &mut self.curr_pyramid);
        self.curr_pyramid.build_into(&thinned);
        self.frame_count += 1;
        Ok(())
    }

    /// Detect Shi-Tomasi corner features on the most recently pushed frame,
    /// writing up to [`max_features`](Self::max_features) entries into
    /// [`features`](Self::features) (cleared first), in descending quality
    /// order. Errors [`TrackError::NoCurrentFrame`] if no frame has been
    /// pushed.
    pub fn detect_features(&mut self) -> Result<(), TrackError> {
        if self.frame_count == 0 {
            return Err(TrackError::NoCurrentFrame);
        }
        let level0 = self.curr_pyramid.levels()[0].as_flat_samples();
        self.features_buffer.detect_into(
            &level0,
            self.feature_quality_level,
            self.feature_min_distance,
            self.max_features,
            &mut self.features,
        );
        Ok(())
    }

    /// Track [`features`](Self::features) from the previous frame into the
    /// most recently pushed frame using Lucas-Kanade, updating each
    /// `Feature`'s position in place. `strength` is left untouched.
    /// Errors [`TrackError::NoPreviousFrame`] if fewer than two frames have
    /// been pushed.
    pub fn calculate_flow(&mut self) -> Result<(), TrackError> {
        if self.frame_count < 2 {
            return Err(TrackError::NoPreviousFrame);
        }

        self.lk_buffer.calc_into(
            self.prev_pyramid.levels(),
            self.curr_pyramid.levels(),
            &mut self.features,
            self.max_iterations,
        );

        Ok(())
    }

    /// Forward-backward validated optical flow.
    ///
    /// Runs the forward pass exactly like
    /// [`calculate_flow`](Self::calculate_flow) (each `Feature`'s position
    /// is advanced into the just-pushed frame, in place), then tracks every
    /// feature *backward* from its new position into the previous frame and
    /// measures the round-trip distance to where it started. A feature is
    /// reported **invalid** when any of the following holds:
    ///
    /// * the forward pass skipped it (window out of bounds / singular
    ///   Hessian) — it never actually moved and is stuck at `(0, 0)`;
    /// * the backward pass skipped it for the same reasons;
    /// * the round-trip error exceeds `max_fb_error` pixels.
    ///
    /// This method does **not** prune `features`. Instead it writes a
    /// per-feature validity mask into `valid_out` (cleared then resized to
    /// `features().len()`, index-aligned with `features()`), leaving the
    /// caller to drop invalid entries while keeping any parallel snapshot
    /// (e.g. previous-frame positions) aligned. The forward positions are
    /// retained in `features` regardless of validity.
    ///
    /// Errors [`TrackError::NoPreviousFrame`] if fewer than two frames have
    /// been pushed. No heap allocation after warm-up (all scratch is
    /// pre-grown to `max_features`).
    pub fn calculate_flow_fb(
        &mut self,
        max_fb_error: f32,
        valid_out: &mut Vec<bool>,
    ) -> Result<(), TrackError> {
        if self.frame_count < 2 {
            return Err(TrackError::NoPreviousFrame);
        }

        // Snapshot original (previous-frame) positions for the round trip.
        self.fb_origin.clear();
        self.fb_origin
            .extend(self.features.iter().map(|f| (f.x, f.y)));

        // Forward pass: prev -> curr, capturing per-feature validity.
        self.lk_buffer.calc_into_status(
            self.prev_pyramid.levels(),
            self.curr_pyramid.levels(),
            &mut self.features,
            self.max_iterations,
            Some(&mut self.fb_fwd_valid),
        );

        // Backward pass: curr -> prev, starting from the forward result.
        self.fb_back.clear();
        self.fb_back.extend(self.features.iter().copied());
        self.lk_buffer.calc_into_status(
            self.curr_pyramid.levels(),
            self.prev_pyramid.levels(),
            &mut self.fb_back,
            self.max_iterations,
            Some(&mut self.fb_back_valid),
        );

        // Round-trip gate.
        let max_err2 = max_fb_error * max_fb_error;
        valid_out.clear();
        valid_out.resize(self.features.len(), false);
        // Index loop: walks five parallel buffers in lockstep.
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.features.len() {
            let fwd_ok = self.fb_fwd_valid[i];
            let back_ok = self.fb_back_valid[i];
            let (ox, oy) = self.fb_origin[i];
            let back = self.fb_back[i];
            let ex = back.x - ox;
            let ey = back.y - oy;
            let err2 = ex * ex + ey * ey;
            valid_out[i] = fwd_ok && back_ok && err2 <= max_err2;
        }

        Ok(())
    }

    /// Borrow the current feature list.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Mutable view of the feature list. Callers may push, pop, retain, or
    /// modify individual entries. The next [`detect_features`] overwrites
    /// the list; [`calculate_flow`] mutates positions in place.
    ///
    /// Do not change the underlying `Vec`'s capacity — the zero-allocation
    /// property depends on it staying at the pre-allocated size.
    ///
    /// [`detect_features`]: Self::detect_features
    /// [`calculate_flow`]: Self::calculate_flow
    pub fn features_mut(&mut self) -> &mut Vec<Feature> {
        &mut self.features
    }

    // --- read-only accessors ---

    /// Number of frames consumed by [`push_frame`](Self::push_frame).
    /// `0` after construction, `1` once a current frame exists, `≥ 2` once
    /// [`calculate_flow`](Self::calculate_flow) can be called.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// `(width, height)` configured at build time.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of pyramid levels configured at build time.
    pub fn pyramid_levels(&self) -> usize {
        self.pyramid_levels
    }

    /// Configured LK search window side length, in pixels.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Configured maximum LK refinement iterations per pyramid level.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Configured Shi-Tomasi quality threshold (0.0..=1.0).
    pub fn feature_quality_level(&self) -> f32 {
        self.feature_quality_level
    }

    /// Configured minimum spatial separation between features, in pixels.
    pub fn feature_min_distance(&self) -> u32 {
        self.feature_min_distance
    }

    /// Configured upper bound on detected features.
    pub fn max_features(&self) -> usize {
        self.max_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LayoutError;
    use image::flat::SampleLayout;

    fn make_view(buf: &[u8], width: u32, height: u32) -> FlatSamples<&[u8]> {
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
                let on = ((x / cell) + (y / cell)).is_multiple_of(2);
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

        let mut tracker = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .window_size(5)
            .max_iterations(5)
            .feature_min_distance(4)
            .feature_quality_level(0.1)
            .build();

        assert_eq!(tracker.frame_count(), 0);
        assert!(tracker.features().is_empty());

        tracker.push_frame(&make_view(&frame_a, W, H)).unwrap();
        assert_eq!(tracker.frame_count(), 1);
        assert!(tracker.features().is_empty(), "push does not populate features");

        tracker.detect_features().unwrap();
        let n = tracker.features().len();
        assert!(n > 0, "checkerboard yields features");
        assert!(tracker.features().iter().all(|f| f.strength >= 0.0));

        let prev_positions: Vec<(f32, f32)> =
            tracker.features().iter().map(|f| (f.x, f.y)).collect();

        tracker.push_frame(&make_view(&frame_b, W, H)).unwrap();
        assert_eq!(tracker.frame_count(), 2);
        assert_eq!(
            tracker.features().len(),
            n,
            "push_frame does not clear features",
        );

        tracker.calculate_flow().unwrap();
        assert_eq!(tracker.features().len(), n);

        // For nearly-identical frames, positions should barely move.
        for ((px, py), feat) in prev_positions.iter().zip(tracker.features().iter()) {
            assert!((feat.x - px).abs() < 2.0);
            assert!((feat.y - py).abs() < 2.0);
        }
    }

    #[test]
    fn detect_features_errors_before_any_push() {
        let mut tracker = OpticalFlowBuilder::new(16, 16).build();
        assert_eq!(tracker.detect_features(), Err(TrackError::NoCurrentFrame));
    }

    #[test]
    fn calculate_flow_errors_before_two_pushes() {
        let mut tracker = OpticalFlowBuilder::new(16, 16).build();
        let frame = vec![0u8; 16 * 16];
        assert_eq!(tracker.calculate_flow(), Err(TrackError::NoPreviousFrame));

        tracker.push_frame(&make_view(&frame, 16, 16)).unwrap();
        assert_eq!(tracker.calculate_flow(), Err(TrackError::NoPreviousFrame));

        tracker.push_frame(&make_view(&frame, 16, 16)).unwrap();
        assert!(tracker.calculate_flow().is_ok());
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut tracker = OpticalFlowBuilder::new(100, 50).build();
        let img = vec![0u8; 200 * 50];
        let err = tracker.push_frame(&make_view(&img, 200, 50)).unwrap_err();
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
        let mut tracker = OpticalFlowBuilder::new(8, 8).build();
        let data = [0u8; 8 * 8 * 3];
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
        let err = tracker.push_frame(&view).unwrap_err();
        assert_eq!(err, TrackError::Layout(LayoutError::UnsupportedChannels(3)));
    }

    #[test]
    fn layout_error_unsupported_width_stride() {
        let mut tracker = OpticalFlowBuilder::new(8, 8).build();
        let data = [0u8; 8 * 8 * 2];
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
        let err = tracker.push_frame(&view).unwrap_err();
        assert_eq!(
            err,
            TrackError::Layout(LayoutError::UnsupportedWidthStride(2))
        );
    }

    #[test]
    fn layout_error_overlapping_rows() {
        let mut tracker = OpticalFlowBuilder::new(8, 8).build();
        let data = [0u8; 8 * 8];
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
        let err = tracker.push_frame(&view).unwrap_err();
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
        let mut tracker = OpticalFlowBuilder::new(8, 8).build();
        let data = [0u8; 10];
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
        let err = tracker.push_frame(&view).unwrap_err();
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
        let tracker = OpticalFlowBuilder::new(320, 240)
            .pyramid_levels(4)
            .window_size(11)
            .max_iterations(20)
            .feature_quality_level(0.25)
            .feature_min_distance(7)
            .max_features(123)
            .build();

        assert_eq!(tracker.dimensions(), (320, 240));
        assert_eq!(tracker.pyramid_levels(), 4);
        assert_eq!(tracker.window_size(), 11);
        assert_eq!(tracker.max_iterations(), 20);
        assert!((tracker.feature_quality_level() - 0.25).abs() < 1e-6);
        assert_eq!(tracker.feature_min_distance(), 7);
        assert_eq!(tracker.max_features(), 123);
    }

    #[test]
    fn detect_features_caps_at_max_features() {
        const W: u32 = 64;
        const H: u32 = 64;
        let frame = checkerboard(W, H, 4);

        let mut tracker = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .feature_min_distance(2)
            .feature_quality_level(0.01)
            .max_features(10)
            .build();

        tracker.push_frame(&make_view(&frame, W, H)).unwrap();
        tracker.detect_features().unwrap();

        assert!(tracker.features().len() <= 10);
        assert!(!tracker.features().is_empty());
    }

    #[test]
    fn features_mut_allows_caller_edits() {
        let mut tracker = OpticalFlowBuilder::new(16, 16).build();
        let frame = vec![0u8; 16 * 16];
        tracker.push_frame(&make_view(&frame, 16, 16)).unwrap();

        tracker.features_mut().push(Feature { x: 1.0, y: 2.0, strength: 3.0 });
        tracker.features_mut().push(Feature { x: 4.0, y: 5.0, strength: 6.0 });
        assert_eq!(tracker.features().len(), 2);

        tracker.features_mut().retain(|f| f.x > 2.0);
        assert_eq!(tracker.features().len(), 1);
        assert_eq!(tracker.features()[0].x, 4.0);
    }

    #[test]
    fn calculate_flow_recovers_known_translation() {
        const W: u32 = 64;
        const H: u32 = 64;
        const DX: i32 = 3;
        const DY: i32 = -2;

        let frame_a: Vec<u8> = (0..(W * H) as usize)
            .map(|i| {
                let x = (i as u32) % W;
                let y = (i as u32) / W;
                let stripe = (y / 3) % 4;
                let vstripe = (x / 3) % 4;
                (stripe.wrapping_mul(50) + vstripe.wrapping_mul(30)) as u8
            })
            .collect();

        let mut frame_b = vec![0u8; (W * H) as usize];
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let sx = x - DX;
                let sy = y - DY;
                if sx >= 0 && sx < W as i32 && sy >= 0 && sy < H as i32 {
                    frame_b[(y as u32 * W + x as u32) as usize] =
                        frame_a[(sy as u32 * W + sx as u32) as usize];
                }
            }
        }

        let mut tracker = OpticalFlowBuilder::new(W, H)
            .pyramid_levels(2)
            .window_size(7)
            .max_iterations(20)
            .build();

        tracker.push_frame(&make_view(&frame_a, W, H)).unwrap();
        tracker.features_mut().push(Feature { x: 30.0, y: 30.0, strength: 1.0 });

        tracker.push_frame(&make_view(&frame_b, W, H)).unwrap();
        // push_frame does not touch features — they're still at (30, 30),
        // now interpreted as positions in the previous frame.
        assert_eq!(tracker.features().len(), 1);
        assert_eq!(tracker.features()[0].x, 30.0);

        tracker.calculate_flow().unwrap();
        assert_eq!(tracker.features().len(), 1);
        let tracked = tracker.features()[0];
        let expected_x = 30.0 + DX as f32;
        let expected_y = 30.0 + DY as f32;
        assert!((tracked.x - expected_x).abs() < 1.0);
        assert!((tracked.y - expected_y).abs() < 1.0);
        // Strength preserved through tracking.
        assert_eq!(tracked.strength, 1.0);
    }
}
