//! High-performance computer vision algorithms for real-time applications
//!
//! Provides implementations of:
//! - Lucas-Kanade optical flow
//! - Shi-Tomasi feature detection
//! - Optimized image processing pipelines
//!
//! Designed to be compatible with WebAssembly (Wasm).
//!
//! # Two API layers
//!
//! - [`OpticalFlowBuilder`] + [`OpticalFlowTracker`] — convenience pipeline
//!   that owns the pyramid rotation, feature list, and detection/LK scratch
//!   buffers. After warm-up, `push_frame` performs zero per-frame
//!   allocations.
//! - [`buffers`] — the underlying [`PyramidBuffer`](buffers::PyramidBuffer),
//!   [`LkBuffer`](buffers::LkBuffer), and
//!   [`FeaturesBuffer`](buffers::FeaturesBuffer) plus free functions that
//!   operate on them. Use this to assemble custom pipelines (e.g., sharing
//!   a pyramid across multiple LK runs, keeping more than two pyramids in
//!   memory, or driving detection without a tracker).
//!
//! The top-level functions [`build_pyramid`], [`good_features_to_track`], and
//! [`calc_optical_flow`] are deprecated since 0.4.0 and kept only for
//! backward compatibility.

mod error;
mod feature;
mod features;
mod lk;
mod pyramid;
mod tracker;
mod utils;
mod validate;

pub use error::{LayoutError, TrackError};
pub use feature::Feature;
pub use tracker::{OpticalFlowBuilder, OpticalFlowTracker};

#[allow(deprecated)]
pub use features::good_features_to_track;
#[allow(deprecated)]
pub use lk::calc_optical_flow;
#[allow(deprecated)]
pub use pyramid::build_pyramid;

// Convenience re-exports so callers don't have to dig into the image crate.
pub use image::flat::{FlatSamples, SampleLayout};

/// Manual pipeline assembly: the pre-allocated buffer types plus free
/// functions that operate on them.
///
/// Use this when [`OpticalFlowTracker`](crate::OpticalFlowTracker) doesn't
/// fit — e.g., sharing one pyramid across multiple LK runs, keeping more
/// than two pyramids in memory, or running detection without a tracker.
///
/// Each free function takes its buffer(s) as `&mut` so the call site is
/// explicit about which storage is being reused. After warm-up, none of
/// them allocate.
pub mod buffers {
    use image::flat::FlatSamples;

    use crate::error::TrackError;
    use crate::feature::Feature;
    use crate::validate::{as_byte_view, validate_dimensions, validate_layout};

    pub use crate::features::FeaturesBuffer;
    pub use crate::lk::LkBuffer;
    pub use crate::pyramid::PyramidBuffer;

    /// Validate `image` against the configured dimensions of `buf` and
    /// rebuild every pyramid level in place. Returns the same `TrackError`
    /// variants as [`OpticalFlowTracker::push_frame`](crate::OpticalFlowTracker::push_frame).
    pub fn build_pyramid<B: AsRef<[u8]>>(
        buf: &mut PyramidBuffer,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError> {
        let (w, h) = buf.levels()[0].dimensions();
        validate_dimensions(image, w, h)?;
        validate_layout(image)?;
        let view = as_byte_view(image);
        buf.build_into(&view);
        Ok(())
    }

    /// Detect Shi-Tomasi corner features on level 0 of `pyramid`, writing
    /// up to `max_features` entries into `out` (cleared first), in
    /// descending quality order.
    pub fn detect_features(
        buf: &mut FeaturesBuffer,
        pyramid: &PyramidBuffer,
        quality_level: f32,
        min_distance: u32,
        max_features: usize,
        out: &mut Vec<Feature>,
    ) {
        let level0 = pyramid.levels()[0].as_flat_samples();
        buf.detect_into(&level0, quality_level, min_distance, max_features, out);
    }

    /// Track `features` from `prev` into `curr` using Lucas-Kanade,
    /// mutating each `Feature`'s `(x, y)` in place. `strength` is left
    /// untouched.
    pub fn track(
        buf: &mut LkBuffer,
        prev: &PyramidBuffer,
        curr: &PyramidBuffer,
        features: &mut [Feature],
        max_iterations: usize,
    ) {
        buf.calc_into(prev.levels(), curr.levels(), features, max_iterations);
    }
}
