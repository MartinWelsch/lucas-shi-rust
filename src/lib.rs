//! High-performance computer vision algorithms for real-time applications
//!
//! Provides implementations of:
//! - Lucas-Kanade optical flow
//! - Shi-Tomasi feature detection
//! - Optimized image processing pipelines
//!
//! Designed to be compatible with WebAssembly (Wasm).
//!
//! # Input formats
//!
//! The primary API uses [`OpticalFlowBuilder`] + [`OpticalFlowBuffer`]:
//! `push_frame` accepts any `&FlatSamples<B>` matching the configured
//! resolution — including subrects of larger images and the Y plane of an
//! NV12 buffer — with zero per-frame allocations after warm-up.
//!
//! The top-level functions [`build_pyramid`], [`good_features_to_track`], and
//! [`calc_optical_flow`] are deprecated since 0.4.0 and kept only for
//! backward compatibility.

mod buffer;
mod error;
mod features;
mod lk;
mod pyramid;
mod utils;

pub use buffer::{OpticalFlowBuffer, OpticalFlowBuilder};
pub use error::{LayoutError, TrackError};
#[allow(deprecated)]
pub use features::good_features_to_track;
#[allow(deprecated)]
pub use lk::calc_optical_flow;
#[allow(deprecated)]
pub use pyramid::build_pyramid;

// Convenience re-exports so callers don't have to dig into the image crate.
pub use image::flat::{FlatSamples, SampleLayout};
