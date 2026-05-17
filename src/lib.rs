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
//! Two input shapes are supported:
//!
//! - Top-level functions ([`build_pyramid`], [`good_features_to_track`]) take `&GrayImage`
//!   and are infallible.
//! - The [`generic`] module takes `&FlatSamples<B>` and returns `Result<_, LayoutError>` —
//!   use it for zero-copy ingestion of subrects or NV12 Y planes.
//! - For real-time tracking with zero per-frame allocations, build an
//!   [`OpticalFlowBuffer`] via [`OpticalFlowBuilder`].

mod buffer;
mod error;
mod features;
mod lk;
mod pyramid;
mod utils;

pub mod generic;

pub use buffer::{OpticalFlowBuffer, OpticalFlowBuilder};
pub use error::{LayoutError, TrackError};
pub use features::good_features_to_track;
pub use lk::calc_optical_flow;
pub use pyramid::build_pyramid;

// Convenience re-exports so callers don't have to dig into the image crate.
pub use image::flat::{FlatSamples, SampleLayout};
