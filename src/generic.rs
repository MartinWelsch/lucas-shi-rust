//! Generic input API.
//!
//! These entry points accept `image::flat::FlatSamples<B>` — any single-channel byte buffer
//! with arbitrary row stride. This enables zero-copy ingestion of subrects of a larger
//! grayscale image and of the Y plane of an NV12/YUV buffer.
//!
//! The layout is validated once at entry; on violation a [`LayoutError`] is returned.

use image::flat::FlatSamples;
use image::GrayImage;

use crate::LayoutError;
use crate::features::good_features_to_track_impl;
use crate::pyramid::build_pyramid_impl;

/// Build a Gaussian-style pyramid from any single-channel buffer.
///
/// # Errors
/// Returns [`LayoutError`] if the `FlatSamples` layout cannot be safely consumed:
/// - [`LayoutError::UnsupportedChannels`] — `channels != 1`
/// - [`LayoutError::UnsupportedWidthStride`] — `width_stride != 1`
/// - [`LayoutError::OverlappingRows`] — `height_stride < width`
/// - [`LayoutError::BufferTooSmall`] — `samples` is shorter than the declared layout requires
pub fn build_pyramid<B: AsRef<[u8]>>(
    image: &FlatSamples<B>,
    levels: usize,
) -> Result<Vec<GrayImage>, LayoutError> {
    validate(image)?;
    Ok(build_pyramid_impl(&thin(image), levels))
}

/// Find good features to track from any single-channel buffer.
///
/// # Errors
/// Same as [`build_pyramid`].
pub fn good_features_to_track<B: AsRef<[u8]>>(
    image: &FlatSamples<B>,
    quality_level: f32,
    min_distance: u32,
) -> Result<Vec<(u32, u32, f32)>, LayoutError> {
    validate(image)?;
    Ok(good_features_to_track_impl(
        &thin(image),
        quality_level,
        min_distance,
    ))
}

fn validate<B: AsRef<[u8]>>(fs: &FlatSamples<B>) -> Result<(), LayoutError> {
    if fs.layout.channels != 1 {
        return Err(LayoutError::UnsupportedChannels(fs.layout.channels));
    }
    if fs.layout.width_stride != 1 {
        return Err(LayoutError::UnsupportedWidthStride(fs.layout.width_stride));
    }
    let width = fs.layout.width;
    let height = fs.layout.height;
    if (fs.layout.height_stride) < width as usize {
        return Err(LayoutError::OverlappingRows {
            height_stride: fs.layout.height_stride,
            width,
        });
    }
    if width == 0 || height == 0 {
        // Zero-dimension input is not an error; the workers short-circuit downstream.
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
