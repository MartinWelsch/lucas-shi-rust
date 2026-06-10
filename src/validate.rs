//! Shared `FlatSamples` validation used by both the [`OpticalFlowTracker`]
//! convenience API and the [`buffers`](crate::buffers) free functions.
//!
//! [`OpticalFlowTracker`]: crate::OpticalFlowTracker

use image::flat::FlatSamples;

use crate::error::{LayoutError, TrackError};

pub(crate) fn validate_dimensions<B: AsRef<[u8]>>(
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

pub(crate) fn validate_layout<B: AsRef<[u8]>>(fs: &FlatSamples<B>) -> Result<(), LayoutError> {
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

pub(crate) fn as_byte_view<B: AsRef<[u8]>>(fs: &FlatSamples<B>) -> FlatSamples<&[u8]> {
    FlatSamples {
        samples: fs.samples.as_ref(),
        layout: fs.layout,
        color_hint: fs.color_hint,
    }
}
