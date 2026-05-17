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
    let mut buf = crate::pyramid::PyramidBuffer::with_capacity(
        image.layout.width,
        image.layout.height,
        levels,
    );
    buf.build_into(&thin(image));
    Ok(buf.into_levels())
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
    let mut buf = crate::features::FeaturesBuffer::with_capacity(
        image.layout.width,
        image.layout.height,
        min_distance,
    );
    Ok(buf
        .detect_into(&thin(image), quality_level, min_distance)
        .to_vec())
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::flat::SampleLayout;

    fn make_layout(width: u32, height: u32, height_stride: usize) -> SampleLayout {
        SampleLayout {
            channels: 1,
            channel_stride: 1,
            width,
            width_stride: 1,
            height,
            height_stride,
        }
    }

    fn make_valid_samples(width: u32, height: u32) -> Vec<u8> {
        vec![42u8; (width * height) as usize]
    }

    // --- happy paths ---

    #[test]
    fn build_pyramid_accepts_valid_input() {
        let buf = make_valid_samples(8, 8);
        let fs = FlatSamples {
            samples: &buf[..],
            layout: make_layout(8, 8, 8),
            color_hint: None,
        };
        let pyr = build_pyramid(&fs, 3).expect("valid input must succeed");
        assert_eq!(pyr.len(), 3);
        assert_eq!(pyr[0].dimensions(), (8, 8));
    }

    #[test]
    fn good_features_to_track_accepts_valid_input() {
        let buf = make_valid_samples(16, 16);
        let fs = FlatSamples {
            samples: &buf[..],
            layout: make_layout(16, 16, 16),
            color_hint: None,
        };
        let _ = good_features_to_track(&fs, 0.1, 1)
            .expect("valid input must succeed");
    }

    // --- error paths: build_pyramid ---

    #[test]
    fn build_pyramid_rejects_multi_channel() {
        let buf = make_valid_samples(8, 8);
        let mut layout = make_layout(8, 8, 8);
        layout.channels = 3;
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            build_pyramid(&fs, 2),
            Err(LayoutError::UnsupportedChannels(3))
        );
    }

    #[test]
    fn build_pyramid_rejects_non_unit_width_stride() {
        let buf = make_valid_samples(8, 8);
        let mut layout = make_layout(8, 8, 8);
        layout.width_stride = 3;
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            build_pyramid(&fs, 2),
            Err(LayoutError::UnsupportedWidthStride(3))
        );
    }

    #[test]
    fn build_pyramid_rejects_overlapping_rows() {
        let buf = make_valid_samples(8, 8);
        let layout = make_layout(8, 8, 4);
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            build_pyramid(&fs, 2),
            Err(LayoutError::OverlappingRows { height_stride: 4, width: 8 })
        );
    }

    #[test]
    fn build_pyramid_rejects_too_small_buffer() {
        let buf = vec![0u8; 10];
        let layout = make_layout(8, 8, 8);
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        // required = (8 - 1) * 8 + 8 = 64, actual = 10
        assert_eq!(
            build_pyramid(&fs, 2),
            Err(LayoutError::BufferTooSmall { required: 64, actual: 10 })
        );
    }

    // --- error paths: good_features_to_track ---

    #[test]
    fn good_features_rejects_multi_channel() {
        let buf = make_valid_samples(8, 8);
        let mut layout = make_layout(8, 8, 8);
        layout.channels = 4;
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            good_features_to_track(&fs, 0.1, 1),
            Err(LayoutError::UnsupportedChannels(4))
        );
    }

    #[test]
    fn good_features_rejects_non_unit_width_stride() {
        let buf = make_valid_samples(8, 8);
        let mut layout = make_layout(8, 8, 8);
        layout.width_stride = 2;
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            good_features_to_track(&fs, 0.1, 1),
            Err(LayoutError::UnsupportedWidthStride(2))
        );
    }

    #[test]
    fn good_features_rejects_overlapping_rows() {
        let buf = make_valid_samples(8, 8);
        let layout = make_layout(8, 8, 7);
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            good_features_to_track(&fs, 0.1, 1),
            Err(LayoutError::OverlappingRows { height_stride: 7, width: 8 })
        );
    }

    #[test]
    fn good_features_rejects_too_small_buffer() {
        let buf = vec![0u8; 5];
        let layout = make_layout(8, 8, 8);
        let fs = FlatSamples { samples: &buf[..], layout, color_hint: None };
        assert_eq!(
            good_features_to_track(&fs, 0.1, 1),
            Err(LayoutError::BufferTooSmall { required: 64, actual: 5 })
        );
    }
}
