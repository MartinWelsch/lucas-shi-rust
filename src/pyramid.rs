use image::flat::FlatSamples;
use image::{GrayImage, ImageBuffer, Luma};

/// Builds a pyramid of images where each successive layer is half as large in width and height
///
/// This method just takes the average of the 4 pixels, no interpolation or anything like that
///
/// # Arguments
/// * `image` - Source image (grayscale)
/// * `levels` - Level count
///
/// # Returns
/// Vector of layers in descending order of size. First element is source image
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    build_pyramid_impl(&image.as_flat_samples(), levels)
}

pub(crate) fn build_pyramid_impl(
    image: &FlatSamples<&[u8]>,
    levels: usize,
) -> Vec<GrayImage> {
    let width = image.layout.width;
    let height = image.layout.height;
    let row_stride = image.layout.height_stride;
    let src = image.samples;

    // Allocate level 0 as a contiguous GrayImage by row-copying from the (possibly strided) view.
    let mut level0: GrayImage = ImageBuffer::new(width, height);
    {
        let dst = level0.as_mut();
        let dst_stride = width as usize;
        for y in 0..height {
            let src_off = y as usize * row_stride;
            let dst_off = y as usize * dst_stride;
            dst[dst_off..dst_off + dst_stride]
                .copy_from_slice(&src[src_off..src_off + dst_stride]);
        }
    }

    let mut pyramid = vec![level0];

    for level in 1..levels {
        let previous_level = &pyramid[level - 1];
        let (width, height) = (previous_level.width(), previous_level.height());

        // Check that the image can be downscaled
        if width < 2 || height < 2 {
            break;
        }

        let new_width = width / 2;
        let new_height = height / 2;

        let mut new_image = ImageBuffer::new(new_width, new_height);

        for y in 0..new_height {
            for x in 0..new_width {
                let px = 2 * x;
                let py = 2 * y;

                // Average 4 pixels
                let pixel1 = previous_level.get_pixel(px, py)[0] as u32;
                let pixel2 = previous_level.get_pixel(px + 1, py)[0] as u32;
                let pixel3 = previous_level.get_pixel(px, py + 1)[0] as u32;
                let pixel4 = previous_level.get_pixel(px + 1, py + 1)[0] as u32;

                let average = ((pixel1 + pixel2 + pixel3 + pixel4) / 4) as u8;

                new_image.put_pixel(x, y, Luma([average]));
            }
        }

        pyramid.push(new_image);
    }

    pyramid
}
