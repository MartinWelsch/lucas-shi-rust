use image::flat::FlatSamples;
use image::{GrayImage, ImageBuffer, Luma};

/// Reusable pyramid storage. Pre-allocates every level at construction;
/// `build_into` overwrites the existing buffers from a new source view.
pub(crate) struct PyramidBuffer {
    levels: Vec<GrayImage>,
}

impl PyramidBuffer {
    /// Allocate `levels` images sized `(width, height), (width/2, height/2), ...`.
    /// If a halving brings either dimension below 2, no further levels are allocated
    /// (matches the legacy `build_pyramid` early-exit behavior).
    pub(crate) fn with_capacity(width: u32, height: u32, levels: usize) -> Self {
        let mut images = Vec::with_capacity(levels);
        let mut w = width;
        let mut h = height;
        for level in 0..levels {
            images.push(ImageBuffer::new(w, h));
            if level + 1 < levels {
                if w < 2 || h < 2 {
                    break;
                }
                w /= 2;
                h /= 2;
            }
        }
        Self { levels: images }
    }

    /// Overwrite every level's pixels from `image`. No heap allocation when the
    /// buffer was sized for the same dimensions.
    pub(crate) fn build_into(&mut self, image: &FlatSamples<&[u8]>) {
        let width = image.layout.width;
        let height = image.layout.height;
        let row_stride = image.layout.height_stride;
        let src = image.samples;

        // Level 0: row-copy from the (possibly strided) source view.
        let level0 = &mut self.levels[0];
        debug_assert_eq!(level0.dimensions(), (width, height));
        let dst = level0.as_mut();
        let dst_stride = width as usize;
        for y in 0..height {
            let src_off = y as usize * row_stride;
            let dst_off = y as usize * dst_stride;
            dst[dst_off..dst_off + dst_stride]
                .copy_from_slice(&src[src_off..src_off + dst_stride]);
        }

        // Levels 1..n: 2x2-average downsample from the previous level.
        for level in 1..self.levels.len() {
            let (prev, curr) = {
                let (head, tail) = self.levels.split_at_mut(level);
                (&head[level - 1], &mut tail[0])
            };
            let (pw, ph) = prev.dimensions();
            if pw < 2 || ph < 2 {
                break;
            }
            let new_width = pw / 2;
            let new_height = ph / 2;
            debug_assert_eq!(curr.dimensions(), (new_width, new_height));

            for y in 0..new_height {
                for x in 0..new_width {
                    let px = 2 * x;
                    let py = 2 * y;
                    let p1 = prev.get_pixel(px, py)[0] as u32;
                    let p2 = prev.get_pixel(px + 1, py)[0] as u32;
                    let p3 = prev.get_pixel(px, py + 1)[0] as u32;
                    let p4 = prev.get_pixel(px + 1, py + 1)[0] as u32;
                    let avg = ((p1 + p2 + p3 + p4) / 4) as u8;
                    curr.put_pixel(x, y, Luma([avg]));
                }
            }
        }
    }

    pub(crate) fn levels(&self) -> &[GrayImage] {
        &self.levels
    }

    /// Consume the buffer and return its owned levels. Used by the legacy
    /// `build_pyramid(&GrayImage, _)` which still returns `Vec<GrayImage>`.
    pub(crate) fn into_levels(self) -> Vec<GrayImage> {
        self.levels
    }
}

/// Builds a pyramid of images where each successive layer is half as large in width and height.
///
/// This method just takes the average of the 4 pixels, no interpolation or anything like that.
///
/// # Arguments
/// * `image` - Source image (grayscale)
/// * `levels` - Level count
///
/// # Returns
/// Vector of layers in descending order of size. First element is source image.
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let (w, h) = image.dimensions();
    let mut buf = PyramidBuffer::with_capacity(w, h, levels);
    buf.build_into(&image.as_flat_samples());
    buf.into_levels()
}
