use image::flat::FlatSamples;
use image::{GrayImage, ImageBuffer, Luma};

use crate::utils::fast_gradients::compute_gradients_into_no_zero;

/// A single cached Scharr gradient plane (`i16` per pixel).
type GradPlane = ImageBuffer<Luma<i16>, Vec<i16>>;

/// Compute the dimensions of each pyramid level given a starting
/// `(width, height)`. Level 0 is the full resolution; each subsequent
/// level halves both dimensions. The loop stops early if a halving
/// would produce a level with either dimension < 2 — the returned Vec
/// then has fewer entries than `levels`.
pub fn pyramid_dims(width: u32, height: u32, levels: usize) -> Vec<(u32, u32)> {
    let mut dims = Vec::with_capacity(levels);
    let mut w = width;
    let mut h = height;
    for level in 0..levels {
        dims.push((w, h));
        if level + 1 < levels {
            if w < 2 || h < 2 {
                break;
            }
            w /= 2;
            h /= 2;
        }
    }
    dims
}

/// Reusable pyramid storage. Pre-allocates every level at construction;
/// [`build_pyramid`](crate::buffers::build_pyramid) overwrites the existing
/// buffers from a new source view.
///
/// Also caches each level's Scharr gradients (`grad_x`/`grad_y`), computed
/// once per [`build_into`](Self::build_into) call rather than being
/// recomputed by every LK pass that treats this pyramid as the "prev" role
/// (see [`LkBuffer::calc_into_status_cached`](crate::lk::LkBuffer)). Border
/// pixels of the gradient planes are zero from allocation and stay zero
/// forever — the gradient dispatch only ever writes the interior — so they
/// need zeroing exactly once, not on every build.
pub struct PyramidBuffer {
    levels: Vec<GrayImage>,
    grad_x: Vec<GradPlane>,
    grad_y: Vec<GradPlane>,
}

impl PyramidBuffer {
    /// Allocate `levels` images sized `(width, height), (width/2, height/2), ...`.
    /// If a halving brings either dimension below 2, no further levels are allocated
    /// (matches the legacy `build_pyramid` early-exit behavior).
    pub fn with_capacity(width: u32, height: u32, levels: usize) -> Self {
        let dims = pyramid_dims(width, height, levels);
        let images = dims.iter().map(|&(w, h)| ImageBuffer::new(w, h)).collect();
        let grad_x = dims.iter().map(|&(w, h)| ImageBuffer::new(w, h)).collect();
        let grad_y = dims.iter().map(|&(w, h)| ImageBuffer::new(w, h)).collect();
        Self { levels: images, grad_x, grad_y }
    }

    /// Overwrite every level's pixels from `image`, then recompute that
    /// level's cached gradients. No heap allocation when the buffer was
    /// sized for the same dimensions.
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
        compute_gradients_into_no_zero(
            &self.levels[0].as_flat_samples(),
            &mut self.grad_x[0],
            &mut self.grad_y[0],
        );

        // Levels 1..n: 2x2-average downsample from the previous level.
        for level in 1..self.levels.len() {
            let stop = {
                let (head, tail) = self.levels.split_at_mut(level);
                let prev = &head[level - 1];
                let curr = &mut tail[0];
                let (pw, ph) = prev.dimensions();
                if pw < 2 || ph < 2 {
                    true
                } else {
                    let new_width = pw / 2;
                    let new_height = ph / 2;
                    debug_assert_eq!(curr.dimensions(), (new_width, new_height));

                    // Row-slice 2x2-average downsample: no per-pixel bounds
                    // checks (get_pixel/put_pixel each did one), same
                    // (p1+p2+p3+p4)/4 integer math and tap order
                    // ((px,py),(px+1,py),(px,py+1),(px+1,py+1)) as before,
                    // so results are exact, not just "close".
                    let pw = pw as usize;
                    let new_width = new_width as usize;
                    let prev_raw = prev.as_raw();
                    let curr_raw = curr.as_mut();
                    for y in 0..new_height as usize {
                        let row0 = &prev_raw[(2 * y) * pw..(2 * y) * pw + pw];
                        let row1 = &prev_raw[(2 * y + 1) * pw..(2 * y + 1) * pw + pw];
                        let out_row = &mut curr_raw[y * new_width..y * new_width + new_width];
                        for ((c0, c1), out) in row0
                            .chunks_exact(2)
                            .zip(row1.chunks_exact(2))
                            .zip(out_row.iter_mut())
                        {
                            let p1 = c0[0] as u32;
                            let p2 = c0[1] as u32;
                            let p3 = c1[0] as u32;
                            let p4 = c1[1] as u32;
                            *out = ((p1 + p2 + p3 + p4) / 4) as u8;
                        }
                    }
                    false
                }
            };
            if stop {
                break;
            }
            compute_gradients_into_no_zero(
                &self.levels[level].as_flat_samples(),
                &mut self.grad_x[level],
                &mut self.grad_y[level],
            );
        }
    }

    /// Borrow the pre-allocated pyramid levels (level 0 first, full resolution).
    pub fn levels(&self) -> &[GrayImage] {
        &self.levels
    }

    /// Borrow this level's cached Scharr gradient planes, computed by the
    /// most recent [`build_into`](Self::build_into) call.
    ///
    /// INVARIANT (load-bearing for cached LK): the planes are all-zero on a
    /// freshly constructed buffer and only become valid after `build_into`.
    /// A caller that reads them for a buffer that has never been built (e.g.
    /// using a just-allocated pyramid as the flow `prev`) gets zero
    /// gradients → singular Hessian → every feature reported invalid, i.e. a
    /// SILENT tracking dropout, not a panic. The steady-state pipeline avoids
    /// this because flow only ever treats a pyramid as `prev` after it has
    /// served as `curr` (and so been built) — enforced by the
    /// `frame_count >= 2` gate in `calculate_flow`/`calculate_flow_fb`.
    pub(crate) fn grad_planes(
        &self,
        level: usize,
    ) -> (&GradPlane, &GradPlane) {
        (&self.grad_x[level], &self.grad_y[level])
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
///
/// # Deprecated
/// Use [`OpticalFlowBuilder`](crate::OpticalFlowBuilder) +
/// [`OpticalFlowTracker`](crate::OpticalFlowTracker); the pyramid is built
/// internally by `push_frame` with zero per-frame allocations after warm-up.
#[deprecated(
    since = "0.4.0",
    note = "use OpticalFlowBuilder + OpticalFlowTracker; the pyramid is built internally by push_frame"
)]
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let (w, h) = image.dimensions();
    let mut buf = PyramidBuffer::with_capacity(w, h, levels);
    buf.build_into(&image.as_flat_samples());
    buf.into_levels()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference copy of the pre-A3 per-pixel downsample (`get_pixel` /
    /// `put_pixel`, one bounds check each). Kept only in test code so the
    /// row-slice version in [`PyramidBuffer::build_into`] can be pinned
    /// against it.
    fn downsample_reference(prev: &GrayImage) -> Option<GrayImage> {
        let (pw, ph) = prev.dimensions();
        if pw < 2 || ph < 2 {
            return None;
        }
        let new_width = pw / 2;
        let new_height = ph / 2;
        let mut curr = GrayImage::new(new_width, new_height);
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
        Some(curr)
    }

    fn asymmetric_image(width: u32, height: u32) -> GrayImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 53 + y * 29 + (x ^ y) * 7 + (x * y) % 17) & 0xff) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }
        img
    }

    /// A3 pin: `PyramidBuffer::build_into`'s row-slice 2x2-average
    /// downsample must produce byte-identical levels to the original
    /// per-pixel `get_pixel`/`put_pixel` implementation, on a non-square,
    /// non-power-of-two-friendly image (so trailing odd rows/columns and
    /// unaligned chunk boundaries are actually exercised) across every
    /// level of a multi-level pyramid.
    #[test]
    fn build_into_matches_reference_downsample_on_asymmetric_image() {
        const LEVELS: usize = 4;
        let width = 97;
        let height = 61;
        let img = asymmetric_image(width, height);

        // Reference: repeatedly downsample with the old per-pixel algorithm.
        let mut expected_levels = vec![img.clone()];
        while expected_levels.len() < LEVELS {
            match downsample_reference(expected_levels.last().unwrap()) {
                Some(next) => expected_levels.push(next),
                None => break,
            }
        }

        // Actual: PyramidBuffer::build_into.
        let mut buf = PyramidBuffer::with_capacity(width, height, LEVELS);
        buf.build_into(&img.as_flat_samples());
        let actual_levels = buf.levels();

        assert_eq!(actual_levels.len(), expected_levels.len());
        for (level, (expected, actual)) in expected_levels.iter().zip(actual_levels.iter()).enumerate() {
            assert_eq!(expected.dimensions(), actual.dimensions(), "level {level} dims mismatch");
            assert_eq!(expected.as_raw(), actual.as_raw(), "level {level} pixel mismatch");
        }
    }
}
