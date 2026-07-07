use image::flat::FlatSamples;
use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::Ordering;

use crate::feature::Feature;
use crate::utils::{box_filter_3x3::box_filter_3x3_in_place, fast_gradients::compute_gradients_into};

/// Reusable storage for Shi-Tomasi feature detection. Pre-allocates every
/// per-call buffer at construction;
/// [`detect_features`](crate::buffers::detect_features) reuses them.
pub struct FeaturesBuffer {
    gx: ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: ImageBuffer<Luma<i16>, Vec<i16>>,
    ix_sq: ImageBuffer<Luma<i16>, Vec<i16>>,
    iy_sq: ImageBuffer<Luma<i16>, Vec<i16>>,
    ix_iy: ImageBuffer<Luma<i16>, Vec<i16>>,
    features: Vec<(u32, u32, f32)>,
    is_local_max: Vec<bool>,
    grid: Vec<Option<(u32, u32)>>,
}

impl FeaturesBuffer {
    /// Pre-allocate every buffer using best-effort upper bounds derived from
    /// the configured resolution and `min_distance`.
    pub fn with_capacity(width: u32, height: u32, min_distance: u32, _max_features: usize) -> Self {
        let pixels = (width as usize) * (height as usize);
        let cell_size = min_distance.max(1);
        let grid_width = width.div_ceil(cell_size);
        let grid_height = height.div_ceil(cell_size);
        let cells = (grid_width * grid_height) as usize;

        Self {
            gx: ImageBuffer::new(width, height),
            gy: ImageBuffer::new(width, height),
            ix_sq: ImageBuffer::new(width, height),
            iy_sq: ImageBuffer::new(width, height),
            ix_iy: ImageBuffer::new(width, height),
            features: Vec::with_capacity(pixels),
            is_local_max: vec![false; pixels],
            grid: vec![None; cells],
        }
    }

    /// Resize the per-pixel scratch planes when the detection view's
    /// dimensions change (e.g. AOI-bounded detection with a moving rect, or
    /// switching between full-frame and rect-bounded detection). Calls with
    /// stable dimensions allocate nothing.
    ///
    /// Grow-only: each plane's *backing storage* keeps the high-water-mark
    /// pixel count it has ever been sized to (the same pattern already used
    /// for `is_local_max` / `grid` below, which reuse capacity via
    /// `Vec::resize`). A dims change only forces a fresh heap allocation the
    /// first time it needs more pixels than any previous call — e.g. once
    /// `detect_features` (full-frame) has run, every subsequent
    /// `detect_features_in_rect` call (rect ⊆ frame) reuses that same
    /// allocation for the rest of the tracker's lifetime, however many
    /// different rect sizes the AOI takes on. The resulting buffer's
    /// declared dimensions still match the request exactly, so every
    /// downstream kernel (which iterates the plane's full
    /// `width × height`) sees precisely the same view it always did —
    /// bit-identical to always allocating fresh.
    fn ensure_dims(&mut self, width: u32, height: u32) {
        if self.gx.dimensions() == (width, height) {
            return;
        }
        resize_scratch_plane(&mut self.gx, width, height);
        resize_scratch_plane(&mut self.gy, width, height);
        resize_scratch_plane(&mut self.ix_sq, width, height);
        resize_scratch_plane(&mut self.iy_sq, width, height);
        resize_scratch_plane(&mut self.ix_iy, width, height);
    }

    /// Detect Shi-Tomasi features on `image`, writing up to `max_features`
    /// `Feature` entries into `out` (cleared first), in descending quality
    /// order. No heap allocation when the buffer was sized for the same
    /// resolution and `min_distance`, and `out` has enough capacity.
    ///
    /// `image` may be any supported view, including a strided subrect of a
    /// larger frame — quality thresholding, the min-distance grid, and the
    /// `max_features` budget are all local to the view.
    pub(crate) fn detect_into(
        &mut self,
        image: &FlatSamples<&[u8]>,
        quality_level: f32,
        min_distance: u32,
        max_features: usize,
        out: &mut Vec<Feature>,
    ) {
        let width = image.layout.width;
        let height = image.layout.height;
        self.ensure_dims(width, height);

        compute_gradients_into(image, &mut self.gx, &mut self.gy);
        compute_gradient_products_into(&self.gx, &self.gy, &mut self.ix_sq, &mut self.iy_sq, &mut self.ix_iy);

        box_filter_3x3_in_place(&mut self.ix_sq);
        box_filter_3x3_in_place(&mut self.iy_sq);
        box_filter_3x3_in_place(&mut self.ix_iy);

        compute_min_eigenvalues_into(&self.ix_sq, &self.iy_sq, &self.ix_iy, &mut self.features);
        non_maximum_suppression(&mut self.features, width, height, &mut self.is_local_max);
        filter_by_quality(&mut self.features, quality_level);

        self.features.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        filter_by_distance_into(
            &self.features,
            min_distance,
            width,
            height,
            max_features,
            &mut self.grid,
            out,
        );
    }
}

/// Finds good feature points using the Shi-Tomasi algorithm.
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `quality_level` - Quality level. 0.4 is a good value
/// * `min_distance` - Filter points by distance between
///
/// # Returns
/// Vector of features with eigenvalue. Points sorted in descending order of quality.
///
/// # Deprecated
/// Use [`OpticalFlowBuilder`](crate::OpticalFlowBuilder) +
/// [`OpticalFlowTracker::detect_features`](crate::OpticalFlowTracker::detect_features),
/// which reuses pre-allocated detection buffers across calls.
#[deprecated(
    since = "0.4.0",
    note = "use OpticalFlowBuilder + OpticalFlowTracker::detect_features"
)]
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    let (w, h) = image.dimensions();
    let max = (w as usize) * (h as usize);
    let mut buf = FeaturesBuffer::with_capacity(w, h, min_distance, max);
    let mut out: Vec<Feature> = Vec::with_capacity(max);
    buf.detect_into(&image.as_flat_samples(), quality_level, min_distance, max, &mut out);
    out.into_iter().map(|f| (f.x as u32, f.y as u32, f.strength)).collect()
}

/// Resize `buf` to `(width, height)`, reusing its existing backing `Vec`
/// when it already has enough capacity instead of always allocating a fresh
/// one (what plain `ImageBuffer::new` does on every call, even when
/// shrinking). `Vec::resize` only reallocates when growing past current
/// capacity, so repeated calls at-or-below any previously-seen pixel count
/// are heap-allocation-free. Declared dimensions always end up exactly
/// `(width, height)` — identical to a fresh `ImageBuffer::new(width,
/// height)` from the caller's point of view.
///
/// INVARIANT: `Vec::resize` zero-fills only the *grown tail*; on a
/// shrink-then-grow (or grow past a smaller previous size) the retained
/// prefix keeps values from a prior detection. This is only bit-identical
/// to a fresh buffer because every `detect_into` kernel FULLY OVERWRITES
/// each pixel it later reads (gradients, products, box filter, eigenvalues
/// all assign, never `+=` into an assumed-zero interior, and NMS/quality
/// read only written pixels). A future kernel that reads an un-written
/// pixel expecting zero would see stale data — pinned today by
/// `detect_after_dims_churn_matches_fresh_buffer`.
fn resize_scratch_plane(buf: &mut ImageBuffer<Luma<i16>, Vec<i16>>, width: u32, height: u32) {
    let needed = (width as usize) * (height as usize);
    let mut raw = std::mem::replace(buf, ImageBuffer::new(0, 0)).into_raw();
    raw.resize(needed, 0);
    *buf = ImageBuffer::from_raw(width, height, raw)
        .expect("resized buffer length matches width * height");
}

fn compute_gradient_products_into(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
    ix_sq: &mut ImageBuffer<Luma<i16>, Vec<i16>>,
    iy_sq: &mut ImageBuffer<Luma<i16>, Vec<i16>>,
    ix_iy: &mut ImageBuffer<Luma<i16>, Vec<i16>>,
) {
    debug_assert_eq!(gx.dimensions(), ix_sq.dimensions());
    debug_assert_eq!(gx.dimensions(), iy_sq.dimensions());
    debug_assert_eq!(gx.dimensions(), ix_iy.dimensions());

    for ((x, y, gx_val), gy_val) in gx.enumerate_pixels().zip(gy.pixels()) {
        let ix = gx_val[0];
        let iy = gy_val[0];

        ix_sq.put_pixel(x, y, Luma([(ix / 32 * (ix / 32))]));
        iy_sq.put_pixel(x, y, Luma([(iy / 32 * (iy / 32))]));
        ix_iy.put_pixel(x, y, Luma([(ix / 32 * (iy / 32))]));
    }
}

fn compute_min_eigenvalues_into(
    a: &ImageBuffer<Luma<i16>, Vec<i16>>,
    b: &ImageBuffer<Luma<i16>, Vec<i16>>,
    c: &ImageBuffer<Luma<i16>, Vec<i16>>,
    out: &mut Vec<(u32, u32, f32)>,
) {
    out.clear();

    for y in 0..a.height() {
        for x in 0..a.width() {
            let a_val = a.get_pixel(x, y)[0] as i32;
            let b_val = b.get_pixel(x, y)[0] as i32;
            let c_val = c.get_pixel(x, y)[0] as i32;

            let trace = a_val + b_val;
            let discriminant = (a_val - b_val).pow(2) + 4 * c_val.pow(2);
            let min_eigen = (trace as f32 - (discriminant as f32).sqrt()) / 2.0;

            out.push((x, y, min_eigen));
        }
    }
}

fn non_maximum_suppression(
    features: &mut Vec<(u32, u32, f32)>,
    width: u32,
    height: u32,
    is_local_max: &mut Vec<bool>,
) {
    is_local_max.clear();
    is_local_max.resize(features.len(), false);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let current = features[idx].2;

            let mut is_max = true;
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let neighbor_idx = (ny as u32 * width + nx as u32) as usize;
                    if features[neighbor_idx].2 > current {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            is_local_max[idx] = is_max;
        }
    }

    let mut write = 0;
    for read in 0..features.len() {
        let (x, y, _) = features[read];
        let idx = (y * width + x) as usize;
        if is_local_max[idx] {
            features.swap(write, read);
            write += 1;
        }
    }
    features.truncate(write);
}

fn filter_by_quality(features: &mut Vec<(u32, u32, f32)>, quality_level: f32) {
    let max_quality = features
        .iter()
        .map(|&(_, _, q)| q)
        .fold(0.0f32, |a, b| a.max(b));
    let threshold = quality_level * max_quality;
    features.retain(|&(_, _, q)| q >= threshold);
}

fn filter_by_distance_into(
    features: &[(u32, u32, f32)],
    min_distance: u32,
    width: u32,
    height: u32,
    max_features: usize,
    grid: &mut Vec<Option<(u32, u32)>>,
    out: &mut Vec<Feature>,
) {
    let cell_size = min_distance.max(1);
    let grid_width = width.div_ceil(cell_size);
    let grid_height = height.div_ceil(cell_size);
    let cells = (grid_width * grid_height) as usize;

    grid.clear();
    grid.resize(cells, None);
    out.clear();

    let min_dist_sq = (min_distance * min_distance) as i32;

    for &(x, y, q) in features {
        if out.len() >= max_features {
            break;
        }

        let cell_x = x / cell_size;
        let cell_y = y / cell_size;
        let mut too_close = false;

        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                let check_x = cell_x as i32 + dx;
                let check_y = cell_y as i32 + dy;

                if check_x < 0
                    || check_y < 0
                    || check_x >= grid_width as i32
                    || check_y >= grid_height as i32
                {
                    continue;
                }

                let cell_idx = (check_y as u32 * grid_width + check_x as u32) as usize;
                if let Some((px, py)) = grid[cell_idx] {
                    let dist_sq: i32 =
                        (x as i32 - px as i32).pow(2) + (y as i32 - py as i32).pow(2);
                    if dist_sq < min_dist_sq {
                        too_close = true;
                        break;
                    }
                }
            }
            if too_close {
                break;
            }
        }

        if !too_close {
            let cell_idx = (cell_y * grid_width + cell_x) as usize;
            grid[cell_idx] = Some((x, y));
            out.push(Feature { x: x as f32, y: y as f32, strength: q });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(width: u32, height: u32, phase: u32) -> GrayImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 47 + y * 31 + phase * 19 + (x ^ y) * 5 + (x * y) % 13) & 0xff) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }
        img
    }

    fn detect(img: &GrayImage, quality: f32, min_distance: u32, max_features: usize) -> Vec<Feature> {
        let mut buf = FeaturesBuffer::with_capacity(img.width(), img.height(), min_distance, max_features);
        let mut out = Vec::new();
        buf.detect_into(&img.as_flat_samples(), quality, min_distance, max_features, &mut out);
        out
    }

    /// A4 pin: the grow-only `ensure_dims` (backing-storage reuse across
    /// dimension changes) must not change detection results. Run the same
    /// buffer through several dimension changes — growing past its initial
    /// capacity, then shrinking below it, then back to the very first
    /// image — and check that final run against a brand new
    /// `FeaturesBuffer` that only ever sees that one image (so its planes
    /// are freshly allocated at exactly that size, never resized).
    #[test]
    fn detect_after_dims_churn_matches_fresh_buffer() {
        const QUALITY: f32 = 0.1;
        const MIN_DIST: u32 = 4;
        const MAX_FEATURES: usize = 64;

        let rect_image = textured(40, 40, 0);
        let bigger_image = textured(96, 80, 1);
        let smaller_image = textured(20, 30, 2);

        let mut buf = FeaturesBuffer::with_capacity(40, 40, MIN_DIST, MAX_FEATURES);
        let mut out = Vec::new();

        // Baseline run at the buffer's initial dims.
        buf.detect_into(&rect_image.as_flat_samples(), QUALITY, MIN_DIST, MAX_FEATURES, &mut out);
        let baseline = out.clone();
        assert!(!baseline.is_empty(), "textured image should yield corners");

        // Grow past initial capacity, then shrink below it, then bounce
        // between a couple of other sizes to churn the backing storage.
        buf.detect_into(&bigger_image.as_flat_samples(), QUALITY, MIN_DIST, MAX_FEATURES, &mut out);
        buf.detect_into(&smaller_image.as_flat_samples(), QUALITY, MIN_DIST, MAX_FEATURES, &mut out);
        buf.detect_into(&bigger_image.as_flat_samples(), QUALITY, MIN_DIST, MAX_FEATURES, &mut out);
        buf.detect_into(&smaller_image.as_flat_samples(), QUALITY, MIN_DIST, MAX_FEATURES, &mut out);

        // Detect on the original rect/content again, after all that churn.
        buf.detect_into(&rect_image.as_flat_samples(), QUALITY, MIN_DIST, MAX_FEATURES, &mut out);

        // Compare against a fresh buffer that only ever sees this image.
        let fresh = detect(&rect_image, QUALITY, MIN_DIST, MAX_FEATURES);

        assert_eq!(out.len(), baseline.len());
        assert_eq!(out.len(), fresh.len());
        for ((churned, base), fresh) in out.iter().zip(baseline.iter()).zip(fresh.iter()) {
            assert_eq!(churned.x, base.x);
            assert_eq!(churned.y, base.y);
            assert_eq!(churned.strength, base.strength);
            assert_eq!(churned.x, fresh.x);
            assert_eq!(churned.y, fresh.y);
            assert_eq!(churned.strength, fresh.strength);
        }
    }
}
