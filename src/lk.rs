use image::{GrayImage, ImageBuffer, Luma, Primitive};

use crate::feature::Feature;
use crate::pyramid::PyramidBuffer;
use crate::utils::fast_gradients::compute_gradients_into;

/// Reusable storage for the Lucas-Kanade tracking pipeline. Pre-allocates
/// every per-frame buffer at construction;
/// [`track`](crate::buffers::track) reuses them.
pub struct LkBuffer {
    grad_x: Vec<ImageBuffer<Luma<i16>, Vec<i16>>>,
    grad_y: Vec<ImageBuffer<Luma<i16>, Vec<i16>>>,
    prev_patch: Vec<f32>,
    ix_patch: Vec<f32>,
    iy_patch: Vec<f32>,
    displacements: Vec<(f32, f32)>,
    offsets: Vec<(f32, f32)>,
    window_size: usize,
}

impl LkBuffer {
    /// Pre-allocate gradient buffers for `levels` pyramid steps (level 0 at
    /// `(width, height)`, level 1 at `(W/2, H/2)`, …) and patch buffers sized
    /// to `window_size * window_size`. Panics if `window_size` is even.
    pub fn with_capacity(
        width: u32,
        height: u32,
        levels: usize,
        window_size: usize,
        max_features: usize,
    ) -> Self {
        assert!(window_size % 2 == 1, "Window size must be odd");
        let dims = crate::pyramid::pyramid_dims(width, height, levels);
        let grad_x: Vec<_> = dims.iter().map(|&(w, h)| ImageBuffer::new(w, h)).collect();
        let grad_y: Vec<_> = dims.iter().map(|&(w, h)| ImageBuffer::new(w, h)).collect();
        let n_pixels = window_size * window_size;
        let radius = window_size / 2;
        Self {
            grad_x,
            grad_y,
            prev_patch: vec![0.0; n_pixels],
            ix_patch: vec![0.0; n_pixels],
            iy_patch: vec![0.0; n_pixels],
            displacements: Vec::with_capacity(max_features),
            offsets: build_window_offsets(radius),
            window_size,
        }
    }

    /// Track `features` from `prev_pyramid` to `curr_pyramid` in place,
    /// updating each `Feature`'s `(x, y)` by the computed displacement.
    /// `strength` is left untouched. No heap allocation when the buffer was
    /// sized for the same parameters.
    ///
    /// Recomputes gradients for `prev_pyramid` on every call — kept only for
    /// the deprecated [`calc_optical_flow`] entry point, which is handed raw
    /// pyramid slices with no cached gradient planes. Steady-state tracking
    /// (`OpticalFlowTracker` and [`buffers::track`](crate::buffers::track))
    /// uses [`calc_into_cached`](Self::calc_into_cached) instead, which reads
    /// gradients [`PyramidBuffer`] already computed once at build time.
    pub(crate) fn calc_into(
        &mut self,
        prev_pyramid: &[GrayImage],
        curr_pyramid: &[GrayImage],
        features: &mut [Feature],
        max_iterations: usize,
    ) {
        self.calc_into_status(prev_pyramid, curr_pyramid, features, max_iterations, None);
    }

    /// Like [`calc_into`](Self::calc_into) but additionally reports, per
    /// feature, whether Lucas-Kanade actually produced a flow estimate for
    /// it. A feature is reported **invalid** (`false`) when it was *skipped*
    /// at the coarsest pyramid level — either its window fell outside the
    /// previous image (`in_bounds` failed) or its gradient Hessian was
    /// singular. Such a feature keeps displacement `(0, 0)` and is exactly
    /// the kind of "stuck" feature forward-backward validation must reject.
    /// All other features are reported valid (`true`); the caller is
    /// responsible for the round-trip error gate.
    ///
    /// `valid_out`, when `Some`, is cleared and resized to `features.len()`.
    /// Passing `None` skips status bookkeeping entirely (the plain
    /// `calc_into` path). Recomputes `prev_pyramid`'s gradients every call —
    /// see [`calc_into`](Self::calc_into) for why this legacy path still
    /// exists alongside [`calc_into_status_cached`](Self::calc_into_status_cached).
    pub(crate) fn calc_into_status(
        &mut self,
        prev_pyramid: &[GrayImage],
        curr_pyramid: &[GrayImage],
        features: &mut [Feature],
        max_iterations: usize,
        mut valid_out: Option<&mut Vec<bool>>,
    ) {
        assert_eq!(prev_pyramid.len(), curr_pyramid.len());

        let n_levels = prev_pyramid.len();
        let radius = self.window_size / 2;
        let coarsest = n_levels - 1;

        Self::reset_status(&mut self.displacements, valid_out.as_deref_mut(), features.len());

        for level in (0..n_levels).rev() {
            let scale = 2f32.powi(level as i32);
            let prev_img = &prev_pyramid[level];
            let curr_img = &curr_pyramid[level];

            compute_gradients_into(
                &prev_img.as_flat_samples(),
                &mut self.grad_x[level],
                &mut self.grad_y[level],
            );

            track_level(
                prev_img,
                curr_img,
                &self.grad_x[level],
                &self.grad_y[level],
                scale,
                radius,
                max_iterations,
                level == coarsest,
                features,
                &mut self.displacements,
                &self.offsets,
                &mut self.prev_patch,
                &mut self.ix_patch,
                &mut self.iy_patch,
                valid_out.as_deref_mut().map(|v| v.as_mut_slice()),
            );
        }

        Self::apply_displacements(features, &self.displacements);
    }

    /// Like [`calc_into`](Self::calc_into) but sources `prev_pyramid`'s
    /// gradients from its cache instead of recomputing them. Used by the
    /// steady-state tracking path (`OpticalFlowTracker` and
    /// [`buffers::track`](crate::buffers::track)), where each physical
    /// pyramid buffer's gradients are computed exactly once, in
    /// [`PyramidBuffer::build_into`], regardless of how many LK calls treat
    /// it as the "prev" role afterward.
    pub(crate) fn calc_into_cached(
        &mut self,
        prev_pyramid: &PyramidBuffer,
        curr_pyramid: &PyramidBuffer,
        features: &mut [Feature],
        max_iterations: usize,
    ) {
        self.calc_into_status_cached(prev_pyramid, curr_pyramid, features, max_iterations, None);
    }

    /// Cached-gradient counterpart of [`calc_into_status`](Self::calc_into_status);
    /// see that method for the validity-reporting contract and
    /// [`calc_into_cached`](Self::calc_into_cached) for why this variant
    /// exists.
    pub(crate) fn calc_into_status_cached(
        &mut self,
        prev_pyramid: &PyramidBuffer,
        curr_pyramid: &PyramidBuffer,
        features: &mut [Feature],
        max_iterations: usize,
        mut valid_out: Option<&mut Vec<bool>>,
    ) {
        let prev_levels = prev_pyramid.levels();
        let curr_levels = curr_pyramid.levels();
        assert_eq!(prev_levels.len(), curr_levels.len());

        let n_levels = prev_levels.len();
        let radius = self.window_size / 2;
        let coarsest = n_levels - 1;

        Self::reset_status(&mut self.displacements, valid_out.as_deref_mut(), features.len());

        for level in (0..n_levels).rev() {
            let scale = 2f32.powi(level as i32);
            let prev_img = &prev_levels[level];
            let curr_img = &curr_levels[level];
            let (grad_x, grad_y) = prev_pyramid.grad_planes(level);

            track_level(
                prev_img,
                curr_img,
                grad_x,
                grad_y,
                scale,
                radius,
                max_iterations,
                level == coarsest,
                features,
                &mut self.displacements,
                &self.offsets,
                &mut self.prev_patch,
                &mut self.ix_patch,
                &mut self.iy_patch,
                valid_out.as_deref_mut().map(|v| v.as_mut_slice()),
            );
        }

        Self::apply_displacements(features, &self.displacements);
    }

    /// Shared setup for both `calc_into_status*` entry points: clear/resize
    /// the displacement and validity scratch to `n_features`.
    fn reset_status(displacements: &mut Vec<(f32, f32)>, valid_out: Option<&mut Vec<bool>>, n_features: usize) {
        displacements.clear();
        displacements.resize(n_features, (0.0, 0.0));

        if let Some(valid) = valid_out {
            valid.clear();
            // A feature is valid only if it produced a flow estimate at the
            // coarsest level. Start all-false; the coarsest-level pass flips
            // surviving features to true.
            valid.resize(n_features, false);
        }
    }

    /// Shared teardown for both `calc_into_status*` entry points: apply the
    /// accumulated per-level displacement to each feature's position.
    fn apply_displacements(features: &mut [Feature], displacements: &[(f32, f32)]) {
        for (feat, disp) in features.iter_mut().zip(displacements.iter()) {
            feat.x += disp.0;
            feat.y += disp.1;
        }
    }
}

/// Track every feature through one pyramid level: build the per-feature
/// Hessian from the (cached or freshly computed) gradient planes, then run
/// the iterative Lucas-Kanade refinement against `curr_img`. Shared by
/// [`LkBuffer::calc_into_status`] (recomputed gradients) and
/// [`LkBuffer::calc_into_status_cached`] (`PyramidBuffer`-cached gradients)
/// so the hot inner loop — including [`interpolate`] — has a single
/// implementation.
#[allow(clippy::too_many_arguments)]
fn track_level(
    prev_img: &GrayImage,
    curr_img: &GrayImage,
    grad_x: &ImageBuffer<Luma<i16>, Vec<i16>>,
    grad_y: &ImageBuffer<Luma<i16>, Vec<i16>>,
    scale: f32,
    radius: usize,
    max_iterations: usize,
    is_coarsest: bool,
    features: &[Feature],
    displacements: &mut [(f32, f32)],
    offsets: &[(f32, f32)],
    prev_patch: &mut [f32],
    ix_patch: &mut [f32],
    iy_patch: &mut [f32],
    mut valid_out: Option<&mut [bool]>,
) {
    let epsilon = 1e-3;
    let det_epsilon = 1e-6;

    for (fi, (feat, disp)) in features.iter().zip(displacements.iter_mut()).enumerate() {
        let x = feat.x / scale;
        let y = feat.y / scale;
        let mut dx = disp.0 / scale;
        let mut dy = disp.1 / scale;

        if !in_bounds(prev_img, x, y, radius) {
            continue;
        }

        let mut gxx = 0.0f32;
        let mut gxy = 0.0f32;
        let mut gyy = 0.0f32;

        for (idx, (ox, oy)) in offsets.iter().enumerate() {
            let sample_x = x + ox;
            let sample_y = y + oy;
            let ix = interpolate(grad_x, sample_x, sample_y) / 32.0;
            let iy = interpolate(grad_y, sample_x, sample_y) / 32.0;

            prev_patch[idx] = interpolate(prev_img, sample_x, sample_y);
            ix_patch[idx] = ix;
            iy_patch[idx] = iy;
            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
        }

        let Some((inv_h00, inv_h01, inv_h11)) = invert_2x2(gxx, gxy, gyy, det_epsilon) else {
            continue;
        };

        // Reached tracking at the coarsest level ⇒ this feature has a
        // genuine flow estimate (not a stuck (0, 0)).
        if is_coarsest
            && let Some(valid) = valid_out.as_deref_mut()
        {
            valid[fi] = true;
        }

        for _ in 0..max_iterations {
            let curr_x = x + dx;
            let curr_y = y + dy;

            if !in_bounds(curr_img, curr_x, curr_y, radius) {
                break;
            }

            let mut bx = 0.0f32;
            let mut by = 0.0f32;

            for (idx, (ox, oy)) in offsets.iter().enumerate() {
                let curr = interpolate(curr_img, curr_x + ox, curr_y + oy);
                let error = prev_patch[idx] - curr;
                bx += ix_patch[idx] * error;
                by += iy_patch[idx] * error;
            }

            let ddx = inv_h00 * bx + inv_h01 * by;
            let ddy = inv_h01 * bx + inv_h11 * by;
            dx += ddx;
            dy += ddy;

            if ddx.abs() < epsilon && ddy.abs() < epsilon {
                break;
            }
        }

        *disp = (dx * scale, dy * scale);
    }
}

/// Compute optical flow using Lucas-Kanade method.
///
/// # Arguments
/// * `prev_pyramid` - Previous frame (pyramid of grayscale)
/// * `curr_pyramid` - Next frame (pyramid of grayscale)
/// * `prev_points` - Feature points to track (in prev frame)
/// * `window_size` - Size of the search window (odd number)
/// * `max_iterations` - Max iterations for correct points on each layer
///
/// # Returns
/// Vector of points on next frame.
///
/// # Deprecated
/// Use [`OpticalFlowBuilder`](crate::OpticalFlowBuilder) +
/// [`OpticalFlowTracker::push_frame`](crate::OpticalFlowTracker::push_frame),
/// which runs the same algorithm with reused buffers and accepts arbitrary
/// `FlatSamples` input (including subrects and NV12 Y planes).
#[deprecated(
    since = "0.4.0",
    note = "use OpticalFlowBuilder + OpticalFlowTracker::push_frame"
)]
pub fn calc_optical_flow(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    let (w, h) = prev_pyramid[0].dimensions();
    let levels = prev_pyramid.len();
    let mut buf = LkBuffer::with_capacity(w, h, levels, window_size, prev_points.len());
    let mut features: Vec<Feature> = prev_points
        .iter()
        .map(|&(x, y)| Feature { x, y, strength: 0.0 })
        .collect();
    buf.calc_into(prev_pyramid, curr_pyramid, &mut features, max_iterations);
    features.into_iter().map(|f| (f.x, f.y)).collect()
}

fn build_window_offsets(radius: usize) -> Vec<(f32, f32)> {
    let mut offsets = Vec::with_capacity((2 * radius + 1) * (2 * radius + 1));

    for j in -(radius as i32)..=radius as i32 {
        for i in -(radius as i32)..=radius as i32 {
            offsets.push((i as f32, j as f32));
        }
    }

    offsets
}

fn invert_2x2(a00: f32, a01: f32, a11: f32, det_epsilon: f32) -> Option<(f32, f32, f32)> {
    let det = a00 * a11 - a01 * a01;
    if det.abs() <= det_epsilon {
        return None;
    }

    let inv_det = 1.0 / det;
    Some((a11 * inv_det, -a01 * inv_det, a00 * inv_det))
}

/// Checks that the window stays within image bounds.
fn in_bounds(img: &GrayImage, x: f32, y: f32, radius: usize) -> bool {
    let (w, h) = (img.width() as f32, img.height() as f32);
    x >= radius as f32 && x < w - radius as f32 && y >= radius as f32 && y < h - radius as f32
}

/// Bilinear interpolation of the pixel value.
///
/// Weights are computed once from `(dx, dy)`. When the 2×2 sample cell is
/// fully in bounds, a single range check lets us read all four taps via raw
/// slice indexing (no per-tap `get_pixel_checked` + branchy weight select).
/// Otherwise we fall back to the original per-tap bounds-checked path,
/// preserving the existing out-of-bounds→0.0 convention (including the
/// wrinkle where an in-bounds *window* can still sample a tap exactly at
/// `x == width` / `y == height`, contributing 0.0 with nonzero weight).
///
/// BIT-EXACT: both paths accumulate `p * wx * wy` in the same left-to-right
/// order and in the same (x0,y0),(x0,y1),(x1,y0),(x1,y1) tap order as the
/// original implementation, so results are bit-identical to it (see the
/// `interpolate_matches_reference_*` tests below).
fn interpolate<P>(img: &ImageBuffer<Luma<P>, Vec<P>>, x: f32, y: f32) -> f32
where
    P: Primitive + Into<f32>,
{
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let dx1 = 1.0 - dx;
    let dy1 = 1.0 - dy;

    let width = img.width() as i32;
    let height = img.height() as i32;

    if x0 >= 0 && y0 >= 0 && x1 < width && y1 < height {
        // Fast path: the whole 2x2 cell is in bounds — single check, then
        // raw contiguous reads (image buffers are always width-stride here).
        let stride = width as usize;
        let raw = img.as_raw();
        let base = y0 as usize * stride + x0 as usize;
        let p00: f32 = raw[base].into();
        let p01: f32 = raw[base + stride].into();
        let p10: f32 = raw[base + 1].into();
        let p11: f32 = raw[base + stride + 1].into();

        let mut sum = 0.0f32;
        sum += p00 * dx1 * dy1;
        sum += p01 * dx1 * dy;
        sum += p10 * dx * dy1;
        sum += p11 * dx * dy;
        sum
    } else {
        // Slow path: original per-tap bounds-checked behavior.
        let mut sum = 0.0f32;
        for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
            let px = img
                .get_pixel_checked(*sx as u32, *sy as u32)
                .map(|p| p[0].into())
                .unwrap_or(0.0);

            let wx = if sx == &x0 { dx1 } else { dx };
            let wy = if sy == &y0 { dy1 } else { dy };

            sum += px * wx * wy;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::{interpolate, invert_2x2, LkBuffer};
    use image::{GrayImage, Luma};

    #[test]
    fn invert_2x2_returns_inverse_components() {
        let (inv00, inv01, inv11) = invert_2x2(4.0, 1.0, 3.0, 1e-6).unwrap();

        assert!((inv00 - 3.0 / 11.0).abs() < 1e-6);
        assert!((inv01 + 1.0 / 11.0).abs() < 1e-6);
        assert!((inv11 - 4.0 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn invert_2x2_rejects_singular_matrix() {
        assert!(invert_2x2(1.0, 2.0, 4.0, 1e-6).is_none());
    }

    /// Reference copy of the pre-A1 `interpolate` implementation (per-tap
    /// `get_pixel_checked` + branchy weight select). Kept only in test code
    /// so we can pin the optimized version against it.
    fn interpolate_reference<P>(img: &image::ImageBuffer<Luma<P>, Vec<P>>, x: f32, y: f32) -> f32
    where
        P: image::Primitive + Into<f32>,
    {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let mut sum = 0.0f32;
        for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
            let px = img
                .get_pixel_checked(*sx as u32, *sy as u32)
                .map(|p| p[0].into())
                .unwrap_or(0.0);

            let wx = if sx == &x0 { 1.0 - dx } else { dx };
            let wy = if sy == &y0 { 1.0 - dy } else { dy };

            sum += px * wx * wy;
        }

        sum
    }

    fn textured_image(width: u32, height: u32) -> GrayImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                // Non-trivial, non-symmetric texture so every tap combination
                // sees a distinct value.
                let v = ((x * 37 + y * 23 + (x ^ y) * 11 + (x * y) % 13) & 0xff) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }
        img
    }

    /// Dense grid of fractional positions, including cells whose 2x2 window
    /// touches every border (top-left corner through bottom-right corner,
    /// and one step further out so the out-of-bounds slow path is exercised
    /// too), asserting EXACT f32 equality against the reference.
    #[test]
    fn interpolate_matches_reference_dense_grid() {
        let img = textured_image(24, 20);
        let (w, h) = (img.width() as i32, img.height() as i32);

        // Cover x0/y0 from -2 (fully out of bounds) through width/height
        // (tap at the far edge, in-bounds window but zero-weighted OOB tap),
        // at a dense set of fractional offsets.
        let fracs: [f32; 7] = [0.0, 0.05, 0.25, 0.5, 0.5, 0.75, 0.999];
        for base_x in -2..=w {
            for base_y in -2..=h {
                for &fx in &fracs {
                    for &fy in &fracs {
                        let x = base_x as f32 + fx;
                        let y = base_y as f32 + fy;
                        let expected = interpolate_reference(&img, x, y);
                        let actual = interpolate(&img, x, y);
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "mismatch at x={x}, y={y}: actual={actual}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    /// Explicitly pin the four "cell touches this border" corners called out
    /// in the spec: top-left, top-right, bottom-left, bottom-right.
    #[test]
    fn interpolate_matches_reference_at_every_border() {
        let img = textured_image(10, 8);
        let (w, h) = (img.width() as f32, img.height() as f32);
        let positions = [
            (0.0, 0.0),           // top-left corner, cell fully in bounds
            (-0.5, -0.5),         // top-left, cell out of bounds
            (w - 1.0, 0.0),       // top-right, in bounds (tap at x == w-1)
            (w - 0.5, 0.0),       // top-right, tap lands at x == w (OOB tap, in-bounds window edge case)
            (0.0, h - 1.0),       // bottom-left, in bounds
            (0.0, h - 0.5),       // bottom-left, tap at y == h
            (w - 1.0, h - 1.0),   // bottom-right, in bounds
            (w - 0.5, h - 0.5),   // bottom-right, taps at x == w and y == h
            (w, h),               // fully out of bounds
        ];
        for (x, y) in positions {
            let expected = interpolate_reference(&img, x, y);
            let actual = interpolate(&img, x, y);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "mismatch at x={x}, y={y}: actual={actual}, expected={expected}"
            );
        }
    }

    /// A2 bit-exactness pin: `calc_into_status_cached` (reads Scharr
    /// gradients cached by `PyramidBuffer::build_into`) must produce
    /// identical results to `calc_into_status` (recomputes gradients from
    /// the raw level slices every call) — same gradient function, same
    /// pixel inputs, so the two paths must agree to the bit, in both the
    /// forward (`prev -> curr`) and backward (`curr -> prev`) roles used by
    /// `calculate_flow_fb`.
    #[test]
    fn calc_into_status_cached_matches_uncached_reference() {
        use crate::feature::Feature;
        use crate::pyramid::PyramidBuffer;

        const W: u32 = 64;
        const H: u32 = 64;
        const LEVELS: usize = 3;
        const WINDOW: usize = 11;
        const MAX_ITERS: usize = 14;

        fn textured(width: u32, height: u32, phase: u32) -> GrayImage {
            let mut img = GrayImage::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    let v = ((x * 41 + y * 17 + phase * 13 + (x ^ y) * 7 + (x * y) % 11) & 0xff) as u8;
                    img.put_pixel(x, y, Luma([v]));
                }
            }
            img
        }

        let prev_img = textured(W, H, 0);
        let curr_img = textured(W, H, 5);

        let mut prev_pb = PyramidBuffer::with_capacity(W, H, LEVELS);
        let mut curr_pb = PyramidBuffer::with_capacity(W, H, LEVELS);
        prev_pb.build_into(&prev_img.as_flat_samples());
        curr_pb.build_into(&curr_img.as_flat_samples());

        let starting_features: Vec<Feature> = (0..W).step_by(6).flat_map(|x| {
            (0..H).step_by(6).map(move |y| Feature { x: x as f32, y: y as f32, strength: 1.0 })
        }).collect();

        let mut buf = LkBuffer::with_capacity(W, H, LEVELS, WINDOW, starting_features.len());

        // Forward role: prev -> curr.
        let mut fwd_ref = starting_features.clone();
        let mut fwd_ref_valid = Vec::new();
        buf.calc_into_status(
            prev_pb.levels(),
            curr_pb.levels(),
            &mut fwd_ref,
            MAX_ITERS,
            Some(&mut fwd_ref_valid),
        );

        let mut fwd_cached = starting_features.clone();
        let mut fwd_cached_valid = Vec::new();
        buf.calc_into_status_cached(
            &prev_pb,
            &curr_pb,
            &mut fwd_cached,
            MAX_ITERS,
            Some(&mut fwd_cached_valid),
        );

        assert_eq!(fwd_ref_valid, fwd_cached_valid);
        for (r, c) in fwd_ref.iter().zip(fwd_cached.iter()) {
            assert_eq!(r.x.to_bits(), c.x.to_bits());
            assert_eq!(r.y.to_bits(), c.y.to_bits());
        }

        // Backward role: curr -> prev (the role `calculate_flow_fb`'s
        // second pass uses; exercises reading cached gradients off whatever
        // physical pyramid is passed as `prev_pyramid`, here `curr_pb`).
        let mut back_ref = starting_features.clone();
        let mut back_ref_valid = Vec::new();
        buf.calc_into_status(
            curr_pb.levels(),
            prev_pb.levels(),
            &mut back_ref,
            MAX_ITERS,
            Some(&mut back_ref_valid),
        );

        let mut back_cached = starting_features.clone();
        let mut back_cached_valid = Vec::new();
        buf.calc_into_status_cached(
            &curr_pb,
            &prev_pb,
            &mut back_cached,
            MAX_ITERS,
            Some(&mut back_cached_valid),
        );

        assert_eq!(back_ref_valid, back_cached_valid);
        for (r, c) in back_ref.iter().zip(back_cached.iter()) {
            assert_eq!(r.x.to_bits(), c.x.to_bits());
            assert_eq!(r.y.to_bits(), c.y.to_bits());
        }
    }
}
