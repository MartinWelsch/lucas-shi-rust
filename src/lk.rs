use image::{GrayImage, ImageBuffer, Luma, Primitive};

use crate::feature::Feature;
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
    /// `calc_into` path).
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
        let epsilon = 1e-3;
        let det_epsilon = 1e-6;

        self.displacements.clear();
        self.displacements.resize(features.len(), (0.0, 0.0));

        if let Some(valid) = valid_out.as_deref_mut() {
            valid.clear();
            // A feature is valid only if it produced a flow estimate at the
            // coarsest level. Start all-false; the coarsest-level pass flips
            // surviving features to true.
            valid.resize(features.len(), false);
        }
        // The coarsest level drives the validity decision: a feature skipped
        // there never enters tracking at all and stays at (0, 0).
        let coarsest = n_levels - 1;

        for level in (0..n_levels).rev() {
            let scale = 2f32.powi(level as i32);
            let prev_img = &prev_pyramid[level];
            let curr_img = &curr_pyramid[level];

            compute_gradients_into(
                &prev_img.as_flat_samples(),
                &mut self.grad_x[level],
                &mut self.grad_y[level],
            );

            for (fi, (feat, disp)) in features
                .iter()
                .zip(self.displacements.iter_mut())
                .enumerate()
            {
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

                for (idx, (ox, oy)) in self.offsets.iter().enumerate() {
                    let sample_x = x + ox;
                    let sample_y = y + oy;
                    let ix = interpolate(&self.grad_x[level], sample_x, sample_y) / 32.0;
                    let iy = interpolate(&self.grad_y[level], sample_x, sample_y) / 32.0;

                    self.prev_patch[idx] = interpolate(prev_img, sample_x, sample_y);
                    self.ix_patch[idx] = ix;
                    self.iy_patch[idx] = iy;
                    gxx += ix * ix;
                    gxy += ix * iy;
                    gyy += iy * iy;
                }

                let Some((inv_h00, inv_h01, inv_h11)) = invert_2x2(gxx, gxy, gyy, det_epsilon) else {
                    continue;
                };

                // Reached tracking at the coarsest level ⇒ this feature has a
                // genuine flow estimate (not a stuck (0, 0)).
                if level == coarsest
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

                    for (idx, (ox, oy)) in self.offsets.iter().enumerate() {
                        let curr = interpolate(curr_img, curr_x + ox, curr_y + oy);
                        let error = self.prev_patch[idx] - curr;
                        bx += self.ix_patch[idx] * error;
                        by += self.iy_patch[idx] * error;
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

        for (feat, disp) in features.iter_mut().zip(self.displacements.iter()) {
            feat.x += disp.0;
            feat.y += disp.1;
        }
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

#[cfg(test)]
mod tests {
    use super::invert_2x2;

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
}
