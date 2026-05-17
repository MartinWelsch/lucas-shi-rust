use image::{GrayImage, ImageBuffer, Luma};

use crate::utils::fast_gradients::compute_gradients_into;

/// Reusable storage for the Lucas-Kanade tracking pipeline. Pre-allocates
/// every per-frame buffer at construction; `calc_into` reuses them.
pub(crate) struct LkBuffer {
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
    pub(crate) fn with_capacity(
        width: u32,
        height: u32,
        levels: usize,
        window_size: usize,
    ) -> Self {
        assert!(window_size % 2 == 1, "Window size must be odd");
        let mut grad_x = Vec::with_capacity(levels);
        let mut grad_y = Vec::with_capacity(levels);
        let mut w = width;
        let mut h = height;
        for level in 0..levels {
            grad_x.push(ImageBuffer::new(w, h));
            grad_y.push(ImageBuffer::new(w, h));
            if level + 1 < levels {
                if w < 2 || h < 2 {
                    break;
                }
                w /= 2;
                h /= 2;
            }
        }
        let n_pixels = window_size * window_size;
        let radius = window_size / 2;
        Self {
            grad_x,
            grad_y,
            prev_patch: vec![0.0; n_pixels],
            ix_patch: vec![0.0; n_pixels],
            iy_patch: vec![0.0; n_pixels],
            displacements: Vec::new(),
            offsets: build_window_offsets(radius),
            window_size,
        }
    }

    /// Compute flow from `prev_pyramid` to `curr_pyramid` for the points in
    /// `points`, writing the updated positions back in place. No heap
    /// allocation when the buffer was sized for the same parameters.
    pub(crate) fn calc_into(
        &mut self,
        prev_pyramid: &[GrayImage],
        curr_pyramid: &[GrayImage],
        points: &mut Vec<(f32, f32)>,
        max_iterations: usize,
    ) {
        assert_eq!(prev_pyramid.len(), curr_pyramid.len());

        let n_levels = prev_pyramid.len();
        let radius = self.window_size / 2;
        let epsilon = 1e-3;
        let det_epsilon = 1e-6;

        self.displacements.clear();
        self.displacements.resize(points.len(), (0.0, 0.0));

        for level in (0..n_levels).rev() {
            let scale = 2f32.powi(level as i32);
            let prev_img = &prev_pyramid[level];
            let curr_img = &curr_pyramid[level];

            compute_gradients_into(
                &prev_img.as_flat_samples(),
                &mut self.grad_x[level],
                &mut self.grad_y[level],
            );

            for ((prev_x, prev_y), disp) in points.iter().zip(self.displacements.iter_mut()) {
                let x = *prev_x / scale;
                let y = *prev_y / scale;
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
                    let ix = interpolate_alt(&self.grad_x[level], sample_x, sample_y) / 32.0;
                    let iy = interpolate_alt(&self.grad_y[level], sample_x, sample_y) / 32.0;

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

        for (pt, disp) in points.iter_mut().zip(self.displacements.iter()) {
            pt.0 += disp.0;
            pt.1 += disp.1;
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
pub fn calc_optical_flow(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    let (w, h) = prev_pyramid[0].dimensions();
    let levels = prev_pyramid.len();
    let mut buf = LkBuffer::with_capacity(w, h, levels, window_size);
    let mut points = prev_points.to_vec();
    buf.calc_into(
        prev_pyramid,
        curr_pyramid,
        &mut points,
        max_iterations,
    );
    points
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
fn interpolate(img: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);

        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };

        sum += px * wx * wy;
    }

    sum
}

fn interpolate_alt(img: &ImageBuffer<Luma<i16>, Vec<i16>>, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
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
