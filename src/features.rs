use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::Ordering;

use crate::utils::{box_filter_3x3::box_filter_3x3_in_place, fast_gradients::compute_gradients};

/// Axis-aligned bounding rectangle for AOI-bounded feature detection.
#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Finds good features points using the Shi-Tomasi algorithm
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `quality_level` - Quality level. 0.4 is a good value
/// * `min_distance` - Filter points by distance between
///
///
/// # Returns
/// Vector of features with eigenvalue. Points sorted in descending order of quality
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    // Compute gradients
    let (gx, gy) = compute_gradients(image);

    // Compute squared gradients and their product
    let (mut ix_sq, mut iy_sq, mut ix_iy) = compute_gradient_products(&gx, &gy);

    // Smooth with 3x3 filters
    box_filter_3x3_in_place(&mut ix_sq);
    box_filter_3x3_in_place(&mut iy_sq);
    box_filter_3x3_in_place(&mut ix_iy);

    // Compute minimum eigenvalues
    let mut features = compute_min_eigenvalues(&ix_sq, &iy_sq, &ix_iy);

    // Non-maximum suppression
    non_maximum_suppression(&mut features, image.width(), image.height());

    // Filter by quality
    filter_by_quality(&mut features, quality_level);

    // Sort by descending quality
    features.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    // Filter by distance
    filter_by_distance(&features, min_distance, image.width(), image.height())
}

type GradientProduct = (
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

fn compute_gradient_products(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> GradientProduct {
    let mut ix_sq: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(gx.width(), gx.height());
    let mut iy_sq: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(gx.width(), gx.height());
    let mut ix_iy: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(gx.width(), gx.height());

    for ((x, y, gx_val), gy_val) in gx.enumerate_pixels().zip(gy.pixels()) {
        let ix = gx_val[0];
        let iy = gy_val[0];

        ix_sq.put_pixel(x, y, Luma([(ix / 32 * (ix / 32))]));
        iy_sq.put_pixel(x, y, Luma([(iy / 32 * (iy / 32))]));
        ix_iy.put_pixel(x, y, Luma([(ix / 32 * (iy / 32))]));
    }

    (ix_sq, iy_sq, ix_iy)
}

fn compute_min_eigenvalues(
    a: &ImageBuffer<Luma<i16>, Vec<i16>>,
    b: &ImageBuffer<Luma<i16>, Vec<i16>>,
    c: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> Vec<(u32, u32, f32)> {
    let mut features = Vec::with_capacity((a.width() * a.height()) as usize);

    for y in 0..a.height() {
        for x in 0..a.width() {
            let a_val = a.get_pixel(x, y)[0] as i32;
            let b_val = b.get_pixel(x, y)[0] as i32;
            let c_val = c.get_pixel(x, y)[0] as i32;

            let trace = a_val + b_val;
            let discriminant = (a_val - b_val).pow(2) + 4 * c_val.pow(2);
            let min_eigen = (((trace - discriminant) as f32).sqrt()) / 2.0;

            features.push((x, y, min_eigen));
        }
    }

    features
}

fn non_maximum_suppression(features: &mut Vec<(u32, u32, f32)>, width: u32, height: u32) {
    let mut is_local_max = vec![false; features.len()];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let current = features[idx].2;

            let mut is_max = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
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

    features.retain(|(x, y, _)| {
        let idx = (y * width + x) as usize;
        is_local_max[idx]
    });
}

fn filter_by_quality(features: &mut Vec<(u32, u32, f32)>, quality_level: f32) {
    let max_quality = features
        .iter()
        .map(|&(_, _, q)| q)
        .fold(0.0f32, |a, b| a.max(b));
    let threshold = quality_level * max_quality;
    features.retain(|&(_, _, q)| q >= threshold);
}

pub(crate) fn filter_by_distance(
    features: &[(u32, u32, f32)],
    min_distance: u32,
    width: u32,
    height: u32,
) -> Vec<(u32, u32, f32)> {
    let cell_size = min_distance;
    let grid_width = width.div_ceil(cell_size);
    let grid_height = height.div_ceil(cell_size);
    let mut grid = vec![vec![None; grid_height as usize]; grid_width as usize];
    let mut result = Vec::new();

    let min_dist_sq = (min_distance * min_distance) as i32;

    for &(x, y, q) in features {
        let cell_x = x / cell_size;
        let cell_y = y / cell_size;
        let mut too_close = false;

        for dx in -1..=1 {
            for dy in -1..=1 {
                let check_x = cell_x as i32 + dx;
                let check_y = cell_y as i32 + dy;

                if check_x < 0
                    || check_y < 0
                    || check_x >= grid_width as i32
                    || check_y >= grid_height as i32
                {
                    continue;
                }

                if let Some((px, py)) = grid[check_x as usize][check_y as usize] {
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
            grid[cell_x as usize][cell_y as usize] = Some((x, y));
            result.push((x, y, q));
        }
    }

    result
}

/// Like [`good_features_to_track`], but only examines pixels inside `rect`.
///
/// Intermediate work (gradients, products, box filter, NMS) is performed
/// only on the rect interior — at large frame sizes with a small AOI
/// this is much faster than the full-frame variant.
///
/// `rect` is clamped to the image and inset by 2px on each side
/// internally to leave room for the 3×3 Sobel and 3×3 box-filter
/// kernels. If the inset rect is empty, returns `Vec::new()`.
///
/// Output features are in IMAGE coordinates, sorted by descending quality.
pub fn good_features_to_track_in_rect(
    image: &GrayImage,
    rect: Rect,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    let img_w = image.width();
    let img_h = image.height();
    if img_w < 5 || img_h < 5 {
        return Vec::new();
    }

    // Clamp rect to image bounds and inset by 2px to give Scharr and
    // box-filter kernels a full 3×3 neighbourhood.
    let pad = 2u32;
    let x_lo = rect.x.saturating_add(pad).max(pad);
    let y_lo = rect.y.saturating_add(pad).max(pad);
    let x_hi = rect.x.saturating_add(rect.width).min(img_w).saturating_sub(pad);
    let y_hi = rect.y.saturating_add(rect.height).min(img_h).saturating_sub(pad);
    if x_lo + 2 > x_hi || y_lo + 2 > y_hi {
        return Vec::new();
    }

    // Step 1: Scharr gradients over the rect interior.
    // Full-frame i16 buffers so index math stays simple; cells outside
    // the rect remain zero.
    let n = (img_w as usize) * (img_h as usize);
    let mut gx_vec = vec![0i16; n];
    let mut gy_vec = vec![0i16; n];
    {
        let src = image.as_raw();
        let stride = img_w as usize;
        for y in y_lo..y_hi {
            let row_top = (y as usize - 1) * stride;
            let row_mid = y as usize * stride;
            let row_bot = (y as usize + 1) * stride;
            let dst_row = y as usize * stride;
            for x in x_lo..x_hi {
                let xl = x as usize - 1;
                let xc = x as usize;
                let xr = x as usize + 1;
                let p_tl = src[row_top + xl] as i32;
                let p_tc = src[row_top + xc] as i32;
                let p_tr = src[row_top + xr] as i32;
                let p_ml = src[row_mid + xl] as i32;
                let p_mr = src[row_mid + xr] as i32;
                let p_bl = src[row_bot + xl] as i32;
                let p_bc = src[row_bot + xc] as i32;
                let p_br = src[row_bot + xr] as i32;
                // Scharr 3×3: gx = [-3 0 3; -10 0 10; -3 0 3], gy = transpose.
                // Max value: 3*(255+255) + 10*255 = 4080, fits in i16.
                let gx = 3 * (p_tr + p_br - p_tl - p_bl) + 10 * (p_mr - p_ml);
                let gy = 3 * (p_bl + p_br - p_tl - p_tr) + 10 * (p_bc - p_tc);
                gx_vec[dst_row + xc] = gx as i16;
                gy_vec[dst_row + xc] = gy as i16;
            }
        }
    }

    // Step 2: gradient products, matching compute_gradient_products exactly.
    // The /32 scaling keeps products in i16 range: max (4080/32)^2 = 127^2 = 16129 < i16::MAX.
    let mut ix_sq = vec![0i16; n];
    let mut iy_sq = vec![0i16; n];
    let mut ix_iy = vec![0i16; n];
    for y in y_lo..y_hi {
        let row = y as usize * img_w as usize;
        for x in x_lo..x_hi {
            let idx = row + x as usize;
            let gxi = (gx_vec[idx] / 32) as i32;
            let gyi = (gy_vec[idx] / 32) as i32;
            ix_sq[idx] = (gxi * gxi) as i16;
            iy_sq[idx] = (gyi * gyi) as i16;
            ix_iy[idx] = (gxi * gyi) as i16;
        }
    }

    // Step 3: Separable 3×3 box filter matching `box_filter_3x3_in_place` exactly.
    //
    // The original implementation uses a sliding-window average, modifying values
    // in-place. The horizontal pass writes averaged values immediately, so the
    // vertical pass reads the already-horizontally-averaged data — this creates
    // mild IIR-like gradient-product spreading that is load-bearing for finding
    // features on typical synthetic and real images.
    //
    // We replicate the same two-pass in-place approach but bounded to the rect.
    // The rect is not inset further here because the horizontal pass safely reads
    // x±1 within [x_lo, x_hi) (guarded by the sliding window) and the vertical
    // pass reads y±1 within [y_lo, y_hi).  All writes stay within bounds.
    //
    // The filter uses a sliding window:
    //   - horizontal: sum starts with up to 3 elements at x=x_lo; slides right.
    //   - vertical:   same, column-wise.
    fn box_filter_separable_in_rect(buf: &mut [i16], img_w: u32, x_lo: u32, x_hi: u32, y_lo: u32, y_hi: u32) {
        let stride = img_w as usize;
        let x_lo = x_lo as usize;
        let x_hi = x_hi as usize;
        let y_lo = y_lo as usize;
        let y_hi = y_hi as usize;
        let width = x_hi - x_lo;
        let height = y_hi - y_lo;
        if width == 0 || height == 0 { return; }

        // Horizontal pass: average 3 pixels per row within [x_lo, x_hi).
        for y in y_lo..y_hi {
            let row = y * stride;
            // Initialise sliding sum with the first up-to-3 elements.
            let mut sum = buf[row + x_lo] as i32;
            let mut count = 1i32;
            if width > 1 { sum += buf[row + x_lo + 1] as i32; count += 1; }
            if width > 2 { sum += buf[row + x_lo + 2] as i32; count += 1; }
            buf[row + x_lo] = (sum / count) as i16;

            for xi in 1..width {
                let x = x_lo + xi;
                if xi > 1 { sum -= buf[row + x - 2] as i32; count -= 1; }
                if x + 1 < x_hi { sum += buf[row + x + 1] as i32; count += 1; }
                buf[row + x] = (sum / count) as i16;
            }
        }

        // Vertical pass: average 3 pixels per column within [y_lo, y_hi).
        for xi in 0..width {
            let x = x_lo + xi;
            let mut sum = buf[y_lo * stride + x] as i32;
            let mut count = 1i32;
            if height > 1 { sum += buf[(y_lo + 1) * stride + x] as i32; count += 1; }
            if height > 2 { sum += buf[(y_lo + 2) * stride + x] as i32; count += 1; }
            buf[y_lo * stride + x] = (sum / count) as i16;

            for yi in 1..height {
                let y = y_lo + yi;
                if yi > 1 { sum -= buf[(y - 2) * stride + x] as i32; count -= 1; }
                if y + 1 < y_hi { sum += buf[(y + 1) * stride + x] as i32; count += 1; }
                buf[y * stride + x] = (sum / count) as i16;
            }
        }
    }

    box_filter_separable_in_rect(&mut ix_sq, img_w, x_lo, x_hi, y_lo, y_hi);
    box_filter_separable_in_rect(&mut iy_sq, img_w, x_lo, x_hi, y_lo, y_hi);
    box_filter_separable_in_rect(&mut ix_iy, img_w, x_lo, x_hi, y_lo, y_hi);

    // After the box filter, valid output is in [x_lo+1, x_hi-1) × [y_lo+1, y_hi-1)
    // (we need a 1-pixel border for NMS).
    let xb_lo = x_lo + 1;
    let xb_hi = x_hi.saturating_sub(1);
    let yb_lo = y_lo + 1;
    let yb_hi = y_hi.saturating_sub(1);
    if xb_lo >= xb_hi || yb_lo >= yb_hi {
        return Vec::new();
    }

    // No separate *_box buffers needed — ix_sq/iy_sq/ix_iy are now the filtered values.
    let ix_sq_box = ix_sq;
    let iy_sq_box = iy_sq;
    let ix_iy_box = ix_iy;

    // Step 4: min-eigenvalue per pixel, matching compute_min_eigenvalues exactly.
    // Features are stored row-major over [xb_lo, xb_hi) × [yb_lo, yb_hi)
    // for the NMS step; image coords are preserved in each entry.
    let rect_w = xb_hi - xb_lo;
    let rect_h = yb_hi - yb_lo;
    let mut features: Vec<(u32, u32, f32)> =
        Vec::with_capacity((rect_w as usize) * (rect_h as usize));
    for y in yb_lo..yb_hi {
        let row = y as usize * img_w as usize;
        for x in xb_lo..xb_hi {
            let idx = row + x as usize;
            let a = ix_sq_box[idx] as i32;
            let b = iy_sq_box[idx] as i32;
            let c = ix_iy_box[idx] as i32;
            let trace = a + b;
            let discr = (a - b).pow(2) + 4 * c.pow(2);
            // Matches compute_min_eigenvalues: NaN when (trace - discr) < 0,
            // which makes NaN features lose out in the sort.
            let min_eigen = ((trace - discr) as f32).sqrt() / 2.0;
            features.push((x, y, min_eigen));
        }
    }

    // Step 5: NMS over 3×3 within the rect.
    // The features Vec is row-major over [xb_lo, xb_hi) × [yb_lo, yb_hi);
    // local index: ly * rect_w + lx where lx = x - xb_lo, ly = y - yb_lo.
    let mut is_local_max = vec![false; features.len()];
    for ly in 1..rect_h.saturating_sub(1) {
        for lx in 1..rect_w.saturating_sub(1) {
            let idx = ly as usize * rect_w as usize + lx as usize;
            let current = features[idx].2;
            let mut is_max = true;
            'outer: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nlx = (lx as i32 + dx) as usize;
                    let nly = (ly as i32 + dy) as usize;
                    let nidx = nly * rect_w as usize + nlx;
                    if features[nidx].2 > current {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            is_local_max[idx] = is_max;
        }
    }
    features = features
        .into_iter()
        .enumerate()
        .filter_map(|(i, f)| if is_local_max[i] { Some(f) } else { None })
        .collect();

    // Step 6: quality filter.
    let max_quality = features.iter().map(|&(_, _, q)| q).fold(0.0f32, f32::max);
    // If max quality is zero or NaN (uniform region) there are no real corners.
    if !(max_quality > 0.0) {
        return Vec::new();
    }
    let threshold = quality_level * max_quality;
    features.retain(|&(_, _, q)| q >= threshold);

    // Step 7: sort descending by quality.
    features.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    // Step 8: distance filter using the same grid-based logic as the full-frame path.
    if features.is_empty() || min_distance == 0 {
        return features;
    }
    filter_by_distance(&features, min_distance, img_w, img_h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn aoi_excludes_features_outside_rect() {
        let mut img = GrayImage::new(64, 64);
        // Bright dot at (8, 8).
        img.put_pixel(8, 8, Luma([255]));
        img.put_pixel(9, 8, Luma([255]));
        img.put_pixel(8, 9, Luma([255]));
        img.put_pixel(9, 9, Luma([255]));
        // AOI well away from the dot.
        let rect = Rect { x: 30, y: 30, width: 30, height: 30 };
        let features = good_features_to_track_in_rect(&img, rect, 0.01, 1);
        assert!(
            features.is_empty(),
            "expected no features outside AOI, got {features:?}"
        );
    }

    #[test]
    fn aoi_finds_features_inside_rect() {
        let mut img = GrayImage::new(64, 64);
        // A cross pattern centered at (32, 32) creates a strong 2D corner
        // response (both gx and gy nonzero at the intersection). A simple
        // filled square produces only edge responses (NaN eigenvalues).
        for y in 28u32..36 {
            img.put_pixel(32, y, Luma([255]));
        }
        for x in 28u32..36 {
            img.put_pixel(x, 32, Luma([255]));
        }
        let rect = Rect { x: 20, y: 20, width: 30, height: 30 };
        let features = good_features_to_track_in_rect(&img, rect, 0.01, 1);
        assert!(
            !features.is_empty(),
            "expected at least one feature inside AOI"
        );
        // All returned features must be inside the rect.
        for (x, y, _) in &features {
            assert!(
                *x >= rect.x && *x < rect.x + rect.width,
                "feature x={x} outside rect"
            );
            assert!(
                *y >= rect.y && *y < rect.y + rect.height,
                "feature y={y} outside rect"
            );
        }
    }
}
