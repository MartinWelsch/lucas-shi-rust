use image::flat::FlatSamples;
use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::Ordering;

use crate::utils::{box_filter_3x3::box_filter_3x3_in_place, fast_gradients::compute_gradients_into};

/// Finds good features points using the Shi-Tomasi algorithm
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `quality_level` - Quality level. 0.4 is a good value
/// * `min_distance` - Filter points by distance between
///
/// # Returns
/// Vector of features with eigenvalue. Points sorted in descending order of quality
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    good_features_to_track_impl(&image.as_flat_samples(), quality_level, min_distance)
}

pub(crate) fn good_features_to_track_impl(
    image: &FlatSamples<&[u8]>,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    let width = image.layout.width;
    let height = image.layout.height;

    let mut gx: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    let mut gy: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    compute_gradients_into(image, &mut gx, &mut gy);

    let mut ix_sq: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    let mut iy_sq: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    let mut ix_iy: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(width, height);
    compute_gradient_products_into(&gx, &gy, &mut ix_sq, &mut iy_sq, &mut ix_iy);

    box_filter_3x3_in_place(&mut ix_sq);
    box_filter_3x3_in_place(&mut iy_sq);
    box_filter_3x3_in_place(&mut ix_iy);

    let mut features: Vec<(u32, u32, f32)> = Vec::new();
    compute_min_eigenvalues_into(&ix_sq, &iy_sq, &ix_iy, &mut features);

    let mut is_local_max: Vec<bool> = Vec::new();
    non_maximum_suppression(&mut features, width, height, &mut is_local_max);
    filter_by_quality(&mut features, quality_level);

    features.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    let mut grid: Vec<Option<(u32, u32)>> = Vec::new();
    let mut out: Vec<(u32, u32, f32)> = Vec::new();
    filter_by_distance_into(&features, min_distance, width, height, &mut grid, &mut out);
    out
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
    // Capacity is set by the caller (FeaturesBuffer::with_capacity in Task 6).

    for y in 0..a.height() {
        for x in 0..a.width() {
            let a_val = a.get_pixel(x, y)[0] as i32;
            let b_val = b.get_pixel(x, y)[0] as i32;
            let c_val = c.get_pixel(x, y)[0] as i32;

            let trace = a_val + b_val;
            let discriminant = (a_val - b_val).pow(2) + 4 * c_val.pow(2);
            let min_eigen = (((trace - discriminant) as f32).sqrt()) / 2.0;

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
    grid: &mut Vec<Option<(u32, u32)>>,
    out: &mut Vec<(u32, u32, f32)>,
) {
    let cell_size = min_distance;
    let grid_width = width.div_ceil(cell_size);
    let grid_height = height.div_ceil(cell_size);
    let cells = (grid_width * grid_height) as usize;

    grid.clear();
    grid.resize(cells, None);
    out.clear();

    let min_dist_sq = (min_distance * min_distance) as i32;

    for &(x, y, q) in features {
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
            out.push((x, y, q));
        }
    }
}
