use image::{flat::FlatSamples, GrayImage, ImageBuffer, Luma};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const HORIZONTAL_SCHARR_3X3_OLD: [i32; 9] = [-3, 0, 3, -10, 0, 10, -3, 0, 3];
const VERTICAL_SCHARR_3X3_OLD: [i32; 9] = [-3, -10, -3, 0, 0, 0, 3, 10, 3];

type GradientProduct = (
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

/// Computes signed Scharr gradients.
///
/// Selection is done per target:
/// - `aarch64`: NEON
/// - `x86`/`x86_64`: runtime AVX2 detection, otherwise scalar fallback
/// - everything else: historical scalar implementation
#[cfg(target_arch = "aarch64")]
pub fn compute_gradients(image: &FlatSamples<&[u8]>) -> GradientProduct {
    unsafe { compute_gradients_neon(image) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn compute_gradients(image: &FlatSamples<&[u8]>) -> GradientProduct {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            return compute_gradients_avx2(image);
        }
    }

    compute_gradients_manual(image, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
pub fn compute_gradients(image: &FlatSamples<&[u8]>) -> GradientProduct {
    compute_gradients_manual(image, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
fn zero_gradients(width: u32, height: u32) -> GradientProduct {
    (
        ImageBuffer::new(width, height),
        ImageBuffer::new(width, height),
    )
}

fn compute_gradients_manual(
    image: &FlatSamples<&[u8]>,
    kernel_x: &[i32; 9],
    kernel_y: &[i32; 9],
) -> GradientProduct {
    let width = image.layout.width;
    let height = image.layout.height;
    let row_stride = image.layout.height_stride;
    let src = image.samples;

    let mut grad_x = ImageBuffer::new(width, height);
    let mut grad_y = ImageBuffer::new(width, height);

    if width < 3 || height < 3 {
        return (grad_x, grad_y);
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3u32 {
                for kx in 0..3u32 {
                    let sx = (x + kx - 1) as usize;
                    let sy = (y + ky - 1) as usize;
                    let pixel = src[sy * row_stride + sx] as i32;
                    gx += pixel * kernel_x[(ky * 3 + kx) as usize];
                    gy += pixel * kernel_y[(ky * 3 + kx) as usize];
                }
            }

            grad_x.put_pixel(x, y, Luma([gx as i16]));
            grad_y.put_pixel(x, y, Luma([gy as i16]));
        }
    }

    (grad_x, grad_y)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn compute_gradients_avx2(image: &FlatSamples<&[u8]>) -> GradientProduct {
    let width_u32 = image.layout.width;
    let height_u32 = image.layout.height;
    let width = width_u32 as usize;
    let height = height_u32 as usize;
    let row_stride = image.layout.height_stride;

    if width < 3 || height < 3 {
        return zero_gradients(width_u32, height_u32);
    }

    let src = image.samples;
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = _mm256_set1_epi16(3);
    let coeff10 = _mm256_set1_epi16(10);
    let interior_chunks_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * row_stride),
                src.as_ptr().add(y * row_stride),
                src.as_ptr().add((y + 1) * row_stride),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < interior_chunks_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16(top.add(x - 1)),
                    load_u8x16_as_i16(top.add(x)),
                    load_u8x16_as_i16(top.add(x + 1)),
                    load_u8x16_as_i16(mid.add(x - 1)),
                    load_u8x16_as_i16(mid.add(x + 1)),
                    load_u8x16_as_i16(bottom.add(x - 1)),
                    load_u8x16_as_i16(bottom.add(x)),
                    load_u8x16_as_i16(bottom.add(x + 1)),
                )
            };

            let gx3 = _mm256_sub_epi16(_mm256_add_epi16(tr, br), _mm256_add_epi16(tl, bl));
            let gx10 = _mm256_sub_epi16(mr, ml);
            let gx = _mm256_add_epi16(
                _mm256_mullo_epi16(gx3, coeff3),
                _mm256_mullo_epi16(gx10, coeff10),
            );

            let gy3 = _mm256_sub_epi16(_mm256_add_epi16(bl, br), _mm256_add_epi16(tl, tr));
            let gy10 = _mm256_sub_epi16(bc, tc);
            let gy = _mm256_add_epi16(
                _mm256_mullo_epi16(gy3, coeff3),
                _mm256_mullo_epi16(gy10, coeff10),
            );

            unsafe {
                _mm256_storeu_si256(grad_x.as_mut_ptr().add(row + x) as *mut __m256i, gx);
                _mm256_storeu_si256(grad_y.as_mut_ptr().add(row + x) as *mut __m256i, gy);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * row_stride + x + 1] as i32 + src[(y + 1) * row_stride + x + 1] as i32)
                    - (src[(y - 1) * row_stride + x - 1] as i32 + src[(y + 1) * row_stride + x - 1] as i32))
                + 10 * (src[y * row_stride + x + 1] as i32 - src[y * row_stride + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * row_stride + x - 1] as i32 + src[(y + 1) * row_stride + x + 1] as i32)
                    - (src[(y - 1) * row_stride + x - 1] as i32 + src[(y - 1) * row_stride + x + 1] as i32))
                + 10 * (src[(y + 1) * row_stride + x] as i32 - src[(y - 1) * row_stride + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(width_u32, height_u32, grad_x).unwrap(),
        ImageBuffer::from_vec(width_u32, height_u32, grad_y).unwrap(),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn load_u8x16_as_i16(ptr: *const u8) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(ptr as *const __m128i)) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compute_gradients_neon(image: &FlatSamples<&[u8]>) -> GradientProduct {
    let width_u32 = image.layout.width;
    let height_u32 = image.layout.height;
    let width = width_u32 as usize;
    let height = height_u32 as usize;
    let row_stride = image.layout.height_stride;

    if width < 3 || height < 3 {
        return zero_gradients(width_u32, height_u32);
    }

    let src = image.samples;
    let mut grad_x = vec![0i16; width * height];
    let mut grad_y = vec![0i16; width * height];

    let coeff3 = vdupq_n_s16(3);
    let coeff10 = vdupq_n_s16(10);
    let interior_chunks_end = 1 + ((width - 2) / 16) * 16;

    for y in 1..height - 1 {
        let (top, mid, bottom) = unsafe {
            (
                src.as_ptr().add((y - 1) * row_stride),
                src.as_ptr().add(y * row_stride),
                src.as_ptr().add((y + 1) * row_stride),
            )
        };
        let row = y * width;

        let mut x = 1usize;
        while x < interior_chunks_end {
            let (tl, tc, tr, ml, mr, bl, bc, br) = unsafe {
                (
                    load_u8x16_as_i16x8x2(top.add(x - 1)),
                    load_u8x16_as_i16x8x2(top.add(x)),
                    load_u8x16_as_i16x8x2(top.add(x + 1)),
                    load_u8x16_as_i16x8x2(mid.add(x - 1)),
                    load_u8x16_as_i16x8x2(mid.add(x + 1)),
                    load_u8x16_as_i16x8x2(bottom.add(x - 1)),
                    load_u8x16_as_i16x8x2(bottom.add(x)),
                    load_u8x16_as_i16x8x2(bottom.add(x + 1)),
                )
            };

            let gx_lo = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(tr.0, br.0), vaddq_s16(tl.0, bl.0)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(mr.0, ml.0), coeff10),
            );
            let gx_hi = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(tr.1, br.1), vaddq_s16(tl.1, bl.1)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(mr.1, ml.1), coeff10),
            );
            let gy_lo = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(bl.0, br.0), vaddq_s16(tl.0, tr.0)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(bc.0, tc.0), coeff10),
            );
            let gy_hi = vaddq_s16(
                vmulq_s16(
                    vsubq_s16(vaddq_s16(bl.1, br.1), vaddq_s16(tl.1, tr.1)),
                    coeff3,
                ),
                vmulq_s16(vsubq_s16(bc.1, tc.1), coeff10),
            );

            unsafe {
                vst1q_s16(grad_x.as_mut_ptr().add(row + x), gx_lo);
                vst1q_s16(grad_x.as_mut_ptr().add(row + x + 8), gx_hi);
                vst1q_s16(grad_y.as_mut_ptr().add(row + x), gy_lo);
                vst1q_s16(grad_y.as_mut_ptr().add(row + x + 8), gy_hi);
            }
            x += 16;
        }

        while x < width - 1 {
            let idx = row + x;
            let gx = 3
                * ((src[(y - 1) * row_stride + x + 1] as i32 + src[(y + 1) * row_stride + x + 1] as i32)
                    - (src[(y - 1) * row_stride + x - 1] as i32 + src[(y + 1) * row_stride + x - 1] as i32))
                + 10 * (src[y * row_stride + x + 1] as i32 - src[y * row_stride + x - 1] as i32);
            let gy = 3
                * ((src[(y + 1) * row_stride + x - 1] as i32 + src[(y + 1) * row_stride + x + 1] as i32)
                    - (src[(y - 1) * row_stride + x - 1] as i32 + src[(y - 1) * row_stride + x + 1] as i32))
                + 10 * (src[(y + 1) * row_stride + x] as i32 - src[(y - 1) * row_stride + x] as i32);

            grad_x[idx] = gx as i16;
            grad_y[idx] = gy as i16;
            x += 1;
        }
    }

    (
        ImageBuffer::from_vec(width_u32, height_u32, grad_x).unwrap(),
        ImageBuffer::from_vec(width_u32, height_u32, grad_y).unwrap(),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn load_u8x16_as_i16x8x2(ptr: *const u8) -> (int16x8_t, int16x8_t) {
    let bytes = unsafe { vld1q_u8(ptr) };
    (
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes))),
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bytes))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(width: u32, height: u32) -> GrayImage {
        let mut img = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let v = ((x * 31 + y * 17 + (x ^ y) * 13) & 0xff) as u8;
                img.put_pixel(x, y, Luma([v]));
            }
        }

        img
    }

    #[test]
    fn selected_gradients_match_manual_reference() {
        let img = make_test_image(128, 96);
        let fs = img.as_flat_samples();
        let expected = compute_gradients_manual(
            &fs, &HORIZONTAL_SCHARR_3X3_OLD, &VERTICAL_SCHARR_3X3_OLD,
        );
        let actual = compute_gradients(&fs);

        assert_eq!(expected.0, actual.0, "horizontal gradients differ");
        assert_eq!(expected.1, actual.1, "vertical gradients differ");
    }

    #[test]
    fn tiny_images_return_zero_gradients() {
        for (width, height) in [(0, 0), (1, 1), (2, 2), (2, 5), (5, 2)] {
            let img = GrayImage::new(width, height);
            let fs = img.as_flat_samples();
            let (gx, gy) = compute_gradients(&fs);

            assert_eq!(gx.dimensions(), (width, height));
            assert_eq!(gy.dimensions(), (width, height));
            assert!(gx.pixels().all(|p| p[0] == 0));
            assert!(gy.pixels().all(|p| p[0] == 0));
        }
    }
}
