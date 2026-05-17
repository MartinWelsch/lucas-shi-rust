//! Allocation regression tests for the v0.4.0 buffer API.
//!
//! Run with:
//!   cargo test --test allocations -- --test-threads=1
//!
//! The custom global allocator counts every alloc/dealloc; multi-threaded
//! execution would race the counters.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use image::flat::{FlatSamples, SampleLayout};
use optical_flow_lk::{OpticalFlowBuilder, generic};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

fn allocs() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

fn measure<F: FnOnce()>(f: F) -> usize {
    let before = allocs();
    f();
    allocs() - before
}

fn make_view<'a>(buf: &'a [u8], width: u32, height: u32) -> FlatSamples<&'a [u8]> {
    FlatSamples {
        samples: buf,
        layout: SampleLayout {
            channels: 1,
            channel_stride: 1,
            width,
            width_stride: 1,
            height,
            height_stride: width as usize,
        },
        color_hint: None,
    }
}

fn checkerboard(width: u32, height: u32, cell: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            out[(y * width + x) as usize] = if on { 255 } else { 0 };
        }
    }
    out
}

// ----- Upper-bound assertions for the legacy generic path -----
//
// Measured on initial v0.4.0 implementation. Bump these only after a deliberate
// review of the new allocation site. To find the current measured value: set
// the constant to 0 and read the failure message.

const MAX_ALLOCS_GENERIC_BUILD_PYRAMID: usize = 4;
const MAX_ALLOCS_GENERIC_FEATURES: usize = 11;

#[test]
fn generic_build_pyramid_allocations_within_bound() {
    let frame = vec![0u8; 256 * 256];
    let view = make_view(&frame, 256, 256);

    // Warm-up — first call may trigger one-time setup.
    let _ = generic::build_pyramid(&view, 3).unwrap();

    let n_alloc = measure(|| {
        let _ = generic::build_pyramid(&view, 3).unwrap();
    });

    assert!(
        n_alloc <= MAX_ALLOCS_GENERIC_BUILD_PYRAMID,
        "generic::build_pyramid allocated {n_alloc} times \
         (upper bound {MAX_ALLOCS_GENERIC_BUILD_PYRAMID}). \
         If this increase is intentional and justified, bump the bound \
         after reviewing the new allocation site."
    );
}

#[test]
fn generic_good_features_allocations_within_bound() {
    let frame = checkerboard(256, 256, 8);
    let view = make_view(&frame, 256, 256);
    let _ = generic::good_features_to_track(&view, 0.4, 10).unwrap();

    let n_alloc = measure(|| {
        let _ = generic::good_features_to_track(&view, 0.4, 10).unwrap();
    });

    assert!(
        n_alloc <= MAX_ALLOCS_GENERIC_FEATURES,
        "generic::good_features_to_track allocated {n_alloc} times \
         (upper bound {MAX_ALLOCS_GENERIC_FEATURES}). \
         If this increase is intentional and justified, bump the bound \
         after reviewing the new allocation site."
    );
}

// ----- Strict zero-allocation assertion for the buffer path -----

#[test]
fn buffer_path_is_steady_state_zero_alloc() {
    const W: u32 = 256;
    const H: u32 = 256;

    let frame_a = checkerboard(W, H, 8);
    let frame_b: Vec<u8> = frame_a.iter().map(|p| p.wrapping_add(2)).collect();

    let mut buf = OpticalFlowBuilder::new(W, H).build();

    // Warm-up:
    //   1. first push primes prev_pyramid (no flow)
    //   2. detect populates points to steady-state size
    //   3. additional pushes run LK for the first time, which may grow
    //      displacements / output buffers to the steady-state size
    buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
    buf.reset_with_good_features_to_track().unwrap();
    buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
    buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
    buf.push_frame(&make_view(&frame_b, W, H)).unwrap();

    let n_alloc = measure(|| {
        for _ in 0..10 {
            buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
            buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
        }
    });

    assert_eq!(
        n_alloc, 0,
        "OpticalFlowBuffer::push_frame allocated {n_alloc} times in 20 frames \
         after warm-up; expected 0. Common causes: hidden .to_vec() or .collect() \
         in the hot path; Vec::resize past pre-allocated capacity; new \
         intermediate Vec declared inside a per-frame method; panic codepath."
    );
}

#[test]
fn buffer_path_reset_with_good_features_is_zero_alloc_after_warmup() {
    const W: u32 = 256;
    const H: u32 = 256;
    let frame = checkerboard(W, H, 8);
    let view = make_view(&frame, W, H);

    let mut buf = OpticalFlowBuilder::new(W, H).build();
    buf.push_frame(&view).unwrap();
    // Warm-up: the first detect may grow `self.points` to its steady size.
    buf.reset_with_good_features_to_track().unwrap();
    buf.reset_with_good_features_to_track().unwrap();

    let n_alloc = measure(|| {
        for _ in 0..5 {
            buf.reset_with_good_features_to_track().unwrap();
        }
    });

    assert_eq!(
        n_alloc, 0,
        "reset_with_good_features_to_track allocated {n_alloc} times in 5 calls \
         after warm-up; expected 0."
    );
}
