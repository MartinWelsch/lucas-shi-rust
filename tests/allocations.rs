//! Allocation regression tests for the v0.4.0 buffer API.
//!
//! The custom global allocator counts every alloc on the calling thread via
//! a thread-local counter, so tests are safe to run in parallel under any
//! `--test-threads` setting.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

use image::flat::{FlatSamples, SampleLayout};
use optical_flow_lk::OpticalFlowBuilder;

struct CountingAllocator;

thread_local! {
    // Number of allocations on this thread while MEASURING is true.
    static THREAD_ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    // Whether this thread is currently inside a `measure` call.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MEASURING.with(|m| {
            if m.get() {
                THREAD_ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

// Serializes test measurements so the counting allocator's atomic counters
// have an uncontended measurement window, regardless of how the test runner
// schedules tests.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

fn measure<F: FnOnce()>(f: F) -> usize {
    let _guard = MEASUREMENT_LOCK.lock().expect("measurement lock poisoned");
    THREAD_ALLOC_COUNT.with(|c| c.set(0));
    MEASURING.with(|m| m.set(true));
    f();
    MEASURING.with(|m| m.set(false));
    THREAD_ALLOC_COUNT.with(|c| c.get())
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

// ----- Strict zero-allocation assertion for the buffer path -----

#[test]
fn buffer_path_is_steady_state_zero_alloc() {
    const W: u32 = 256;
    const H: u32 = 256;

    let frame_a = checkerboard(W, H, 8);
    let frame_b: Vec<u8> = frame_a.iter().map(|p| p.wrapping_add(2)).collect();

    let mut buf = OpticalFlowBuilder::new(W, H).build();

    let mut points: Vec<(f32, f32)> = Vec::new();
    let mut tracked: Vec<(f32, f32)> = Vec::new();

    // Warm-up:
    //   1. push primes curr.
    //   2. detect populates `points` to its steady-state size.
    //   3. push rotates so prev=a, curr=b.
    //   4. calculate_flow fills `tracked` to steady-state.
    //   5. Additional push+flow cycles stabilize internal Vecs.
    buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
    buf.good_features_to_track(&mut points).unwrap();
    buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
    buf.calculate_flow(&points, &mut tracked).unwrap();
    std::mem::swap(&mut points, &mut tracked);
    buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
    buf.calculate_flow(&points, &mut tracked).unwrap();
    std::mem::swap(&mut points, &mut tracked);
    buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
    buf.calculate_flow(&points, &mut tracked).unwrap();
    std::mem::swap(&mut points, &mut tracked);

    let n_alloc = measure(|| {
        for _ in 0..10 {
            buf.push_frame(&make_view(&frame_a, W, H)).unwrap();
            buf.calculate_flow(&points, &mut tracked).unwrap();
            std::mem::swap(&mut points, &mut tracked);
            buf.push_frame(&make_view(&frame_b, W, H)).unwrap();
            buf.calculate_flow(&points, &mut tracked).unwrap();
            std::mem::swap(&mut points, &mut tracked);
        }
    });

    assert_eq!(
        n_alloc, 0,
        "OpticalFlowBuffer steady-state allocated {n_alloc} times in 20 frames \
         after warm-up; expected 0."
    );
}

#[test]
fn buffer_path_good_features_to_track_is_zero_alloc_after_warmup() {
    const W: u32 = 256;
    const H: u32 = 256;
    let frame = checkerboard(W, H, 8);
    let view = make_view(&frame, W, H);

    let mut buf = OpticalFlowBuilder::new(W, H).build();
    buf.push_frame(&view).unwrap();

    let mut points: Vec<(f32, f32)> = Vec::new();
    // Warm-up: the first detect may grow `points` to its steady size.
    buf.good_features_to_track(&mut points).unwrap();
    buf.good_features_to_track(&mut points).unwrap();

    let n_alloc = measure(|| {
        for _ in 0..5 {
            buf.good_features_to_track(&mut points).unwrap();
        }
    });

    assert_eq!(
        n_alloc, 0,
        "good_features_to_track allocated {n_alloc} times in 5 calls \
         after warm-up; expected 0."
    );
}
