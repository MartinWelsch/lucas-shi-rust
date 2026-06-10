# Lucas Canade Optical Flow and Shi-Tomasi feature detection on Rust

[![Crates.io](https://img.shields.io/crates/v/optical-flow-lk)](https://crates.io/crates/optical-flow-lk)
[![Documentation](https://docs.rs/optical-flow-lk/badge.svg)](https://docs.rs/optical-flow-lk)

High-performance Rust implementation of Lucas-Kanade optical flow and Shi-Tomasi feature detection, optimized for real-time applications and WebAssembly (Wasm) compatibility.

## Features

- 🔍 Efficient feature point detection using Shi-Tomasi
- 🖼️ Integration with `image` and `imageproc` crates
- 🌐 WebAssembly (Wasm) compatible

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
optical-flow-lk = "0.4"
```

## Quick start — `OpticalFlowTracker`

```rust
use optical_flow_lk::OpticalFlowBuilder;

let mut tracker = OpticalFlowBuilder::new(width, height)
    .pyramid_levels(4)
    .window_size(21)
    .max_iterations(30)
    .feature_quality_level(0.1)
    .feature_min_distance(5)
    .max_features(300)
    .build();

// Push the first frame and detect features on it.
tracker.push_frame(&first_frame.as_flat_samples())?;
tracker.detect_features()?;

// Track those features through subsequent frames.
for frame in frames {
    tracker.push_frame(&frame.as_flat_samples())?;
    tracker.calculate_flow()?;
    for &feature in tracker.features() {
        // feature.x, feature.y, feature.strength
    }
}
```

`features()` returns a single feature list that describes positions in the
most recently consumed frame. `push_frame` does not touch it; the next
`calculate_flow()` mutates the positions in place from the previous frame
into the just-pushed one. After warm-up, no method on the tracker
allocates.

## Zero-copy input — subrect or NV12 Y plane

`OpticalFlowTracker::push_frame` accepts any `&FlatSamples<B>` matching the
tracker's configured resolution. Subrects of larger images and NV12 Y
planes (possibly with row padding) are zero-copy.

### Subrect of a larger `GrayImage`

```rust
use optical_flow_lk::{FlatSamples, OpticalFlowBuilder, SampleLayout};

let mut tracker = OpticalFlowBuilder::new(w, h).build();

let stride = big.width() as usize;
let off = oy as usize * stride + ox as usize;
let view = FlatSamples {
    samples: &big.as_raw()[off..],
    layout: SampleLayout {
        channels: 1, channel_stride: 1,
        width: w, width_stride: 1,
        height: h, height_stride: stride,
    },
    color_hint: None,
};
tracker.push_frame(&view)?;
```

### NV12 Y plane (full or subrect)

```rust
use optical_flow_lk::{FlatSamples, OpticalFlowBuilder, SampleLayout};

let mut tracker = OpticalFlowBuilder::new(width, height).build();
let view = FlatSamples {
    samples: &nv12_buffer[..],
    layout: SampleLayout {
        channels: 1, channel_stride: 1,
        width, width_stride: 1,
        height, height_stride: y_stride,    // may exceed width for padded NV12
    },
    color_hint: None,
};
tracker.push_frame(&view)?;
```

`push_frame` returns `Err(TrackError::Layout(LayoutError::…))` if the
`FlatSamples` layout cannot be safely consumed.

## Manual pipeline — `buffers` module

When `OpticalFlowTracker` doesn't fit (e.g., sharing one pyramid across
multiple LK runs, keeping more than two pyramids in memory, or running
detection without a tracker), use the building blocks directly:

```rust
use optical_flow_lk::Feature;
use optical_flow_lk::buffers::{
    FeaturesBuffer, LkBuffer, PyramidBuffer,
    build_pyramid, detect_features, track,
};

let (w, h, levels, window, max_features) = (640, 480, 3, 21, 500);

let mut prev_pyr = PyramidBuffer::with_capacity(w, h, levels);
let mut curr_pyr = PyramidBuffer::with_capacity(w, h, levels);
let mut lk = LkBuffer::with_capacity(w, h, levels, window, max_features);
let mut fb = FeaturesBuffer::with_capacity(w, h, /* min_distance */ 5, max_features);
let mut features: Vec<Feature> = Vec::with_capacity(max_features);

build_pyramid(&mut curr_pyr, &first_frame.as_flat_samples())?;
detect_features(&mut fb, &curr_pyr, 0.1, 5, max_features, &mut features);

for frame in frames {
    std::mem::swap(&mut prev_pyr, &mut curr_pyr);
    build_pyramid(&mut curr_pyr, &frame.as_flat_samples())?;
    track(&mut lk, &prev_pyr, &curr_pyr, &mut features, 30);
    // features now hold positions in `frame`.
}
```
