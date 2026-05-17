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

## Quick start

```rust
use image::open;
use optical_flow_lk::OpticalFlowBuilder;

let prev_frame = open("examples/input1.png")?.into_luma8();
let next_frame = open("examples/input2.png")?.into_luma8();
let (w, h) = (prev_frame.width(), prev_frame.height());

let mut buf = OpticalFlowBuilder::new(w, h)
    .pyramid_levels(4)
    .window_size(21)
    .max_iterations(30)
    .feature_quality_level(0.1)
    .feature_min_distance(5)
    .build();

// Prime the pipeline with the first frame and detect features on it.
buf.push_frame(&prev_frame.as_flat_samples())?;
buf.reset_with_good_features_to_track()?;

// Track those features into the next frame.
buf.push_frame(&next_frame.as_flat_samples())?;

for &(x, y) in buf.points() {
    println!("tracked point at ({x}, {y})");
}
```

## Zero-copy input — subrect or NV12 Y plane

`OpticalFlowBuffer::push_frame` accepts any `&FlatSamples<B>` matching the
buffer's configured resolution. Subrects of larger images and NV12 Y planes
(possibly with row padding) are zero-copy.

### Subrect of a larger `GrayImage`

```rust
use optical_flow_lk::{FlatSamples, OpticalFlowBuilder, SampleLayout};

let mut buf = OpticalFlowBuilder::new(w, h).build();

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
buf.push_frame(&view)?;
```

### NV12 Y plane (full or subrect)

```rust
use optical_flow_lk::{FlatSamples, OpticalFlowBuilder, SampleLayout};

let mut buf = OpticalFlowBuilder::new(width, height).build();
let view = FlatSamples {
    samples: &nv12_buffer[..],
    layout: SampleLayout {
        channels: 1, channel_stride: 1,
        width, width_stride: 1,
        height, height_stride: y_stride,    // may exceed width for padded NV12
    },
    color_hint: None,
};
buf.push_frame(&view)?;
```

`push_frame` returns `Err(TrackError::Layout(LayoutError::…))` if the
`FlatSamples` layout cannot be safely consumed — see [`TrackError`] /
[`LayoutError`].

## Real-time tracking — `OpticalFlowBuffer`

After the initial warm-up `OpticalFlowBuffer` performs zero heap allocations
per frame. `push_frame` accepts any `&FlatSamples<B>` matching the configured
resolution, so subrects of larger images and NV12 Y planes are zero-copy.

`buf.points_mut()` allows direct manipulation of the tracked list — push, pop,
or remove individual points without going through `reset`.

Manual reset (custom seed):

```rust
buf.reset(vec![(10.0, 20.0), (200.0, 50.0)]);
```
