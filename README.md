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
optical-flow-lk = "0.1"
```

Basic example:
```rust
use image::{open, GrayImage, Rgba};
use optical_flow_lk::{build_pyramid, calc_optical_flow, good_features_to_track};

let prev_frame: GrayImage = open("examples/input1.png").unwrap().clone().into_luma8();
let next_frame: GrayImage = open("examples/input2.png").unwrap().clone().into_luma8();

let prev_frame_pyr = build_pyramid(&prev_frame, 4);
let next_frame_pyr = build_pyramid(&next_frame, 4);

let mut points = good_features_to_track(&prev_frame, 0.1, 5);
points.truncate(100);
let prev_points: Vec<(f32, f32)> = points.iter().map(|&x| (x.0 as f32, x.1 as f32)).collect();

let next_points = calc_optical_flow(&prev_frame_pyr, &next_frame_pyr, &prev_points, 21, 30);
```

## Zero-copy input — subrect or NV12 Y plane

Use `optical_flow_lk::generic::*` to pass a subrect of a larger image or the Y plane of an
NV12 buffer without copying. These functions accept `image::flat::FlatSamples<B>`, which
carries a row stride and borrows its data; the existing top-level `build_pyramid` /
`good_features_to_track` (taking `&GrayImage`) are unchanged.

### Subrect of a larger `GrayImage`

```rust
use optical_flow_lk::{generic, FlatSamples, SampleLayout};

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
let pyramid = generic::build_pyramid(&view, 4)?;
```

### NV12 Y plane (full or subrect)

```rust
use optical_flow_lk::{generic, FlatSamples, SampleLayout};

let view = FlatSamples {
    samples: &nv12_buffer[..],
    layout: SampleLayout {
        channels: 1, channel_stride: 1,
        width, width_stride: 1,
        height, height_stride: y_stride,    // may exceed width for padded NV12
    },
    color_hint: None,
};
let pyramid = generic::build_pyramid(&view, 4)?;
```

The `generic::*` functions validate the layout once at entry and return
`Result<_, LayoutError>`. The four error variants — `UnsupportedChannels`,
`UnsupportedWidthStride`, `OverlappingRows`, `BufferTooSmall` — cover only conditions that
would cause wrong output or out-of-bounds reads.
