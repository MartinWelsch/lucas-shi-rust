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

For per-frame tracking loops, build an `OpticalFlowBuffer` once and call
`push_frame` for each new frame. After warm-up the library performs zero
heap allocations per frame.

```rust
use optical_flow_lk::OpticalFlowBuilder;

let mut buf = OpticalFlowBuilder::new(width, height)
    .pyramid_levels(3)
    .window_size(21)
    .max_iterations(30)
    .feature_quality_level(0.4)
    .feature_min_distance(10)
    .build();

// Prime the pipeline with the first frame and seed the points.
buf.push_frame(&first_frame_view)?;
buf.reset_with_good_features_to_track()?;

for frame_view in frames {
    buf.push_frame(&frame_view)?;
    // buf.points() now holds the updated positions.
    for &(x, y) in buf.points() {
        // …
    }
}
```

`push_frame` accepts any `&FlatSamples<B>` matching the configured resolution,
so subrects of larger images and NV12 Y planes are zero-copy.

`buf.points_mut()` allows direct manipulation of the tracked list — push, pop,
or remove individual points without going through `reset`.

Manual reset (custom seed):

```rust
buf.reset(vec![(10.0, 20.0), (200.0, 50.0)]);
```
