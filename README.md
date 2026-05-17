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
use optical_flow_lk::OpticalFlowBuilder;

let mut buf = OpticalFlowBuilder::new(width, height)
    .pyramid_levels(4)
    .window_size(21)
    .max_iterations(30)
    .feature_quality_level(0.1)
    .feature_min_distance(5)
    .build();

// Prime the pipeline with the first frame and detect features on it.
buf.push_frame(&first_frame_view)?;
buf.good_features_to_track()?;

// Track those features through subsequent frames.
for frame_view in frames {
    buf.push_frame(&frame_view)?;
    buf.calculate_flow()?;
    for &feature in buf.current_features() {
        // feature.x, feature.y, feature.strength
    }
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

Feature buffers are owned by the pipeline and are accessible via
`current_features()` and `previous_features()`. Each `Feature` carries its
spatial position and its Shi-Tomasi strength through tracking.

Re-detect features at any time by calling `good_features_to_track()` after a
`push_frame`. The next `calculate_flow()` call will track whatever features
are in the previous frame buffer.
