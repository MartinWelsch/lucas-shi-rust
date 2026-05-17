# Spec: `OpticalFlowBuilder` / `OpticalFlowBuffer` — Allocation-Free Per-Frame Pipeline

**Status:** Draft (rev 10)
**Date:** 2026-05-17
**Scope:** New public types `OpticalFlowBuilder` and `OpticalFlowBuffer` for a steady-state, zero-per-frame-allocation tracking loop. Existing `build_pyramid`, `good_features_to_track`, `calc_optical_flow`, and `generic::*` APIs unchanged.

---

## 1. Motivation

The current v0.3.0 API allocates on every frame:

| Per-frame call | Allocations |
|---|---|
| `build_pyramid(_, 3)` | ~4 (Vec + 3 levels) |
| `calc_optical_flow(...)` | ~17 for 3 levels (per-level grad_x/grad_y/patch buffers + displacements + output) |
| `good_features_to_track(_)` (if periodic) | ~9 + per-cell grid Vecs |

For a real-time tracking loop at 30 fps the malloc/free churn is the dominant variable cost — both in latency jitter and in allocator pressure. The pixel data itself can't be reused frame-to-frame, but every Vec the library owns has a stable size determined by the image resolution and configured pyramid depth / window size, and can be reused indefinitely.

This spec introduces a stateful tracking type that pre-allocates every internal buffer at construction time and reuses them on each frame. After warm-up, `push_frame` performs zero heap allocations.

## 2. Goals

- After construction, `push_frame` performs **zero heap allocations** in the library (modulo the unchanged level-0 pixel copy).
- A single user-facing type owns:
  - Two pre-sized `PyramidBuffer` instances that are swapped per frame.
  - All `compute_gradients` / Lucas-Kanade / Shi-Tomasi reusable buffers.
  - The caller manages their own `Vec<(f32, f32)>` feature buffers; the buffer type does not own a points list.
- All internal state is private. Configuration values and lifecycle flags are
  exposed only through read-only accessors (`has_current_frame()`,
  `has_previous_frame()`, `dimensions()`, `window_size()`, etc.) so future
  changes to internal representation don't break callers.
- A builder pattern (`OpticalFlowBuilder`) collects all global parameters: image resolution, pyramid depth, LK window size, LK max iterations, and Shi-Tomasi quality/distance defaults.
- The buffer accepts arbitrary `image::flat::FlatSamples<B>` (full image, subrect, NV12 Y plane) as input to `push_frame`, matching the configured resolution.
- Feature detection (`good_features_to_track(&mut Vec<(f32, f32)>)`) and flow (`calculate_flow(&[(f32, f32)], &mut Vec<(f32, f32)>)`) are separate explicit calls. The caller controls when each runs and which buffers to pass.
- The existing v0.3.0 public API (`build_pyramid`, `good_features_to_track`, `calc_optical_flow`, `generic::*`) is **unchanged**.
- **No code duplication.** Each algorithm has exactly one implementation, expressed
  as a `*_into` method on its buffer type. Both the old free-function APIs and
  the new `OpticalFlowBuffer` route through the same `*_into` methods. The free
  functions become thin shims that allocate a buffer, call `*_into`, and discard
  the buffer; the new buffer reuses the same buffer across frames. See §6 for
  the exact layering.

## 3. Non-Goals

- Eliminating the level-0 pixel copy. Listed as out-of-scope follow-up in the v0.3.0 spec; would require pyramid-type surgery and a lifetime parameter on the buffer.
- Multi-resolution pipelines. The buffer is constructed for one fixed `(width, height)` and rejects inputs with different dimensions.
- Multithreaded tracking across frames. The buffer is `!Send` only if its internals are (it'll naturally be `Send` for `Vec<u8>`-backed buffers). Concurrent calls to `push_frame` on the same buffer are undefined and not part of the API contract.
- Pyramid-only or features-only convenience methods. `OpticalFlowBuffer` is for the full tracking pipeline; callers needing just the pyramid or just feature detection use the existing free functions.
- Tracking points outside the image. Out-of-bounds points are silently retained in the points list (consistent with current `calc_optical_flow` behavior, which skips them via `in_bounds`).

## 4. Public API

### 4.1 `OpticalFlowBuilder`

```rust
pub struct OpticalFlowBuilder {
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,
}

impl OpticalFlowBuilder {
    /// Start a builder for a fixed-resolution pipeline.
    ///
    /// Defaults: 3 pyramid levels, 21x21 window, 30 max iterations,
    /// quality_level=0.4, min_distance=10.
    pub fn new(width: u32, height: u32) -> Self;

    pub fn pyramid_levels(self, levels: usize) -> Self;
    pub fn window_size(self, size: usize) -> Self;          // must be odd
    pub fn max_iterations(self, n: usize) -> Self;
    pub fn feature_quality_level(self, q: f32) -> Self;
    pub fn feature_min_distance(self, d: u32) -> Self;

    /// Construct the buffer, pre-allocating all internal storage.
    /// Panics if `width == 0`, `height == 0`, `pyramid_levels == 0`,
    /// or `window_size` is even.
    pub fn build(self) -> OpticalFlowBuffer;
}
```

Defaults are chosen to match the values used in `examples/optical_flow.rs` so first-time callers get sensible behavior.

### 4.2 `Feature` and `OpticalFlowBuffer`

#### `Feature`

```rust
/// A tracked feature point with spatial position and detection strength.
///
/// `strength` is the Shi-Tomasi min-eigenvalue from feature detection
/// (always >= 0). Features produced by `calculate_flow` inherit their input
/// feature's strength unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feature {
    pub x: f32,
    pub y: f32,
    pub strength: f32,
}
```

#### `OpticalFlowBuffer`

All fields are private. Feature lists are owned by the buffer and are read
through `current_features()` / `previous_features()`. There are no public
fields — this enforces the lifecycle contract and keeps the internal
representation free to evolve.

```rust
pub struct OpticalFlowBuffer {
    // Frozen at build time:
    width: u32,
    height: u32,
    pyramid_levels: usize,
    window_size: usize,
    max_iterations: usize,
    feature_quality_level: f32,
    feature_min_distance: u32,

    // Reusable storage (each FrameBuffer holds a pyramid + feature list):
    prev_frame: FrameBuffer,
    curr_frame: FrameBuffer,
    lk_buffer: LkBuffer,
    features_buffer: FeaturesBuffer,
    staging_positions: Vec<(f32, f32)>,  // reusable LK input scratch

    // Lifecycle:
    has_curr: bool,
    has_prev: bool,
}

impl OpticalFlowBuffer {
    // --- mutation: tracking pipeline ---

    /// Rotate prev/curr and build the new frame's pyramid into curr.
    /// Clears `curr_frame.features` (the new frame has no features yet).
    pub fn push_frame<B: AsRef<[u8]>>(
        &mut self,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError>;

    /// Detect Shi-Tomasi features on the current frame and write them into
    /// `current_features()` (cleared first). Uses the builder's
    /// `feature_quality_level` and `feature_min_distance` settings.
    /// Each feature carries its Shi-Tomasi min-eigenvalue as `strength`.
    ///
    /// Errors with `TrackError::NoCurrentFrame` if no frame has been pushed.
    pub fn good_features_to_track(&mut self) -> Result<(), TrackError>;

    /// Track `previous_features()` from the previous frame into the current
    /// frame using Lucas-Kanade. Writes results into `current_features()`
    /// (cleared first). Strength is preserved from each input feature.
    ///
    /// Errors with `TrackError::NoPreviousFrame` if fewer than two frames
    /// have been pushed.
    pub fn calculate_flow(&mut self) -> Result<(), TrackError>;

    // --- access: feature lists ---

    /// Features for the most recently pushed frame (empty until
    /// `good_features_to_track` or `calculate_flow` is called after the
    /// last `push_frame`).
    pub fn current_features(&self) -> &[Feature];

    /// Features for the frame prior to the most recently pushed one (empty
    /// until a second `push_frame` has been called).
    pub fn previous_features(&self) -> &[Feature];

    // --- access: lifecycle and configuration (read-only) ---

    /// True after at least one `push_frame` call.
    pub fn has_current_frame(&self) -> bool;

    /// True after at least two `push_frame` calls — i.e., `calculate_flow`
    /// has the data it needs.
    pub fn has_previous_frame(&self) -> bool;

    pub fn dimensions(&self) -> (u32, u32);
    pub fn pyramid_levels(&self) -> usize;
    pub fn window_size(&self) -> usize;
    pub fn max_iterations(&self) -> usize;
    pub fn feature_quality_level(&self) -> f32;
    pub fn feature_min_distance(&self) -> u32;
}
```

The accessor methods are all inline `&self`-returning getters; they cost
nothing at the call site.

### 4.3 `TrackError`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackError {
    /// The provided `FlatSamples` failed the same layout checks as
    /// `generic::*` — wrapped here so callers get one error type.
    Layout(LayoutError),

    /// `image.layout.width` or `.height` did not match the buffer's
    /// configured resolution.
    DimensionMismatch { expected: (u32, u32), actual: (u32, u32) },

    /// `calculate_flow` was called before two frames had been pushed,
    /// so there is no `prev` frame to track from.
    NoPreviousFrame,

    /// `good_features_to_track` was called before any frame had been pushed,
    /// so there is nothing to detect features on.
    NoCurrentFrame,
}

impl From<LayoutError> for TrackError { ... }
impl std::fmt::Display for TrackError { ... }
impl std::error::Error for TrackError { ... }
```

### 4.4 Lifecycle and semantics

State machine trace:

```
Initial:
  prev_frame = { pyramid: ~, features: [] }
  curr_frame = { pyramid: ~, features: [] }
  has_curr = false, has_prev = false

push_frame(f0):
  swap (no-op effectively)
  curr_frame.pyramid.build_into(f0)
  curr_frame.features.clear()  // already empty
  has_curr = true
  State: prev = { ~, [] }, curr = { f0, [] }

good_features_to_track():
  detect on curr.pyramid.level(0) = f0
  curr_frame.features = features_in_f0 (each as Feature { x, y, strength })
  State: prev = { ~, [] }, curr = { f0, F0 }
  current_features() == F0, previous_features() == []

push_frame(f1):
  swap → prev = { f0, F0 }, curr = { ~, [] }  (old prev's features moved to curr)
  curr_frame.pyramid.build_into(f1)
  curr_frame.features.clear()  // discard stale features moved in by the swap
  has_prev = true
  State: prev = { f0, F0 }, curr = { f1, [] }

calculate_flow():
  staging_positions = positions from prev_frame.features (= F0 positions)
  lk_buffer.calc_into(prev=f0, curr=f1, &mut staging_positions, max_iter)
  curr_frame.features = zip(prev.features, staging).map(prev.strength + new pos)
  State: prev = { f0, F0 }, curr = { f1, F1_tracked }
  current_features() = features tracked into f1
  previous_features() = original features in f0
```

The clear-on-push in `push_frame` is load-bearing: without it, stale features
from two frames ago would resurface after each swap.

Equivalent Rust sketch:

```rust
// 0. Build:
let mut buf = OpticalFlowBuilder::new(W, H).build();
// buf.has_current_frame() == false, buf.has_previous_frame() == false

// 1. First frame primes curr; no flow yet.
buf.push_frame(&f0)?;
// has_current_frame() == true, has_previous_frame() == false
// current_features() is empty (push clears it)

// 2. Detect features on curr (= f0).
buf.good_features_to_track()?;
// current_features() == F0

// 3. Second frame rotates: prev = { f0, F0 }, curr = { f1, [] }
buf.push_frame(&f1)?;
// has_previous_frame() == true
// previous_features() == F0, current_features() is empty

// 4. Track from f0 -> f1.
buf.calculate_flow()?;
// current_features() = F1 (tracked positions, strength preserved)
// previous_features() = F0 (unchanged)

// 5. Continue for f2, f3, ...
buf.push_frame(&f2)?;
buf.calculate_flow()?;

// 6. Re-detect at any time after at least one push_frame.
buf.push_frame(&f3)?;
buf.good_features_to_track()?;  // re-seeds curr with fresh detections
```

Rules:
- `push_frame` always performs swap-then-build: `swap(prev_frame, curr_frame)` first, then `curr_frame.pyramid.build_into(image)`, then `curr_frame.features.clear()`. It never runs optical flow.
- `has_curr` flips to `true` on the first `push_frame` call.
- `has_prev` flips to `true` on the second `push_frame` call (i.e., when `has_curr` was already `true` at the start of `push_frame`).
- `good_features_to_track` requires `has_curr`; detects on `curr_frame.pyramid` and writes into `curr_frame.features`.
- `calculate_flow` requires `has_prev`; reads `prev_frame.features` positions, runs LK between `prev_frame.pyramid` and `curr_frame.pyramid`, writes `curr_frame.features` with preserved strengths.
- Feature lists travel with their owning `FrameBuffer` through the prev/curr swap, so `previous_features()` always reflects "the frame we just tracked from".

### 4.5 Re-exports

```rust
// src/lib.rs
pub use buffer::{OpticalFlowBuilder, OpticalFlowBuffer, TrackError};
```

## 5. Internal Types

### 5.1 `PyramidBuffer`

```rust
pub(crate) struct PyramidBuffer {
    levels: Vec<GrayImage>,
}

impl PyramidBuffer {
    /// Pre-allocate `n` levels sized for `(width, height)` and its halves.
    pub(crate) fn with_capacity(width: u32, height: u32, n: usize) -> Self;

    /// Rebuild from a `FlatSamples<&[u8]>` view, reusing existing storage.
    /// No heap allocation if the buffer was sized for the same resolution.
    pub(crate) fn build_into(&mut self, image: &FlatSamples<&[u8]>);

    pub(crate) fn levels(&self) -> &[GrayImage];
    pub(crate) fn level(&self, i: usize) -> &GrayImage;

    /// Consume the buffer and return its owned levels. Used by the legacy
    /// `pub fn build_pyramid(&GrayImage, _)` which still returns `Vec<GrayImage>`.
    pub(crate) fn into_levels(self) -> Vec<GrayImage>;
}
```

`build_into` writes pixel data into the existing `GrayImage` Vecs via `as_mut_slice().copy_from_slice(...)` for level 0 and the downsample loop for levels 1..n. The `Vec<u8>` backing each `GrayImage` keeps its capacity across calls.

### 5.2 `LkBuffer`

```rust
pub(crate) struct LkBuffer {
    // Per pyramid level — pre-sized at build time:
    grad_x: Vec<ImageBuffer<Luma<i16>, Vec<i16>>>,
    grad_y: Vec<ImageBuffer<Luma<i16>, Vec<i16>>>,

    // Per-call working buffers — sized once at build:
    prev_patch: Vec<f32>,
    ix_patch:   Vec<f32>,
    iy_patch:   Vec<f32>,

    // Per-call output — same length as `points`:
    displacements: Vec<(f32, f32)>,
}

impl LkBuffer {
    pub(crate) fn with_capacity(
        width: u32, height: u32, levels: usize, window_size: usize,
    ) -> Self;

    /// Compute flow from `prev_pyramid` to `curr_pyramid` for `points`,
    /// writing the updated positions back into `points` in place.
    /// No heap allocation.
    ///
    /// Takes pyramid slices (not `&PyramidBuffer`) so the legacy
    /// `pub fn calc_optical_flow(&[GrayImage], &[GrayImage], ...)` can call
    /// straight through without constructing a `PyramidBuffer`.
    pub(crate) fn calc_into(
        &mut self,
        prev_pyramid: &[GrayImage],
        curr_pyramid: &[GrayImage],
        points: &mut Vec<(f32, f32)>,
        window_size: usize,
        max_iterations: usize,
    );
}
```

`displacements` is resized to `points.len()` on entry via `resize(_, (0.0, 0.0))`, which only allocates if the new length exceeds the existing capacity. Same for any inner Vecs that depend on point count.

### 5.3 `FeaturesBuffer`

```rust
pub(crate) struct FeaturesBuffer {
    grad_x:   ImageBuffer<Luma<i16>, Vec<i16>>,
    grad_y:   ImageBuffer<Luma<i16>, Vec<i16>>,
    ix_sq:    ImageBuffer<Luma<i16>, Vec<i16>>,
    iy_sq:    ImageBuffer<Luma<i16>, Vec<i16>>,
    ix_iy:    ImageBuffer<Luma<i16>, Vec<i16>>,
    features: Vec<(u32, u32, f32)>,
    is_local_max: Vec<bool>,
    grid: Vec<Option<(u32, u32)>>,    // flattened grid, sized at build
}

impl FeaturesBuffer {
    /// Pre-allocate all buffers using best-effort upper bounds derived from
    /// the configured resolution and the builder's `min_distance`:
    /// - The five `ImageBuffer<Luma<i16>>` are exactly `width * height`.
    /// - `features` and `is_local_max` start with capacity `width * height`
    ///   (the worst case before non-maximum suppression).
    /// - `grid` is exactly `ceil(width / min_distance) * ceil(height / min_distance)`,
    ///   filled with `None`.
    ///
    /// For 1920x1080 with `min_distance = 10` this costs roughly:
    /// - 5 × 2 × W × H bytes for the i16 buffers ≈ 21 MB
    /// - features Vec ≈ 12 × W × H bytes ≈ 25 MB
    /// - is_local_max ≈ W × H bytes ≈ 2 MB
    /// - grid ≈ 192 × 108 × 24 bytes ≈ 0.5 MB
    ///
    /// Eager allocation is intentional — see §11.
    pub(crate) fn with_capacity(width: u32, height: u32, min_distance: u32) -> Self;

    /// Detect features on the provided `FlatSamples<&[u8]>` view (which is
    /// always level 0 of the source — Shi-Tomasi does not use a pyramid).
    /// Returns a slice into `self.features` containing the filtered, sorted
    /// results: `(x, y, min_eigenvalue)` triples. No heap allocation.
    ///
    /// Callers that want only `(x, y)` positions (the `OpticalFlowBuffer`
    /// case) iterate the slice and drop the third element. Callers that want
    /// the legacy `Vec<(u32, u32, f32)>` shape call `.to_vec()` on the slice.
    pub(crate) fn detect_into(
        &mut self,
        image: &FlatSamples<&[u8]>,
        quality_level: f32,
        min_distance: u32,
    ) -> &[(u32, u32, f32)];
}
```

The grid in `filter_by_distance` is currently `Vec<Vec<Option<(u32, u32)>>>` (one inner Vec per grid column). The buffer flattens this to a single `Vec<Option<(u32, u32)>>` of length `grid_width * grid_height`, indexed manually. This avoids the per-cell inner-Vec allocations entirely.

## 6. Internal Layering — No Code Duplication

Each algorithm has **one implementation**: the `*_into` method on the relevant
buffer type. Both the legacy free-function API and the new `OpticalFlowBuffer`
route through the same method.

### 6.1 The pattern

```
                  ┌───────────────────────────────────────────┐
                  │  Algorithm body (the one source of truth) │
                  │                                           │
                  │  PyramidBuffer::build_into                │
                  │  LkBuffer::calc_into                      │
                  │  FeaturesBuffer::detect_into              │
                  └─────────────────▲─────────────────────────┘
                                    │ takes &mut self + inputs
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
       Legacy free fns       generic::* fns       OpticalFlowBuffer
   (allocate + delegate)  (validate + allocate    (long-lived buffers;
                          + delegate)              just call *_into)
```

### 6.2 Concrete signatures (legacy wrappers)

**`pub fn build_pyramid(&GrayImage, usize) -> Vec<GrayImage>`** becomes:

```rust
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let (w, h) = image.dimensions();
    let mut buf = PyramidBuffer::with_capacity(w, h, levels);
    buf.build_into(&image.as_flat_samples());
    buf.into_levels()
}
```

**`pub fn generic::build_pyramid(&FlatSamples<B>, usize) -> Result<...>`** becomes:

```rust
pub fn build_pyramid<B: AsRef<[u8]>>(
    image: &FlatSamples<B>,
    levels: usize,
) -> Result<Vec<GrayImage>, LayoutError> {
    validate(image)?;
    let w = image.layout.width;
    let h = image.layout.height;
    let mut buf = PyramidBuffer::with_capacity(w, h, levels);
    buf.build_into(&thin(image));
    Ok(buf.into_levels())
}
```

**`pub fn calc_optical_flow(&[GrayImage], &[GrayImage], &[(f32, f32)], usize, usize) -> Vec<(f32, f32)>`** becomes:

```rust
pub fn calc_optical_flow(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    let (w, h) = prev_pyramid[0].dimensions();
    let levels = prev_pyramid.len();
    let mut buf = LkBuffer::with_capacity(w, h, levels, window_size);
    let mut points = prev_points.to_vec();
    buf.calc_into(
        prev_pyramid, curr_pyramid, &mut points, window_size, max_iterations,
    );
    points
}
```

**`pub fn good_features_to_track(&GrayImage, f32, u32) -> Vec<(u32, u32, f32)>`** becomes:

```rust
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    let (w, h) = image.dimensions();
    let mut buf = FeaturesBuffer::with_capacity(w, h, min_distance);
    buf.detect_into(&image.as_flat_samples(), quality_level, min_distance)
       .to_vec()
}
```

**`pub fn generic::good_features_to_track(&FlatSamples<B>, f32, u32) -> Result<...>`** — analogous.

### 6.3 New surface (`OpticalFlowBuffer::push_frame`)

The long-lived buffers are reused across frames; no allocation:

```rust
impl OpticalFlowBuffer {
    pub fn push_frame<B: AsRef<[u8]>>(
        &mut self,
        image: &FlatSamples<B>,
    ) -> Result<(), TrackError> {
        // 1. Validate (resolution match + layout rules).
        check_dimensions(image, self.width, self.height)?;
        let thinned = thin_validated(image);   // shared with generic::*

        // 2. Build new pyramid into curr_pyramid (no alloc).
        self.curr_pyramid.build_into(&thinned);

        // 3. If we have a previous frame and points to track, run LK.
        if self.has_prev_frame && !self.points.is_empty() {
            self.lk_buffer.calc_into(
                self.prev_pyramid.levels(),
                self.curr_pyramid.levels(),
                &mut self.points,
                self.window_size,
                self.max_iterations,
            );
        }

        // 4. Swap so prev_pyramid now represents the just-pushed frame.
        std::mem::swap(&mut self.prev_pyramid, &mut self.curr_pyramid);
        self.has_prev_frame = true;
        Ok(())
    }

    pub fn reset_with_good_features_to_track(&mut self) -> Result<(), TrackError> {
        if !self.has_prev_frame {
            return Err(TrackError::NoPreviousFrame);
        }
        let level0 = self.prev_pyramid.level(0);
        let detected = self.features_buffer.detect_into(
            &level0.as_flat_samples(),
            self.feature_quality_level,
            self.feature_min_distance,
        );
        self.points.clear();
        self.points.extend(detected.iter().map(|&(x, y, _)| (x as f32, y as f32)));
        Ok(())
    }
}
```

`push_frame`, `reset`, and `reset_with_good_features_to_track` together contain
**zero algorithm code** — only orchestration. All real work lives in the three
`*_into` methods.

### 6.4 Refactor order (implementation steps for the plan doc)

1. **`PyramidBuffer`** — new struct in `src/pyramid.rs`. Move the body of today's
   `build_pyramid_impl` into `PyramidBuffer::build_into`. Existing
   `build_pyramid(&GrayImage, _)` and `generic::build_pyramid` become 5-line
   wrappers per §6.2. The current `pub(crate) fn build_pyramid_impl(...)`
   disappears — replaced by `PyramidBuffer::build_into`.
2. **`LkBuffer`** — new struct in `src/lk.rs`. Move the body of today's
   `calc_optical_flow` into `LkBuffer::calc_into`. The free function becomes a
   5-line wrapper per §6.2. The `prev_patch`/`ix_patch`/`iy_patch` Vecs (today
   re-allocated per pyramid level inside the function) become fields on
   `LkBuffer`, sized at `with_capacity` for the configured window.
3. **`FeaturesBuffer`** — new struct in `src/features.rs`. Move the body of
   today's `good_features_to_track_impl` into `FeaturesBuffer::detect_into`.
   The free function and `generic::good_features_to_track` become wrappers.
   The `Vec<Vec<Option<...>>>` grid is replaced with the flat representation
   from §5.3. Today's `compute_gradient_products`, `compute_min_eigenvalues`,
   `non_maximum_suppression`, `filter_by_quality`, `filter_by_distance` either
   become private methods on `FeaturesBuffer` or remain free functions
   operating on `&mut self`'s fields — implementer's choice, no behavioral
   change required.
4. **`OpticalFlowBuilder`** and **`OpticalFlowBuffer`** — new module
   `src/buffer.rs`. Composes the three buffer types and orchestrates the
   swap/reset/push lifecycle. The whole module is the code shown in §4 and §6.3.
5. **`TrackError`** — extend `src/error.rs` (which already houses `LayoutError`).
6. **Tests** — see §7.
7. **Bench** — add a `tracking_loop` bench measuring per-frame cost on 1920×1080
   over 10 frames with `OpticalFlowBuffer` vs. the existing free-function pipeline.
8. **Allocation verification** — see §9.

### 6.5 What this buys

- **One implementation per algorithm.** Reading `LkBuffer::calc_into` tells you
  exactly how Lucas-Kanade works in this crate. No duplicated math.
- **Free-function APIs stay 5–10 lines.** They're trivially correct by
  inspection: allocate a buffer, call `*_into`, return.
- **Allocation profile is a property of the wrapper, not the algorithm.** The
  algorithm is allocation-free by construction; the *wrapper* decides whether
  to allocate fresh or reuse.

## 7. Tests

### 7.1 Existing tests stay green

Same principle as v0.3.0: the existing public API is unchanged, so all v0.3.0 tests (gradient correctness, validation, LK math) must continue to pass without modification.

### 7.2 New unit tests

- `PyramidBuffer::build_into` produces level dimensions `(W, H), (W/2, H/2), ...` and the same level 0 bytes as the input `FlatSamples`.
- `PyramidBuffer::build_into` called twice with the same dimensions does not change the `Vec<u8>::capacity()` of any level (no reallocation). Verified by snapshotting capacities before and after.
- `LkBuffer::calc_into` on a contrived prev/curr pair with one tracked point produces the same result as the legacy `calc_optical_flow` free function (which now delegates to it — so this is effectively a smoke test that the delegation is wired correctly).
- `FeaturesBuffer::detect_into` on a fixture image returns a slice whose contents (after `.to_vec()`) match `good_features_to_track`.

### 7.3 New integration tests for `OpticalFlowBuffer`

- **`lifecycle_push_detect_push_track`**: push f0 → detect into `points` → push f1 → `calculate_flow(&points, &mut tracked)`. Assert `has_current_frame()` is `false` initially, `true` after first push; `has_previous_frame()` is `false` after one push, `true` after two. Assert `points` is non-empty after detect, `tracked.len() == points.len()` after flow.
- **`good_features_errors_before_any_push`**: call `good_features_to_track` with no prior push; assert `TrackError::NoCurrentFrame`.
- **`calculate_flow_errors_before_two_pushes`**: assert `NoPreviousFrame` with 0 pushes, assert `NoPreviousFrame` with 1 push, assert `Ok` after 2 pushes.
- **`dimension_mismatch_error`**: build for (100, 50), push a (200, 50) `FlatSamples`, assert `TrackError::DimensionMismatch { expected: (100, 50), actual: (200, 50) }`.
- **`layout_error_unsupported_channels`**: build for (8, 8), push a `FlatSamples` with `channels = 3`, assert `TrackError::Layout(LayoutError::UnsupportedChannels(3))`.
- **`layout_error_unsupported_width_stride`**: assert `LayoutError::UnsupportedWidthStride(2)`.
- **`layout_error_overlapping_rows`**: assert `LayoutError::OverlappingRows`.
- **`layout_error_buffer_too_small`**: assert `LayoutError::BufferTooSmall`.
- **`accessors_return_builder_values`**: `dimensions()`, `window_size()`, `max_iterations()`, `feature_quality_level()`, `feature_min_distance()`, `pyramid_levels()` all return what the builder configured.

### 7.4 Allocation behavior

Allocation behavior is verified in a dedicated section — see §9. Briefly: a
separate integration test binary with a counting `#[global_allocator]` runs
two scenarios — one for the v0.3.0 `generic::*` path (must allocate per call)
and one for the new `OpticalFlowBuffer` path (must not allocate after warm-up).

## 8. Migration / Compatibility

This release is purely additive at the public-API level. Callers using v0.3.0 functions continue to work.

Internally, `build_pyramid`, `good_features_to_track`, `calc_optical_flow`, and
the `generic::*` overloads are refactored to delegate to the new buffer types
via the `*_into` methods described in §6. Each legacy entry point allocates a
fresh `PyramidBuffer` / `LkBuffer` / `FeaturesBuffer` per call (so observable
behavior — including the per-call allocations — is unchanged from the caller's
perspective) and discards it on return. Callers seeking the steady-state
allocation profile migrate to `OpticalFlowBuilder`.

After this refactor, every algorithm in the crate has exactly one
implementation — the `*_into` method on its buffer. The legacy free functions
contain orchestration only (allocate, call, return). The current
`pyramid::build_pyramid_impl` and `features::good_features_to_track_impl`
`pub(crate)` helpers introduced in v0.3.0 are removed; their bodies move into
the buffer methods.

A short migration note goes in the README and CHANGELOG: "For real-time loops, prefer `OpticalFlowBuilder` to avoid per-frame allocations."

## 9. Verification

Two questions need empirical answers, separate from the unit-test correctness checks in §7:

1. The legacy `generic::*` path still allocates per call. **How many** allocations, and is that number stable across changes?
2. The new `OpticalFlowBuffer` path is supposed to be allocation-free after warm-up. **Is it actually**, or is there a hidden Vec growth, intermediate collection, or accidental `to_vec()` somewhere in the pipeline?

Both are verified by a counting-allocator integration test. Benchmarks measure wall time but say nothing about allocation count; a unit-style test with hard assertions catches regressions on the first line of unexpected behavior, in any CI run.

### 9.1 Approach — counting `#[global_allocator]` in a dedicated test binary

Each Rust integration test compiles to its own binary, which means each binary can declare its own `#[global_allocator]`. A small allocator wrapping `std::alloc::System` with `AtomicUsize` counters gives a per-process count of `alloc` and `dealloc` calls.

The allocator and its tests live in `tests/allocations.rs`. Lib tests, doc tests, and other integration tests are unaffected because they compile to separate binaries.

```rust
// tests/allocations.rs

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

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
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

fn allocs() -> usize { ALLOC_COUNT.load(Ordering::Relaxed) }

fn measure<F: FnOnce()>(f: F) -> usize {
    let before = allocs();
    f();
    allocs() - before
}
```

The tests in this binary must run single-threaded so the counter is not raced. This is enforced by adding to `Cargo.toml`:

```toml
[[test]]
name = "allocations"
harness = true
```

…and running with `cargo test --test allocations -- --test-threads=1`. The plan's verification step uses that command verbatim, and the CHANGELOG snippet for v0.4.0 mentions the flag for anyone running it locally.

### 9.2 Test case A — v0.3.0 `generic::*` baseline

Upper-bound assertion only. The implementer measures the actual allocation count
after the v0.4.0 refactor and sets `MAX_*` to that exact value. The bound is a
**regression tripwire**, not an invariant: an increase fails the test and
forces a human to review whether the new allocation is justified. A decrease
(e.g., a future optimization removes a Vec) passes the test silently — that's
fine, an improvement is not a regression.

```rust
// Measured on $HOST after the v0.4.0 refactor. Bump only after deliberate review.
const MAX_ALLOCS_GENERIC_BUILD_PYRAMID: usize = /* fill in from cargo test output */;
const MAX_ALLOCS_GENERIC_FEATURES:      usize = /* fill in from cargo test output */;

#[test]
fn generic_build_pyramid_allocations_within_bound() {
    let buf = vec![0u8; 256 * 256];
    let view = make_flat_samples(&buf, 256, 256);   // helper

    // Warm up — first call may trigger one-time allocator/runtime setup.
    let _ = optical_flow_lk::generic::build_pyramid(&view, 3).unwrap();

    let n_alloc = measure(|| {
        let _ = optical_flow_lk::generic::build_pyramid(&view, 3).unwrap();
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
    let buf = vec![0u8; 256 * 256];
    let view = make_flat_samples(&buf, 256, 256);
    let _ = optical_flow_lk::generic::good_features_to_track(&view, 0.4, 10).unwrap();

    let n_alloc = measure(|| {
        let _ = optical_flow_lk::generic::good_features_to_track(&view, 0.4, 10).unwrap();
    });

    assert!(
        n_alloc <= MAX_ALLOCS_GENERIC_FEATURES,
        "generic::good_features_to_track allocated {n_alloc} times \
         (upper bound {MAX_ALLOCS_GENERIC_FEATURES}). \
         If this increase is intentional and justified, bump the bound \
         after reviewing the new allocation site."
    );
}
```

**How to set the initial bounds.** When implementing v0.4.0:

1. Write the test with `MAX_* = 0` (so it fails).
2. Run `cargo test --test allocations -- --test-threads=1 --nocapture` and read the failure message — it prints the measured `n_alloc`.
3. Set `MAX_*` to that measured value.
4. Re-run; the test now passes.
5. Document the measured value in a one-line comment so reviewers know where it came from.

The point of the bound is not to enumerate every allocation but to catch
silent regressions. If the count creeps from 5 to 6 because someone added a
`.collect()` to the hot path, the test fails and the change becomes a
conversation. If the count drops from 5 to 3 because someone removed a `Vec`,
the test silently passes — improvements are welcome without ceremony.

### 9.3 Test case B — new `OpticalFlowBuffer` steady state

```rust
#[test]
fn buffer_path_is_steady_state_zero_alloc() {
    use optical_flow_lk::{OpticalFlowBuilder, FlatSamples, SampleLayout};

    const W: u32 = 256;
    const H: u32 = 256;

    let frame_a = vec![0u8; (W * H) as usize];
    let frame_b = vec![1u8; (W * H) as usize];
    let view_a = make_flat_samples(&frame_a, W, H);
    let view_b = make_flat_samples(&frame_b, W, H);

    let mut buf = OpticalFlowBuilder::new(W, H).build();

    // Warm-up:
    //  - first push primes prev_pyramid (no flow)
    //  - reset_with_good_features detects on it
    //  - second push computes flow for the first time, which may grow
    //    the points-derived intermediate Vecs to their steady-state size.
    buf.push_frame(&view_a).unwrap();
    buf.reset_with_good_features_to_track().unwrap();
    buf.push_frame(&view_b).unwrap();
    buf.push_frame(&view_a).unwrap();
    buf.push_frame(&view_b).unwrap();

    // Measurement phase:
    let n_alloc = measure(|| {
        for _ in 0..10 {
            buf.push_frame(&view_a).unwrap();
            buf.push_frame(&view_b).unwrap();
        }
    });

    assert_eq!(
        n_alloc, 0,
        "OpticalFlowBuffer::push_frame allocated {n_alloc} times in 20 frames \
         after warm-up; expected 0. Inspect for hidden Vec growth, intermediate \
         collections, or accidental .to_vec()/.collect() calls."
    );
}

#[test]
fn buffer_path_reset_with_good_features_is_zero_alloc_after_warmup() {
    use optical_flow_lk::OpticalFlowBuilder;

    const W: u32 = 256;
    const H: u32 = 256;
    let frame = vec![0u8; (W * H) as usize];
    let view = make_flat_samples(&frame, W, H);

    let mut buf = OpticalFlowBuilder::new(W, H).build();

    buf.push_frame(&view).unwrap();
    // Warm-up: the first detect may grow the points Vec to its steady size.
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
```

### 9.4 What zero-alloc actually guarantees

The custom allocator counts every call to `std::alloc::alloc` /
`std::alloc::dealloc`, which covers all `Vec`, `Box`, `String`, etc. growth.
It does *not* count:

- Stack allocations.
- Operations that re-use existing capacity (`Vec::clear`, `Vec::extend` within
  capacity, `slice::copy_from_slice`, `Vec::resize` within capacity).

This is exactly the boundary we care about: the new buffer types are required
to operate within their pre-allocated capacities after warm-up. If they do,
the allocator sees zero `alloc` calls in the measurement phase.

### 9.5 If the buffer-path tests ever fail

When a steady-state allocation appears, the most common culprits are:

- A `.to_vec()` / `.collect()` newly introduced in the hot path.
- `Vec::resize` or `Vec::extend` past the buffer's pre-allocated capacity (size the buffer's capacity at `with_capacity` to the worst case).
- A new intermediate `Vec` declared inside a per-frame method instead of as a field on the buffer.
- A `format!()` / `panic!()` triggered on an unexpected code path.

The test's failure message points implementers at these categories so the fix is mechanical.

## 10. Versioning

Recommended version bump: **0.4.0** (minor — new public surface, no breaking changes).

CHANGELOG entry under "Added":
- `OpticalFlowBuilder`, `OpticalFlowBuffer`, `TrackError`.
- Allocation-free per-frame tracking after warm-up.

Under "Changed":
- Internal `build_pyramid`, `good_features_to_track`, `calc_optical_flow` are now thin wrappers over reusable buffer types. Behavior unchanged.

## 11. Risks and Open Questions

- **Should `OpticalFlowBuffer` own the input data?** Today `push_frame(&FlatSamples)` borrows. The buffer copies level 0 into its pyramid, so the input can be dropped immediately after. Good.

- **Resizing pyramid pre-allocations.** If the configured resolution is larger than the actual input, the buffers are oversized but functional? Currently no — the input is validated against the configured `(width, height)` exactly. Resizing inputs isn't supported in v0.4.0; if the resolution changes mid-stream the caller rebuilds the buffer. Documented in `OpticalFlowBuilder::new` rustdoc.

- **Feature detection buffer sizing.** `FeaturesBuffer` is allocated eagerly at `build()` with best-effort upper-bound sizes (see §5.3). For 1920×1080 this costs roughly 50 MB total across the i16 buffers, the features Vec, the is_local_max Vec, and the grid. This is intentional: the whole point of the buffer type is predictable steady-state allocation, including the first frame after a `good_features_to_track()` call. Callers that cannot afford this upfront cost should keep using the free-function `good_features_to_track` instead.

- **Buffer-owned feature lists.** `OpticalFlowBuffer` owns two `Vec<Feature>` slices (one per `FrameBuffer`) and a `staging_positions: Vec<(f32, f32)>` scratch buffer for the LK call. All three grow during warm-up to their steady-state size and are reused thereafter. Callers read features through `current_features()` and `previous_features()` without any ownership transfer. This is simpler than the previous caller-managed-Vec design and preserves zero-alloc behavior after warm-up.

- **Thread safety.** `OpticalFlowBuffer` is naturally `Send` if its components are. We do not promise `Sync`; concurrent `push_frame` is undefined. Documented.

- **`reset_with_good_features_to_track` semantics on a strided input.** Detection runs on `prev_pyramid.level(0)` which is always a contiguous owned `GrayImage` (copied from the input on push). So the detection result is in level-0 pixel coordinates, which match the input view's coordinates. No ambiguity.

## 12. Out-of-Scope Follow-Ups (not part of this spec)

- Borrowed level 0 in the pyramid (still on the v0.3.0 follow-up list).
- Multi-resolution support in a single buffer.
- A `Sync`-safe variant for parallel tracking.
- GPU offload of the pyramid / gradient stages.
- A `points_status` parallel Vec to expose per-point tracking validity. (Currently dropped points are silently retained; a `Vec<bool>` flagging "tracked vs. lost" would be cleaner. Deferred.)
