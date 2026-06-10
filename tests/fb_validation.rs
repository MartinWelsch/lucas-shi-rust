//! Forward-backward (FB) validation tests for `calculate_flow_fb`.
//!
//! A feature sitting on well-textured content that translates consistently
//! should survive the round-trip error gate; a feature whose window leaves
//! the image (out of bounds) or sits on flat/ambiguous texture should be
//! reported invalid.

use image::flat::{FlatSamples, SampleLayout};
use optical_flow_lk::{Feature, OpticalFlowBuilder};

fn make_view(buf: &[u8], width: u32, height: u32) -> FlatSamples<&[u8]> {
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

/// A frame with a strong, unambiguous textured patch around `(cx, cy)` on an
/// otherwise flat mid-gray background. The patch is a smooth, asymmetric
/// intensity bump (a 2-D ramp clipped to a disk): it has well-conditioned
/// gradients on *both* axes and survives 2x2-average pyramid downsampling,
/// so LK locks onto it cleanly at every level. Flat mid-gray everywhere
/// else gives an ambiguous (singular-Hessian) region for the negative
/// control.
fn frame_with_patch(width: u32, height: u32, cx: i32, cy: i32) -> Vec<u8> {
    let mut out = vec![128u8; (width * height) as usize];
    let r = 7i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let x = cx + dx;
            let y = cy + dy;
            if dx * dx + dy * dy > r * r {
                continue;
            }
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                // Asymmetric ramp: distinct gradient on both axes, so the
                // 2x2 Hessian is well-conditioned.
                let v = 128 + 8 * dx + 5 * dy;
                out[(y as u32 * width + x as u32) as usize] = v.clamp(0, 255) as u8;
            }
        }
    }
    out
}

#[test]
fn fb_survives_consistent_translation_prunes_flat_and_oob() {
    const W: u32 = 64;
    const H: u32 = 64;
    const DX: i32 = 2;
    const DY: i32 = 1;

    // Frame A: textured patch at (30, 30). Frame B: same patch shifted by
    // (DX, DY).
    let frame_a = frame_with_patch(W, H, 30, 30);
    let frame_b = frame_with_patch(W, H, 30 + DX, 30 + DY);

    let mut tracker = OpticalFlowBuilder::new(W, H)
        .pyramid_levels(2)
        .window_size(7)
        .max_iterations(20)
        .build();

    tracker.push_frame(&make_view(&frame_a, W, H)).unwrap();

    // Three features:
    //   0: on the textured patch -> should track and survive.
    //   1: on flat background -> ambiguous, should fail the round trip.
    //   2: forced out of bounds (window leaves the image) -> must fail.
    tracker.features_mut().push(Feature { x: 30.0, y: 30.0, strength: 1.0 });
    tracker.features_mut().push(Feature { x: 10.0, y: 55.0, strength: 1.0 });
    tracker.features_mut().push(Feature { x: 2.0, y: 2.0, strength: 1.0 });

    tracker.push_frame(&make_view(&frame_b, W, H)).unwrap();

    let mut valid = Vec::new();
    tracker.calculate_flow_fb(1.0, &mut valid).unwrap();

    assert_eq!(valid.len(), 3, "one validity flag per feature");
    assert!(valid[0], "textured patch feature should survive FB validation");
    assert!(!valid[1], "flat-texture feature should be pruned");
    assert!(!valid[2], "out-of-bounds feature should be pruned");

    // The surviving feature actually moved by ~ (DX, DY).
    let f0 = tracker.features()[0];
    assert!((f0.x - (30.0 + DX as f32)).abs() < 1.0);
    assert!((f0.y - (30.0 + DY as f32)).abs() < 1.0);
}

#[test]
fn fb_rejects_stuck_oob_feature_even_with_zero_motion() {
    // A feature whose window is out of bounds never moves; FB must mark it
    // invalid even when the rest of the scene is static (round-trip error
    // for a stuck feature is 0, so the gate alone would pass it — the
    // skipped-flow detection is what rejects it).
    const W: u32 = 32;
    const H: u32 = 32;
    let frame = frame_with_patch(W, H, 16, 16);

    let mut tracker = OpticalFlowBuilder::new(W, H)
        .pyramid_levels(2)
        .window_size(7)
        .max_iterations(10)
        .build();

    tracker.push_frame(&make_view(&frame, W, H)).unwrap();
    // (1, 1) with radius 3 window -> out of bounds at level 0.
    tracker.features_mut().push(Feature { x: 1.0, y: 1.0, strength: 1.0 });
    tracker.push_frame(&make_view(&frame, W, H)).unwrap();

    let mut valid = Vec::new();
    tracker.calculate_flow_fb(2.0, &mut valid).unwrap();

    assert_eq!(valid.len(), 1);
    assert!(
        !valid[0],
        "stuck out-of-bounds feature must fail FB validation despite zero round-trip error"
    );
}

#[test]
fn fb_errors_before_two_pushes() {
    let mut tracker = OpticalFlowBuilder::new(16, 16).build();
    let frame = vec![0u8; 16 * 16];
    let mut valid = Vec::new();
    assert!(tracker.calculate_flow_fb(1.0, &mut valid).is_err());

    tracker.push_frame(&make_view(&frame, 16, 16)).unwrap();
    assert!(tracker.calculate_flow_fb(1.0, &mut valid).is_err());
}
