//! Public `Feature` type — a tracked point with position and detection strength.

/// A tracked feature point with spatial position and detection strength.
///
/// `strength` is the Shi-Tomasi min-eigenvalue from feature detection
/// (always >= 0).
/// [`OpticalFlowTracker::calculate_flow`](crate::OpticalFlowTracker::calculate_flow)
/// leaves `strength` untouched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feature {
    /// Sub-pixel x coordinate in image space.
    pub x: f32,
    /// Sub-pixel y coordinate in image space.
    pub y: f32,
    /// Shi-Tomasi min-eigenvalue from detection (≥ 0). Tracking leaves it
    /// untouched.
    pub strength: f32,
}
