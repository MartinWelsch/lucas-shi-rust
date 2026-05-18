use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    UnsupportedChannels(u8),
    UnsupportedWidthStride(usize),
    OverlappingRows { height_stride: usize, width: u32 },
    BufferTooSmall { required: usize, actual: usize },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::UnsupportedChannels(c) =>
                write!(f, "unsupported channel count: {} (expected 1)", c),
            LayoutError::UnsupportedWidthStride(s) =>
                write!(f, "unsupported width_stride: {} (expected 1)", s),
            LayoutError::OverlappingRows { height_stride, width } =>
                write!(f, "rows overlap: height_stride={} < width={}", height_stride, width),
            LayoutError::BufferTooSmall { required, actual } =>
                write!(f, "samples buffer too small: required={} actual={}", required, actual),
        }
    }
}

impl std::error::Error for LayoutError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_render() {
        assert_eq!(
            LayoutError::UnsupportedChannels(3).to_string(),
            "unsupported channel count: 3 (expected 1)"
        );
        assert_eq!(
            LayoutError::OverlappingRows { height_stride: 4, width: 8 }.to_string(),
            "rows overlap: height_stride=4 < width=8"
        );
    }
}

/// Errors returned by [`OpticalFlowBuffer`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackError {
    /// The provided `FlatSamples` failed the layout checks performed by
    /// [`OpticalFlowBuffer::push_frame`] — channels, stride, buffer length.
    Layout(LayoutError),

    /// `image.layout.width` or `.height` did not match the buffer's
    /// configured resolution.
    DimensionMismatch { expected: (u32, u32), actual: (u32, u32) },

    /// `calculate_flow` was called before two frames had been pushed,
    /// so there is no `prev` frame to track from.
    NoPreviousFrame,

    /// `detect_features` was called before any frame had been pushed,
    /// so there is nothing to detect features on.
    NoCurrentFrame,
}

impl From<LayoutError> for TrackError {
    fn from(value: LayoutError) -> Self {
        TrackError::Layout(value)
    }
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackError::Layout(err) => write!(f, "layout error: {err}"),
            TrackError::DimensionMismatch { expected, actual } => write!(
                f,
                "dimension mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            TrackError::NoPreviousFrame =>
                write!(f, "no previous frame pushed yet"),
            TrackError::NoCurrentFrame =>
                write!(f, "no current frame pushed yet"),
        }
    }
}

impl std::error::Error for TrackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrackError::Layout(err) => Some(err),
            _ => None,
        }
    }
}

#[cfg(test)]
mod track_error_tests {
    use super::*;

    #[test]
    fn track_error_wraps_layout_error_via_from() {
        let layout = LayoutError::UnsupportedChannels(4);
        let track: TrackError = layout.clone().into();
        assert_eq!(track, TrackError::Layout(layout));
    }

    #[test]
    fn track_error_display_renders() {
        assert_eq!(
            TrackError::DimensionMismatch { expected: (10, 20), actual: (30, 40) }.to_string(),
            "dimension mismatch: expected (10, 20), got (30, 40)"
        );
        assert_eq!(
            TrackError::NoPreviousFrame.to_string(),
            "no previous frame pushed yet"
        );
        assert_eq!(
            TrackError::NoCurrentFrame.to_string(),
            "no current frame pushed yet"
        );
    }
}
