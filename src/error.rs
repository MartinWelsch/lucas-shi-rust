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
