use core::fmt;

pub type Result<T> = core::result::Result<T, VkFftError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VkFftError {
    EmptyDimensions,
    ZeroLength {
        axis: usize,
    },
    ZeroBatchCount,
    ArithmeticOverflow {
        operation: &'static str,
    },
    InvalidTransformLength {
        axis: usize,
        transform: &'static str,
        length: usize,
    },
    InvalidKernelIr(&'static str),
    InvalidLut(&'static str),
    UnsupportedKernelPath(&'static str),
    UnsupportedPrecision {
        backend: &'static str,
        precision: &'static str,
    },
    ResourceLimitExceeded {
        resource: &'static str,
        required: usize,
        available: usize,
    },
    InputLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ValueOutOfRange {
        field: &'static str,
    },
    InvalidZeroPaddingRange {
        axis: usize,
        left: usize,
        right: usize,
        length: usize,
    },
    ShaderCompilation(String),
    VulkanUnavailable(String),
    VulkanRuntime(String),
    NativeUnavailable {
        backend: &'static str,
        message: String,
    },
    NativeRuntime {
        backend: &'static str,
        message: String,
    },
    InvalidPlannerTuning(&'static str),
}

impl fmt::Display for VkFftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => write!(f, "FFT dimensions must not be empty"),
            Self::ZeroLength { axis } => write!(f, "FFT axis {axis} has zero length"),
            Self::ZeroBatchCount => write!(f, "FFT batch count must be greater than zero"),
            Self::ArithmeticOverflow { operation } => {
                write!(f, "integer overflow while computing {operation}")
            }
            Self::InvalidTransformLength {
                axis,
                transform,
                length,
            } => write!(f, "axis {axis} length {length} is invalid for {transform}"),
            Self::InvalidKernelIr(message) => write!(f, "invalid kernel IR: {message}"),
            Self::InvalidLut(message) => write!(f, "invalid FFT lookup table: {message}"),
            Self::UnsupportedKernelPath(message) => write!(f, "unsupported kernel path: {message}"),
            Self::UnsupportedPrecision { backend, precision } => {
                write!(
                    f,
                    "{backend} backend does not yet support precision {precision}"
                )
            }
            Self::ResourceLimitExceeded {
                resource,
                required,
                available,
            } => write!(
                f,
                "{resource} requires {required} bytes/units but only {available} are available"
            ),
            Self::InputLengthMismatch { expected, actual } => write!(
                f,
                "input contains {actual} complex elements but kernel expects {expected}"
            ),
            Self::ValueOutOfRange { field } => {
                write!(
                    f,
                    "{field} cannot be represented by the current backend index type"
                )
            }
            Self::InvalidZeroPaddingRange {
                axis,
                left,
                right,
                length,
            } => write!(
                f,
                "zero-padding interval [{left}, {right}) is invalid for axis {axis} of length {length}"
            ),
            Self::ShaderCompilation(message) => {
                write!(f, "shader compilation failed: {message}")
            }
            Self::VulkanUnavailable(message) => write!(f, "Vulkan is unavailable: {message}"),
            Self::VulkanRuntime(message) => write!(f, "Vulkan runtime error: {message}"),
            Self::NativeUnavailable { backend, message } => {
                write!(f, "{backend} runtime is unavailable: {message}")
            }
            Self::NativeRuntime { backend, message } => {
                write!(f, "{backend} runtime error: {message}")
            }
            Self::InvalidPlannerTuning(message) => write!(f, "invalid planner tuning: {message}"),
        }
    }
}

impl std::error::Error for VkFftError {}
