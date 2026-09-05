//! Level Zero/OpenCL-C kernel source lowering.
//!
//! VkFFT's Level Zero backend consumes OpenCL-C-like kernel source before the
//! Level Zero compiler/runtime layer, so this backend intentionally shares that
//! language surface while retaining distinct backend metadata and scheduling.

#[cfg(feature = "level-zero-runtime")]
pub mod runtime;

use crate::backend::native::define_native_backend;
use crate::config::Backend;

define_native_backend!(LevelZeroSourceBackend, Backend::LevelZero);
