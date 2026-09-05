//! Metal Shading Language source lowering.

#[cfg(feature = "metal-runtime")]
pub mod runtime;

use crate::backend::native::define_native_backend;
use crate::config::Backend;

define_native_backend!(MetalSourceBackend, Backend::Metal);
