//! OpenCL C source lowering.

#[cfg(feature = "opencl-runtime")]
pub mod runtime;

use crate::backend::native::define_native_backend;
use crate::config::Backend;

define_native_backend!(OpenClSourceBackend, Backend::OpenCl);
