//! CUDA C++ source lowering.

#[cfg(feature = "cuda-runtime")]
pub mod runtime;

use crate::backend::native::define_native_backend;
use crate::config::Backend;

define_native_backend!(CudaSourceBackend, Backend::Cuda);
