//! HIP C++ source lowering.

#[cfg(feature = "hip-runtime")]
pub mod runtime;

use crate::backend::native::define_native_backend;
use crate::config::Backend;

define_native_backend!(HipSourceBackend, Backend::Hip);
