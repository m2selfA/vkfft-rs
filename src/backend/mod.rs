//! Backend lowering interfaces.
//!
//! Backends consume [`KernelIr`] rather than planner
//! internals. This keeps the algorithm/scheduling layer independent from Vulkan,
//! CUDA, HIP, OpenCL, Level Zero, or Metal syntax and runtime ownership.

use crate::error::Result;
use crate::kernel_ir::KernelIr;

pub mod cuda;
pub mod hip;
pub mod level_zero;
pub mod metal;
pub(crate) mod native;
#[cfg(feature = "native-runtime")]
pub mod native_runtime;
pub mod opencl;
pub mod vulkan;

pub use native::{NativeProgramSource, NativeShaderSource, NativeSourceBackend};
#[cfg(feature = "native-runtime")]
pub use native_runtime::{
    NativeAsyncRuntime, NativeAsyncTransformTicket32, NativeAsyncTransformTicket64,
    NativeCompiledPassResourceReport, NativeCompiledResourceMetrics,
    NativeOccupancyLimitingResource, NativeProgramTicket32, NativeProgramTicket64, NativeRuntime,
    NativeRuntimeAvailability, NativeTheoreticalOccupancyReport, NativeTransformInput32,
    NativeTransformInput64, NativeTransformOutput32, NativeTransformOutput64,
};

#[cfg(all(
    test,
    any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )
))]
#[derive(Debug, Default)]
struct GpuTestContextState {
    owner: Option<std::thread::ThreadId>,
    depth: usize,
}

#[cfg(all(
    test,
    any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )
))]
fn gpu_test_context_lock() -> &'static (std::sync::Mutex<GpuTestContextState>, std::sync::Condvar) {
    static LOCK: std::sync::OnceLock<(std::sync::Mutex<GpuTestContextState>, std::sync::Condvar)> =
        std::sync::OnceLock::new();
    LOCK.get_or_init(|| {
        (
            std::sync::Mutex::new(GpuTestContextState::default()),
            std::sync::Condvar::new(),
        )
    })
}

#[cfg(all(
    test,
    any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )
))]
#[derive(Debug)]
pub(crate) struct GpuTestContextGuard {
    owner: std::thread::ThreadId,
}

#[cfg(all(
    test,
    any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )
))]
pub(crate) fn gpu_test_context_guard() -> GpuTestContextGuard {
    let owner = std::thread::current().id();
    let (lock, wake) = gpu_test_context_lock();
    let mut state = match lock.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    loop {
        match state.owner.as_ref() {
            None => {
                state.owner = Some(owner);
                state.depth = 1;
                break;
            }
            Some(current) if current == &owner => {
                state.depth += 1;
                break;
            }
            Some(_) => {
                state = match wake.wait(state) {
                    Ok(state) => state,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
        }
    }
    GpuTestContextGuard { owner }
}

#[cfg(all(
    test,
    any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )
))]
impl Drop for GpuTestContextGuard {
    fn drop(&mut self) {
        let (lock, wake) = gpu_test_context_lock();
        let mut state = match lock.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert_eq!(state.owner.as_ref(), Some(&self.owner));
        assert!(state.depth > 0);
        state.depth -= 1;
        if state.depth == 0 {
            state.owner = None;
            wake.notify_one();
        }
    }
}

pub trait KernelBackend {
    type Output;

    fn lower(&self, kernel: &KernelIr) -> Result<Self::Output>;
}
