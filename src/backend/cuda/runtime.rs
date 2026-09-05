//! CUDA Driver API + NVRTC correctness-first runtime.
//!
//! The runtime is loaded dynamically: enabling `cuda-runtime` does not link CUDA at
//! build time. Each ProgramIr pass is NVRTC-compiled to PTX, loaded through the CUDA
//! Driver API, and launched in order on the default stream.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Mutex;

use libloading::Library;

use crate::backend::native::{NativeProgramSource, NativeShaderSource, NativeSourceBackend};
use crate::backend::native_runtime::{
    NativeCompiledPassResourceReport, NativeCompiledResourceMetrics,
    NativeOccupancyLimitingResource, NativeRuntimeAvailability, NativeTheoreticalOccupancyReport,
    NativeTransformInput32, NativeTransformInput64, NativeTransformOutput32,
    NativeTransformOutput64, PreparedProgramStorage, finish_transform_output32,
    finish_transform_output64, load_first_library, prepare_program_complex32,
    prepare_program_complex32_resident_input, prepare_program_complex64,
    prepare_program_double_double, prepare_program_double_double_scalar,
    prepare_program_f64_scalar, prepare_transform_input32, prepare_transform_input64,
    runtime_error, unavailable,
};
use crate::complex::{Complex32, Complex64};
use crate::config::{Backend, DeviceProfile, GpuVendor, SubgroupProfile};
use crate::double_double::{ComplexDoubleDouble, DoubleDouble};
use crate::error::{Result, VkFftError};
use crate::program_ir::{ExternalBufferLayout, ProgramAllocationKind};
use crate::{ScalarType, TransformIr};

const CUDA_SUCCESS: c_int = 0;
const NVRTC_SUCCESS: c_int = 0;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 2;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 3;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 4;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 81;
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: c_int = 82;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: c_int = 106;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;

type CuDevice = c_int;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type NvrtcProgram = *mut c_void;

type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> c_int;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> c_int;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> c_int;
type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> c_int;
type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> c_int;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> c_int;
type CuCtxSynchronize = unsafe extern "C" fn() -> c_int;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, c_uint) -> c_int;
type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> c_int;
type CuStreamSynchronize = unsafe extern "C" fn(CuStream) -> c_int;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> c_int;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> c_int;
type CuModuleGetFunction = unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> c_int;
type CuFuncGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuFunction) -> c_int;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> c_int;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> c_int;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> c_int;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> c_int;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type NvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> c_int;
type NvrtcCompileProgram = unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int;
type NvrtcGetPtxSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetPtx = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;

struct CudaDriverApi {
    _library: Library,
    init: CuInit,
    driver_get_version: CuDriverGetVersion,
    device_get_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_get_attribute: CuDeviceGetAttribute,
    device_get_name: CuDeviceGetName,
    ctx_create: CuCtxCreate,
    ctx_destroy: CuCtxDestroy,
    ctx_set_current: CuCtxSetCurrent,
    ctx_synchronize: CuCtxSynchronize,
    stream_create: CuStreamCreate,
    stream_destroy: CuStreamDestroy,
    stream_synchronize: CuStreamSynchronize,
    module_load_data: CuModuleLoadData,
    module_unload: CuModuleUnload,
    module_get_function: CuModuleGetFunction,
    func_get_attribute: CuFuncGetAttribute,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    launch_kernel: CuLaunchKernel,
}

impl CudaDriverApi {
    fn load() -> Result<Self> {
        let (library, _) = load_first_library(
            Backend::Cuda,
            &["libcuda.so.1", "libcuda.so", "/usr/lib/libcuda.so.1"],
        )?;
        Ok(Self {
            init: load_symbol(&library, b"cuInit\0")?,
            driver_get_version: load_symbol(&library, b"cuDriverGetVersion\0")?,
            device_get_count: load_symbol(&library, b"cuDeviceGetCount\0")?,
            device_get: load_symbol(&library, b"cuDeviceGet\0")?,
            device_get_attribute: load_symbol(&library, b"cuDeviceGetAttribute\0")?,
            device_get_name: load_symbol(&library, b"cuDeviceGetName\0")?,
            ctx_create: load_symbol(&library, b"cuCtxCreate_v2\0")?,
            ctx_destroy: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
            ctx_set_current: load_symbol(&library, b"cuCtxSetCurrent\0")?,
            ctx_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
            stream_create: load_symbol(&library, b"cuStreamCreate\0")?,
            stream_destroy: load_symbol(&library, b"cuStreamDestroy_v2\0")?,
            stream_synchronize: load_symbol(&library, b"cuStreamSynchronize\0")?,
            module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
            module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
            module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
            func_get_attribute: load_symbol(&library, b"cuFuncGetAttribute\0")?,
            mem_alloc: load_symbol(&library, b"cuMemAlloc_v2\0")?,
            mem_free: load_symbol(&library, b"cuMemFree_v2\0")?,
            memcpy_htod: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
            memcpy_dtoh: load_symbol(&library, b"cuMemcpyDtoH_v2\0")?,
            launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
            _library: library,
        })
    }
}

const NVRTC_LIBRARY_CANDIDATES: &[&str] = &[
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11.2",
    "libnvrtc.so.11.1",
    "libnvrtc.so.11.0",
    "libnvrtc.so.10.2",
    "libnvrtc.so.10.1",
    "libnvrtc.so.10.0",
    "libnvrtc.so.9.2",
    "libnvrtc.so.9.1",
    "libnvrtc.so.9.0",
    "libnvrtc.so.8.0",
    "libnvrtc.so",
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/local/cuda-12.4/lib64/libnvrtc.so",
    "/usr/local/cuda-12.6/lib64/libnvrtc.so",
];

fn nvrtc_compile_failure_message(
    version: (i32, i32),
    library_name: &str,
    compile_result: c_int,
    log: &str,
) -> String {
    let builtins_hint = if log.contains("failed to load builtins") {
        " NVRTC requires libnvrtc-builtins from the same CUDA toolkit; start the process with that toolkit's lib64 directory ahead of other CUDA toolkits in LD_LIBRARY_PATH."
    } else {
        ""
    };
    format!(
        "NVRTC {}.{} ({library_name}) returned {compile_result}: {log}{builtins_hint}",
        version.0, version.1
    )
}

struct NvrtcApi {
    _library: Library,
    library_name: String,
    version: (i32, i32),
    create_program: NvrtcCreateProgram,
    destroy_program: NvrtcDestroyProgram,
    compile_program: NvrtcCompileProgram,
    get_ptx_size: NvrtcGetPtxSize,
    get_ptx: NvrtcGetPtx,
    get_log_size: NvrtcGetProgramLogSize,
    get_log: NvrtcGetProgramLog,
}

fn nvrtc_version_is_driver_compatible(
    nvrtc_version: (i32, i32),
    driver_cuda_version: (i32, i32),
) -> bool {
    nvrtc_version <= driver_cuda_version
}

impl NvrtcApi {
    fn load_all() -> Result<Vec<Self>> {
        let mut compilers = Vec::<Self>::new();
        let mut errors = Vec::<String>::new();
        for candidate in NVRTC_LIBRARY_CANDIDATES {
            // SAFETY: every successfully loaded library is retained by the returned API.
            let library = match unsafe { Library::new(candidate) } {
                Ok(library) => library,
                Err(error) => {
                    errors.push(format!("{candidate}: {error}"));
                    continue;
                }
            };
            match Self::from_library(library, candidate) {
                Ok(api) => {
                    if !compilers
                        .iter()
                        .any(|existing| existing.version == api.version)
                    {
                        compilers.push(api);
                    }
                }
                Err(error) => errors.push(format!("{candidate}: {error}")),
            }
        }
        if compilers.is_empty() {
            return Err(unavailable(
                Backend::Cuda,
                format!(
                    "no usable NVRTC compiler library could be loaded ({})",
                    errors.join("; ")
                ),
            ));
        }
        compilers.sort_by(|lhs, rhs| rhs.version.cmp(&lhs.version));
        Ok(compilers)
    }

    fn from_library(library: Library, library_name: &str) -> Result<Self> {
        let version_fn = load_symbol::<NvrtcVersion>(&library, b"nvrtcVersion\0")?;
        let mut major = 0;
        let mut minor = 0;
        check_nvrtc(
            unsafe { version_fn(&mut major, &mut minor) },
            "nvrtcVersion",
        )?;
        if major <= 0 || minor < 0 {
            return Err(runtime_error(
                Backend::Cuda,
                format!("NVRTC reported invalid version {major}.{minor}"),
            ));
        }
        Ok(Self {
            create_program: load_symbol(&library, b"nvrtcCreateProgram\0")?,
            destroy_program: load_symbol(&library, b"nvrtcDestroyProgram\0")?,
            compile_program: load_symbol(&library, b"nvrtcCompileProgram\0")?,
            get_ptx_size: load_symbol(&library, b"nvrtcGetPTXSize\0")?,
            get_ptx: load_symbol(&library, b"nvrtcGetPTX\0")?,
            get_log_size: load_symbol(&library, b"nvrtcGetProgramLogSize\0")?,
            get_log: load_symbol(&library, b"nvrtcGetProgramLog\0")?,
            library_name: library_name.to_owned(),
            version: (major, minor),
            _library: library,
        })
    }

    fn is_driver_compatible(&self, driver_cuda_version: (i32, i32)) -> bool {
        nvrtc_version_is_driver_compatible(self.version, driver_cuda_version)
    }

    fn compile_ptx(&self, source: &str, compute_capability: (i32, i32)) -> Result<Vec<u8>> {
        let source = CString::new(source).map_err(|_| {
            runtime_error(Backend::Cuda, "CUDA source contains an interior NUL byte")
        })?;
        let name = CString::new("vkfft_rs.cu").expect("static CString");
        let mut program = ptr::null_mut();
        check_nvrtc(
            unsafe {
                (self.create_program)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "nvrtcCreateProgram",
        )?;

        let architecture = CString::new(format!(
            "--gpu-architecture=compute_{}{}",
            compute_capability.0, compute_capability.1
        ))
        .expect("NVRTC option has no NUL");
        let standard = CString::new("--std=c++11").expect("static CString");
        let disable_fmad = CString::new("--fmad=false").expect("static CString");
        let mut options = vec![architecture.as_ptr(), standard.as_ptr()];
        if source
            .to_bytes()
            .windows(b"vkfft_dd_two_sum".len())
            .any(|window| window == b"vkfft_dd_two_sum")
        {
            options.push(disable_fmad.as_ptr());
        }
        let compile_result =
            unsafe { (self.compile_program)(program, options.len() as c_int, options.as_ptr()) };
        if compile_result != NVRTC_SUCCESS {
            let log = self.program_log(program);
            unsafe {
                (self.destroy_program)(&mut program);
            }
            return Err(VkFftError::ShaderCompilation(
                nvrtc_compile_failure_message(
                    self.version,
                    &self.library_name,
                    compile_result,
                    &log,
                ),
            ));
        }

        let mut ptx_size = 0usize;
        check_nvrtc(
            unsafe { (self.get_ptx_size)(program, &mut ptx_size) },
            "nvrtcGetPTXSize",
        )?;
        let mut ptx = vec![0u8; ptx_size];
        check_nvrtc(
            unsafe { (self.get_ptx)(program, ptx.as_mut_ptr().cast()) },
            "nvrtcGetPTX",
        )?;
        check_nvrtc(
            unsafe { (self.destroy_program)(&mut program) },
            "nvrtcDestroyProgram",
        )?;
        Ok(ptx)
    }

    fn program_log(&self, program: NvrtcProgram) -> String {
        let mut size = 0usize;
        if unsafe { (self.get_log_size)(program, &mut size) } != NVRTC_SUCCESS || size == 0 {
            return "no NVRTC log available".to_owned();
        }
        let mut bytes = vec![0u8; size];
        if unsafe { (self.get_log)(program, bytes.as_mut_ptr().cast()) } != NVRTC_SUCCESS {
            return "failed to read NVRTC log".to_owned();
        }
        CStr::from_bytes_until_nul(&bytes)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // SAFETY: all symbol signatures mirror the CUDA 12.x public headers. The owning
    // API structs retain `library`, so copied function pointers stay valid.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            unavailable(
                Backend::Cuda,
                format!(
                    "missing symbol {}: {error}",
                    String::from_utf8_lossy(name).trim_end_matches('\0')
                ),
            )
        })
}

fn check_cuda(code: c_int, operation: &'static str) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(runtime_error(
            Backend::Cuda,
            format!("{operation} returned CUresult {code}"),
        ))
    }
}

fn check_nvrtc(code: c_int, operation: &'static str) -> Result<()> {
    if code == NVRTC_SUCCESS {
        Ok(())
    } else {
        Err(runtime_error(
            Backend::Cuda,
            format!("{operation} returned nvrtcResult {code}"),
        ))
    }
}

fn decode_cuda_driver_version(encoded: c_int) -> Result<(i32, i32)> {
    if encoded <= 0 {
        return Err(runtime_error(
            Backend::Cuda,
            format!("cuDriverGetVersion returned invalid version {encoded}"),
        ));
    }
    Ok((encoded / 1000, (encoded % 1000) / 10))
}

struct CudaDeviceMemory<'a> {
    api: &'a CudaDriverApi,
    ptr: CuDevicePtr,
    bytes: usize,
    owned: bool,
}

impl CudaDeviceMemory<'_> {
    fn relinquish(&mut self) -> CuDevicePtr {
        self.owned = false;
        self.ptr
    }
}

impl Drop for CudaDeviceMemory<'_> {
    fn drop(&mut self) {
        if self.owned && self.ptr != 0 {
            unsafe {
                (self.api.mem_free)(self.ptr);
            }
        }
    }
}

struct CudaResidentProgramOutput<'a> {
    memory: CudaDeviceMemory<'a>,
    scalar: ScalarType,
    layout: ExternalBufferLayout,
}

struct CudaLoadedModule<'a> {
    api: &'a CudaDriverApi,
    module: CuModule,
    function: CuFunction,
    owned: bool,
}

impl Drop for CudaLoadedModule<'_> {
    fn drop(&mut self) {
        if self.owned && !self.module.is_null() {
            unsafe {
                (self.api.module_unload)(self.module);
            }
        }
    }
}

struct CudaOwnedStream<'a> {
    api: &'a CudaDriverApi,
    stream: CuStream,
    owned: bool,
}

impl CudaOwnedStream<'_> {
    fn synchronize(&self) -> Result<()> {
        check_cuda(
            unsafe { (self.api.stream_synchronize)(self.stream) },
            "cuStreamSynchronize",
        )
    }

    fn relinquish(&mut self) -> CuStream {
        self.owned = false;
        self.stream
    }
}

impl Drop for CudaOwnedStream<'_> {
    fn drop(&mut self) {
        if self.owned && !self.stream.is_null() {
            unsafe {
                (self.api.stream_synchronize)(self.stream);
                (self.api.stream_destroy)(self.stream);
            }
            self.stream = ptr::null_mut();
        }
    }
}

struct CudaPendingProgram<'a> {
    stream: CudaOwnedStream<'a>,
    context: &'a CudaExecutionContext,
    prepared: PreparedProgramStorage,
    allocations: Vec<CudaDeviceMemory<'a>>,
    pending_luts: Vec<(usize, Vec<u8>)>,
    completed: bool,
}

impl<'a> CudaPendingProgram<'a> {
    fn synchronize(&self) -> Result<()> {
        check_cuda(
            unsafe { (self.context.driver.ctx_set_current)(self.context.context) },
            "cuCtxSetCurrent",
        )?;
        self.stream.synchronize()
    }

    fn finalize_resources(
        &mut self,
        preserve_allocation: Option<usize>,
    ) -> Result<Option<CudaDeviceMemory<'a>>> {
        if let Some(index) = preserve_allocation {
            let allocation = self.prepared.memory_plan.allocations.get(index).ok_or(
                VkFftError::InvalidKernelIr("CUDA resident output references a missing allocation"),
            )?;
            if allocation.kind == ProgramAllocationKind::LookupTable {
                return Err(VkFftError::InvalidKernelIr(
                    "CUDA resident output cannot preserve a lookup-table allocation",
                ));
            }
        }

        for (index, key) in self.pending_luts.drain(..) {
            let allocation = self
                .allocations
                .get_mut(index)
                .ok_or(VkFftError::InvalidKernelIr(
                    "pending CUDA LUT references a missing allocation",
                ))?;
            let ptr = allocation.relinquish();
            let mut cache = self
                .context
                .lut_cache
                .lock()
                .map_err(|_| runtime_error(Backend::Cuda, "CUDA LUT cache lock is poisoned"))?;
            if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
                entry.insert(ptr);
            } else {
                check_cuda(
                    unsafe { (self.context.driver.mem_free)(ptr) },
                    "cuMemFree_v2",
                )?;
            }
        }

        let mut preserved = None;
        {
            let mut pool = self.context.transient_buffer_pool.lock().map_err(|_| {
                runtime_error(Backend::Cuda, "CUDA transient buffer pool lock is poisoned")
            })?;
            for (index, (allocation, mut device_memory)) in self
                .prepared
                .memory_plan
                .allocations
                .iter()
                .zip(self.allocations.drain(..))
                .enumerate()
            {
                if Some(index) == preserve_allocation {
                    preserved = Some(device_memory);
                } else if allocation.kind != ProgramAllocationKind::LookupTable {
                    let bytes = device_memory.bytes;
                    let ptr = device_memory.relinquish();
                    pool.entry(bytes).or_default().push(ptr);
                }
            }
        }
        {
            let stream = self.stream.relinquish();
            self.context
                .stream_pool
                .lock()
                .map_err(|_| runtime_error(Backend::Cuda, "CUDA stream pool lock is poisoned"))?
                .push(stream);
        }
        self.completed = true;
        Ok(preserved)
    }

    fn finish(&mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        self.synchronize()?;
        let output = self
            .allocations
            .get(self.prepared.output_allocation.0)
            .ok_or(VkFftError::InvalidKernelIr(
                "CUDA program is missing its output allocation",
            ))?;
        let output_index = self.prepared.output_allocation.0;
        let output_allocation =
            self.prepared
                .allocations
                .get_mut(output_index)
                .ok_or(VkFftError::InvalidKernelIr(
                    "CUDA host program is missing its output allocation",
                ))?;
        if output_allocation.host_bytes.is_none() {
            output_allocation.host_bytes = Some(
                self.context
                    .take_host_output_buffer(output_allocation.byte_len)?,
            );
        }
        let output_bytes = output_allocation
            .host_bytes
            .as_deref_mut()
            .expect("CUDA output host buffer initialized above");
        check_cuda(
            unsafe {
                (self.context.driver.memcpy_dtoh)(
                    output_bytes.as_mut_ptr().cast(),
                    output.ptr,
                    output_bytes.len(),
                )
            },
            "cuMemcpyDtoH_v2",
        )?;
        let preserved = self.finalize_resources(None)?;
        debug_assert!(preserved.is_none());
        Ok(())
    }

    fn finish_into_complex32(&mut self, output_values: &mut [Complex32]) -> Result<()> {
        if self.completed {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA direct output was already completed",
            ));
        }
        self.synchronize()?;
        let output = self
            .allocations
            .get(self.prepared.output_allocation.0)
            .ok_or(VkFftError::InvalidKernelIr(
                "CUDA program is missing its output allocation",
            ))?;
        let output_allocation = self
            .prepared
            .allocations
            .get(self.prepared.output_allocation.0)
            .ok_or(VkFftError::InvalidKernelIr(
                "CUDA direct output is missing prepared allocation metadata",
            ))?;
        let output_bytes = std::mem::size_of_val(output_values);
        if output_bytes != output_allocation.byte_len {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA direct output byte size does not match the program allocation",
            ));
        }
        check_cuda(
            unsafe {
                (self.context.driver.memcpy_dtoh)(
                    output_values.as_mut_ptr().cast(),
                    output.ptr,
                    output_bytes,
                )
            },
            "cuMemcpyDtoH_v2",
        )?;
        let preserved = self.finalize_resources(None)?;
        debug_assert!(preserved.is_none());
        Ok(())
    }

    fn into_resident_output(mut self) -> Result<CudaResidentProgramOutput<'a>> {
        if self.completed {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA program output was already completed",
            ));
        }
        self.synchronize()?;
        let output_index = self.prepared.output_allocation.0;
        let allocation = self
            .prepared
            .memory_plan
            .allocations
            .get(output_index)
            .ok_or(VkFftError::InvalidKernelIr(
                "CUDA resident output references a missing allocation",
            ))?;
        let scalar = allocation.scalar;
        let layout = self.prepared.output_layout;
        let memory =
            self.finalize_resources(Some(output_index))?
                .ok_or(VkFftError::InvalidKernelIr(
                    "CUDA resident output allocation was not preserved",
                ))?;
        Ok(CudaResidentProgramOutput {
            memory,
            scalar,
            layout,
        })
    }
}

impl Drop for CudaPendingProgram<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
        let output_index = self.prepared.output_allocation.0;
        if let Some(bytes) = self
            .prepared
            .allocations
            .get_mut(output_index)
            .and_then(|allocation| allocation.host_bytes.take())
        {
            self.context.recycle_host_output_buffer(bytes);
        }
    }
}

pub struct CudaProgramTicket32<'a> {
    pending: CudaPendingProgram<'a>,
}

impl CudaProgramTicket32<'_> {
    pub fn wait(mut self) -> Result<Vec<Complex32>> {
        self.pending.finish()?;
        self.pending.prepared.output_complex32()
    }
}

pub struct CudaProgramTicket64<'a> {
    pending: CudaPendingProgram<'a>,
}

impl CudaProgramTicket64<'_> {
    pub fn wait(mut self) -> Result<Vec<Complex64>> {
        self.pending.finish()?;
        self.pending.prepared.output_complex64()
    }
}

pub struct CudaTransformTicket32<'a> {
    ir: TransformIr,
    ticket: CudaProgramTicket32<'a>,
}

impl CudaTransformTicket32<'_> {
    pub fn wait(self) -> Result<NativeTransformOutput32> {
        let Self { ir, ticket } = self;
        finish_transform_output32(&ir, ticket.wait()?)
    }
}

pub struct CudaTransformTicket64<'a> {
    ir: TransformIr,
    ticket: CudaProgramTicket64<'a>,
}

impl CudaTransformTicket64<'_> {
    pub fn wait(self) -> Result<NativeTransformOutput64> {
        let Self { ir, ticket } = self;
        finish_transform_output64(&ir, ticket.wait()?)
    }
}

pub const CUDA_PTX_CACHE_ARCHIVE_VERSION: u32 = 2;
const CUDA_PTX_CACHE_ARCHIVE_MAGIC: &[u8; 8] = b"VKFTRSPT";
const CUDA_PTX_CACHE_ARCHIVE_COMMIT_BYTES: usize = 40;
const CUDA_PTX_CACHE_ARCHIVE_HEADER_BYTES: usize =
    8 + 4 + 4 + 4 + 4 + 4 + CUDA_PTX_CACHE_ARCHIVE_COMMIT_BYTES + 8;

/// Versioned persistent form of the NVRTC source-to-PTX cache. CUDA modules/functions
/// remain context-local and are reloaded from PTX after restore; the archive only skips
/// recompilation and is deliberately tied to the exact compute capability, CUDA driver
/// compatibility level, and upstream VkFFT porting baseline that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaPtxCacheArchive {
    compute_capability: (i32, i32),
    driver_cuda_version: (i32, i32),
    entries: Vec<(String, Vec<u8>)>,
}

impl CudaPtxCacheArchive {
    pub const fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    pub const fn driver_cuda_version(&self) -> (i32, i32) {
        self.driver_cuda_version
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn encode(&self) -> Vec<u8> {
        let commit = crate::UPSTREAM_VKFFT_COMMIT.as_bytes();
        debug_assert_eq!(commit.len(), CUDA_PTX_CACHE_ARCHIVE_COMMIT_BYTES);
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let body_bytes = entries.iter().fold(0usize, |total, (source, ptx)| {
            total.saturating_add(16 + source.len() + ptx.len())
        });
        let mut bytes = Vec::with_capacity(CUDA_PTX_CACHE_ARCHIVE_HEADER_BYTES + body_bytes);
        bytes.extend_from_slice(CUDA_PTX_CACHE_ARCHIVE_MAGIC);
        bytes.extend_from_slice(&CUDA_PTX_CACHE_ARCHIVE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.compute_capability.0.to_le_bytes());
        bytes.extend_from_slice(&self.compute_capability.1.to_le_bytes());
        bytes.extend_from_slice(&self.driver_cuda_version.0.to_le_bytes());
        bytes.extend_from_slice(&self.driver_cuda_version.1.to_le_bytes());
        bytes.extend_from_slice(commit);
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (source, ptx) in entries {
            bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(ptx.len() as u64).to_le_bytes());
            bytes.extend_from_slice(source.as_bytes());
            bytes.extend_from_slice(ptx);
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let malformed = |message: &str| runtime_error(Backend::Cuda, message);
        if bytes.len() < CUDA_PTX_CACHE_ARCHIVE_HEADER_BYTES {
            return Err(malformed("CUDA PTX cache archive is truncated"));
        }
        if &bytes[..8] != CUDA_PTX_CACHE_ARCHIVE_MAGIC {
            return Err(malformed("CUDA PTX cache archive magic does not match"));
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| malformed("CUDA PTX cache archive version is malformed"))?,
        );
        if version != CUDA_PTX_CACHE_ARCHIVE_VERSION {
            return Err(malformed(
                "CUDA PTX cache archive schema version is unsupported",
            ));
        }
        let major = i32::from_le_bytes(
            bytes[12..16]
                .try_into()
                .map_err(|_| malformed("CUDA PTX cache major capability is malformed"))?,
        );
        let minor = i32::from_le_bytes(
            bytes[16..20]
                .try_into()
                .map_err(|_| malformed("CUDA PTX cache minor capability is malformed"))?,
        );
        let driver_major = i32::from_le_bytes(
            bytes[20..24]
                .try_into()
                .map_err(|_| malformed("CUDA PTX cache driver major version is malformed"))?,
        );
        let driver_minor = i32::from_le_bytes(
            bytes[24..28]
                .try_into()
                .map_err(|_| malformed("CUDA PTX cache driver minor version is malformed"))?,
        );
        if driver_major <= 0 || driver_minor < 0 {
            return Err(malformed("CUDA PTX cache driver version is invalid"));
        }
        let commit_start = 28;
        let commit_end = commit_start + CUDA_PTX_CACHE_ARCHIVE_COMMIT_BYTES;
        if &bytes[commit_start..commit_end] != crate::UPSTREAM_VKFFT_COMMIT.as_bytes() {
            return Err(malformed(
                "CUDA PTX cache archive upstream commit does not match",
            ));
        }
        let count_end = commit_end + 8;
        let count = u64::from_le_bytes(
            bytes[commit_end..count_end]
                .try_into()
                .map_err(|_| malformed("CUDA PTX cache archive entry count is malformed"))?,
        );
        let count = usize::try_from(count)
            .map_err(|_| malformed("CUDA PTX cache entry count does not fit this platform"))?;
        let mut offset = count_end;
        let mut decoded = HashMap::<String, Vec<u8>>::with_capacity(count);
        for _ in 0..count {
            let lengths_end = offset
                .checked_add(16)
                .ok_or_else(|| malformed("CUDA PTX cache entry header overflows"))?;
            if lengths_end > bytes.len() {
                return Err(malformed("CUDA PTX cache entry header is truncated"));
            }
            let source_len = u64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .map_err(|_| malformed("CUDA PTX cache source length is malformed"))?,
            );
            let ptx_len = u64::from_le_bytes(
                bytes[offset + 8..lengths_end]
                    .try_into()
                    .map_err(|_| malformed("CUDA PTX cache PTX length is malformed"))?,
            );
            offset = lengths_end;
            let source_len = usize::try_from(source_len).map_err(|_| {
                malformed("CUDA PTX cache source length does not fit this platform")
            })?;
            let ptx_len = usize::try_from(ptx_len)
                .map_err(|_| malformed("CUDA PTX cache PTX length does not fit this platform"))?;
            let source_end = offset
                .checked_add(source_len)
                .ok_or_else(|| malformed("CUDA PTX cache source length overflows"))?;
            let ptx_end = source_end
                .checked_add(ptx_len)
                .ok_or_else(|| malformed("CUDA PTX cache PTX length overflows"))?;
            if ptx_end > bytes.len() {
                return Err(malformed("CUDA PTX cache entry payload is truncated"));
            }
            let source = String::from_utf8(bytes[offset..source_end].to_vec())
                .map_err(|_| malformed("CUDA PTX cache source is not valid UTF-8"))?;
            let ptx = bytes[source_end..ptx_end].to_vec();
            if decoded.insert(source, ptx).is_some() {
                return Err(malformed(
                    "CUDA PTX cache archive contains duplicate source entries",
                ));
            }
            offset = ptx_end;
        }
        if offset != bytes.len() {
            return Err(malformed("CUDA PTX cache archive has trailing bytes"));
        }
        let mut entries = decoded.into_iter().collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        Ok(Self {
            compute_capability: (major, minor),
            driver_cuda_version: (driver_major, driver_minor),
            entries,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaResidencyLimits {
    multiprocessor_count: usize,
    warp_size: usize,
    max_threads_per_multiprocessor: usize,
    max_shared_memory_bytes_per_multiprocessor: usize,
    max_registers_per_multiprocessor: usize,
    max_blocks_per_multiprocessor: Option<usize>,
}

pub struct CudaExecutionContext {
    driver: CudaDriverApi,
    nvrtc: Vec<NvrtcApi>,
    driver_cuda_version: (i32, i32),
    context: CuContext,
    device: CuDevice,
    device_name: String,
    compute_capability: (i32, i32),
    profile: DeviceProfile,
    residency_limits: CudaResidencyLimits,
    ptx_cache: Mutex<HashMap<String, Vec<u8>>>,
    module_cache: Mutex<HashMap<String, (CuModule, CuFunction)>>,
    lut_cache: Mutex<HashMap<Vec<u8>, CuDevicePtr>>,
    transient_buffer_pool: Mutex<HashMap<usize, Vec<CuDevicePtr>>>,
    host_output_pool: Mutex<HashMap<usize, Vec<u8>>>,
    stream_pool: Mutex<Vec<CuStream>>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CudaExecutionContext {
    pub fn probe() -> NativeRuntimeAvailability {
        let compilers = NvrtcApi::load_all();
        let driver = match CudaDriverApi::load() {
            Ok(driver) => driver,
            Err(error) => {
                return NativeRuntimeAvailability {
                    backend: Backend::Cuda,
                    loader_available: false,
                    compiler_available: compilers.is_ok(),
                    device_count: 0,
                    detail: error.to_string(),
                };
            }
        };
        let init = unsafe { (driver.init)(0) };
        if init != CUDA_SUCCESS {
            return NativeRuntimeAvailability {
                backend: Backend::Cuda,
                loader_available: true,
                compiler_available: compilers.is_ok(),
                device_count: 0,
                detail: format!("cuInit returned {init}"),
            };
        }
        let mut encoded_driver_version = 0;
        let version_result = unsafe { (driver.driver_get_version)(&mut encoded_driver_version) };
        if version_result != CUDA_SUCCESS {
            return NativeRuntimeAvailability {
                backend: Backend::Cuda,
                loader_available: true,
                compiler_available: false,
                device_count: 0,
                detail: format!("cuDriverGetVersion returned {version_result}"),
            };
        }
        let driver_cuda_version = match decode_cuda_driver_version(encoded_driver_version) {
            Ok(version) => version,
            Err(error) => {
                return NativeRuntimeAvailability {
                    backend: Backend::Cuda,
                    loader_available: true,
                    compiler_available: false,
                    device_count: 0,
                    detail: error.to_string(),
                };
            }
        };
        let compiler_available = compilers
            .as_ref()
            .map(|compilers| {
                compilers
                    .iter()
                    .any(|compiler| compiler.is_driver_compatible(driver_cuda_version))
            })
            .unwrap_or(false);
        let mut count = 0;
        let result = unsafe { (driver.device_get_count)(&mut count) };
        NativeRuntimeAvailability {
            backend: Backend::Cuda,
            loader_available: true,
            compiler_available,
            device_count: if result == CUDA_SUCCESS {
                count.max(0) as usize
            } else {
                0
            },
            detail: if result == CUDA_SUCCESS {
                format!(
                    "CUDA Driver API {}.{} reports {} device(s); compatible NVRTC available={compiler_available}",
                    driver_cuda_version.0,
                    driver_cuda_version.1,
                    count.max(0)
                )
            } else {
                format!("cuDeviceGetCount returned {result}")
            },
        }
    }

    pub fn new(device_index: usize) -> Result<Self> {
        Self::new_internal(device_index)
    }

    /// Create a CUDA context and preload a versioned source-to-PTX archive. The PTX
    /// payload is not loaded as a module until first use, and an incompatible compute
    /// capability is rejected before any archived PTX reaches the driver JIT.
    pub fn new_with_ptx_cache_archive(device_index: usize, encoded: &[u8]) -> Result<Self> {
        let context = Self::new_internal(device_index)?;
        context.restore_ptx_cache_archive(encoded)?;
        Ok(context)
    }

    fn new_internal(device_index: usize) -> Result<Self> {
        let driver = CudaDriverApi::load()?;
        check_cuda(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut encoded_driver_version = 0;
        check_cuda(
            unsafe { (driver.driver_get_version)(&mut encoded_driver_version) },
            "cuDriverGetVersion",
        )?;
        let driver_cuda_version = decode_cuda_driver_version(encoded_driver_version)?;
        let nvrtc = NvrtcApi::load_all()?;
        if !nvrtc
            .iter()
            .any(|compiler| compiler.is_driver_compatible(driver_cuda_version))
        {
            let loaded = nvrtc
                .iter()
                .map(|compiler| format!("{}.{}", compiler.version.0, compiler.version.1))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(unavailable(
                Backend::Cuda,
                format!(
                    "CUDA driver {}.{} has no compatible installed NVRTC compiler; loaded versions: {loaded}",
                    driver_cuda_version.0, driver_cuda_version.1
                ),
            ));
        }
        let mut count = 0;
        check_cuda(
            unsafe { (driver.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if device_index >= count.max(0) as usize {
            return Err(unavailable(
                Backend::Cuda,
                format!(
                    "requested device {device_index}, but only {} are available",
                    count.max(0)
                ),
            ));
        }
        let mut device = 0;
        check_cuda(
            unsafe { (driver.device_get)(&mut device, device_index as c_int) },
            "cuDeviceGet",
        )?;
        let max_threads =
            device_attribute(&driver, device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
        let max_block_dim_x =
            device_attribute(&driver, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?;
        let max_block_dim_y =
            device_attribute(&driver, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?;
        let max_block_dim_z =
            device_attribute(&driver, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?;
        let shared = device_attribute(
            &driver,
            device,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )?;
        let major = device_attribute(
            &driver,
            device,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        let minor = device_attribute(
            &driver,
            device,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )?;
        let residency_limits = CudaResidencyLimits {
            multiprocessor_count: device_attribute_usize(
                &driver,
                device,
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                "CUDA multiprocessor count",
            )?,
            warp_size: device_attribute_usize(
                &driver,
                device,
                CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                "CUDA warp size",
            )?,
            max_threads_per_multiprocessor: device_attribute_usize(
                &driver,
                device,
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                "CUDA max threads per multiprocessor",
            )?,
            max_shared_memory_bytes_per_multiprocessor: device_attribute_usize(
                &driver,
                device,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                "CUDA max shared memory per multiprocessor",
            )?,
            max_registers_per_multiprocessor: device_attribute_usize(
                &driver,
                device,
                CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                "CUDA max registers per multiprocessor",
            )?,
            max_blocks_per_multiprocessor: optional_device_attribute_usize(
                &driver,
                device,
                CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
            ),
        };
        let mut name = [0i8; 256];
        check_cuda(
            unsafe { (driver.device_get_name)(name.as_mut_ptr(), name.len() as c_int, device) },
            "cuDeviceGetName",
        )?;
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut context = ptr::null_mut();
        check_cuda(
            unsafe { (driver.ctx_create)(&mut context, 0, device) },
            "cuCtxCreate_v2",
        )?;
        let shared = usize::try_from(shared).map_err(|_| VkFftError::ValueOutOfRange {
            field: "CUDA shared memory bytes",
        })?;
        let max_threads =
            usize::try_from(max_threads).map_err(|_| VkFftError::ValueOutOfRange {
                field: "CUDA max threads per block",
            })?;
        let max_workgroup_size = [max_block_dim_x, max_block_dim_y, max_block_dim_z]
            .map(|value| usize::try_from(value).unwrap_or(0));
        if max_workgroup_size.contains(&0) {
            return Err(VkFftError::ValueOutOfRange {
                field: "CUDA max block dimension",
            });
        }
        let profile = DeviceProfile {
            backend: Backend::Cuda,
            vendor: GpuVendor::Nvidia,
            shared_memory_bytes: shared,
            shared_memory_pow2_bytes: floor_power_of_two(shared),
            max_threads_per_block: max_threads,
            max_workgroup_size,
            coalesced_memory_bytes: 32,
            shared_banks: 32,
            supports_f64: major > 1 || (major == 1 && minor >= 3),
            subgroup: SubgroupProfile {
                size: 32,
                min_size: 32,
                max_size: 32,
                required_size_compute_supported: false,
                compute_supported: true,
                basic_supported: true,
                shuffle_supported: true,
                shuffle_relative_supported: true,
                compute_full_subgroups: true,
            },
        };
        Ok(Self {
            driver,
            nvrtc,
            driver_cuda_version,
            context,
            device,
            device_name,
            compute_capability: (major, minor),
            profile,
            residency_limits,
            ptx_cache: Mutex::new(HashMap::new()),
            module_cache: Mutex::new(HashMap::new()),
            lut_cache: Mutex::new(HashMap::new()),
            transient_buffer_pool: Mutex::new(HashMap::new()),
            host_output_pool: Mutex::new(HashMap::new()),
            stream_pool: Mutex::new(Vec::new()),
            _not_send_sync: PhantomData,
        })
    }

    pub const fn device_profile(&self) -> DeviceProfile {
        self.profile
    }

    pub const fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    pub const fn driver_cuda_version(&self) -> (i32, i32) {
        self.driver_cuda_version
    }

    pub fn nvrtc_versions(&self) -> Vec<(i32, i32)> {
        self.nvrtc.iter().map(|compiler| compiler.version).collect()
    }

    pub fn ptx_cache_archive_data(&self) -> Result<Vec<u8>> {
        let cache = self
            .ptx_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA PTX cache lock is poisoned"))?;
        let mut entries = cache
            .iter()
            .map(|(source, ptx)| (source.clone(), ptx.clone()))
            .collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        Ok(CudaPtxCacheArchive {
            compute_capability: self.compute_capability,
            driver_cuda_version: self.driver_cuda_version,
            entries,
        }
        .encode())
    }

    pub fn restore_ptx_cache_archive(&self, encoded: &[u8]) -> Result<()> {
        let archive = CudaPtxCacheArchive::decode(encoded)?;
        if archive.compute_capability != self.compute_capability {
            return Err(runtime_error(
                Backend::Cuda,
                format!(
                    "CUDA PTX cache archive targets compute_{}{}, but the selected device is compute_{}{}",
                    archive.compute_capability.0,
                    archive.compute_capability.1,
                    self.compute_capability.0,
                    self.compute_capability.1
                ),
            ));
        }
        if archive.driver_cuda_version != self.driver_cuda_version {
            return Err(runtime_error(
                Backend::Cuda,
                format!(
                    "CUDA PTX cache archive targets CUDA driver compatibility {}.{}, but the active driver reports {}.{}",
                    archive.driver_cuda_version.0,
                    archive.driver_cuda_version.1,
                    self.driver_cuda_version.0,
                    self.driver_cuda_version.1
                ),
            ));
        }
        let mut cache = self
            .ptx_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA PTX cache lock is poisoned"))?;
        for (source, ptx) in archive.entries {
            match cache.entry(source) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(ptx);
                }
                std::collections::hash_map::Entry::Occupied(entry) if entry.get() == &ptx => {}
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(runtime_error(
                        Backend::Cuda,
                        "CUDA PTX cache archive conflicts with an existing source entry",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device
    }

    pub fn cached_ptx_count(&self) -> Result<usize> {
        self.ptx_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA PTX cache lock is poisoned"))
    }

    pub fn cached_module_count(&self) -> Result<usize> {
        self.module_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA module cache lock is poisoned"))
    }

    pub fn cached_lut_count(&self) -> Result<usize> {
        self.lut_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA LUT cache lock is poisoned"))
    }

    pub fn cached_transient_buffer_count(&self) -> Result<usize> {
        self.transient_buffer_pool
            .lock()
            .map(|pool| pool.values().map(Vec::len).sum())
            .map_err(|_| {
                runtime_error(Backend::Cuda, "CUDA transient buffer pool lock is poisoned")
            })
    }

    pub fn pooled_stream_count(&self) -> Result<usize> {
        self.stream_pool
            .lock()
            .map(|pool| pool.len())
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA stream pool lock is poisoned"))
    }

    fn compiled_function_resource_metrics(
        &self,
        function: CuFunction,
    ) -> Result<NativeCompiledResourceMetrics> {
        let attribute = |attribute: c_int, field: &'static str| -> Result<usize> {
            let mut value = 0;
            check_cuda(
                unsafe { (self.driver.func_get_attribute)(&mut value, attribute, function) },
                "cuFuncGetAttribute",
            )?;
            usize::try_from(value).map_err(|_| VkFftError::ValueOutOfRange { field })
        };
        Ok(NativeCompiledResourceMetrics::Cuda {
            registers_per_thread: attribute(
                CU_FUNC_ATTRIBUTE_NUM_REGS,
                "CUDA function register count",
            )?,
            static_shared_memory_bytes_per_block: attribute(
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                "CUDA function static shared-memory bytes",
            )?,
            local_memory_bytes_per_thread: attribute(
                CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                "CUDA function local-memory bytes",
            )?,
            max_threads_per_block: attribute(
                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                "CUDA function max threads per block",
            )?,
        })
    }

    fn theoretical_occupancy_reports_from_compiled(
        &self,
        source: &NativeProgramSource,
        compiled_reports: &[NativeCompiledPassResourceReport],
    ) -> Result<Vec<NativeTheoreticalOccupancyReport>> {
        source.validate()?;
        if source.backend != Backend::Cuda
            || source.program.passes.len() != source.shaders.len()
            || compiled_reports.len() != source.shaders.len()
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA theoretical occupancy reporting requires one compiled report per CUDA pass",
            ));
        }
        let limits = self.residency_limits;
        let max_subgroups_per_compute_unit =
            limits.max_threads_per_multiprocessor / limits.warp_size;
        if max_subgroups_per_compute_unit == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA theoretical occupancy reporting has an invalid warp/thread capacity",
            ));
        }

        let mut output = Vec::with_capacity(compiled_reports.len());
        for ((pass, shader), compiled) in source
            .program
            .passes
            .iter()
            .zip(&source.shaders)
            .zip(compiled_reports)
        {
            if pass.name != compiled.pass_name {
                return Err(VkFftError::InvalidKernelIr(
                    "CUDA theoretical occupancy report pass order does not match compiled resources",
                ));
            }
            let NativeCompiledResourceMetrics::Cuda {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread: _,
                max_threads_per_block,
            } = compiled.metrics
            else {
                return Err(VkFftError::InvalidKernelIr(
                    "CUDA theoretical occupancy reporting received non-CUDA compiled resources",
                ));
            };
            let workgroup_threads = (shader.workgroup_size.x as usize)
                .checked_mul(shader.workgroup_size.y as usize)
                .and_then(|value| value.checked_mul(shader.workgroup_size.z as usize))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "CUDA occupancy workgroup thread count",
                })?;
            if workgroup_threads == 0 || workgroup_threads > max_threads_per_block {
                return Err(VkFftError::InvalidKernelIr(
                    "CUDA occupancy workgroup exceeds the compiled function thread limit",
                ));
            }
            let subgroups_per_workgroup = workgroup_threads.div_ceil(limits.warp_size);
            let blocks_by_threads = (limits.max_threads_per_multiprocessor / workgroup_threads)
                .min(max_subgroups_per_compute_unit / subgroups_per_workgroup);

            let padded_register_threads = subgroups_per_workgroup
                .checked_mul(limits.warp_size)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "CUDA occupancy padded register threads",
                })?;
            let blocks_by_registers = if registers_per_thread == 0 {
                None
            } else {
                let registers_per_block = registers_per_thread
                    .checked_mul(padded_register_threads)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "CUDA occupancy registers per block",
                    })?;
                Some(limits.max_registers_per_multiprocessor / registers_per_block)
            };
            let shared_memory_bytes_per_block =
                static_shared_memory_bytes_per_block.max(shader.required_shared_memory_bytes);
            let blocks_by_shared_memory = if shared_memory_bytes_per_block == 0 {
                None
            } else {
                Some(
                    limits.max_shared_memory_bytes_per_multiprocessor
                        / shared_memory_bytes_per_block,
                )
            };

            let mut resident_blocks = blocks_by_threads;
            if let Some(value) = blocks_by_registers {
                resident_blocks = resident_blocks.min(value);
            }
            if let Some(value) = blocks_by_shared_memory {
                resident_blocks = resident_blocks.min(value);
            }
            if let Some(value) = limits.max_blocks_per_multiprocessor {
                resident_blocks = resident_blocks.min(value);
            }
            let resident_subgroups = resident_blocks
                .checked_mul(subgroups_per_workgroup)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "CUDA occupancy resident subgroup count",
                })?
                .min(max_subgroups_per_compute_unit);
            let occupancy_basis_points =
                resident_subgroups
                    .checked_mul(10_000)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "CUDA occupancy basis points",
                    })?
                    / max_subgroups_per_compute_unit;

            let mut limiting_resources = Vec::new();
            if blocks_by_threads == resident_blocks {
                limiting_resources.push(NativeOccupancyLimitingResource::ThreadSubgroups);
            }
            if blocks_by_registers == Some(resident_blocks) {
                limiting_resources.push(NativeOccupancyLimitingResource::Registers);
            }
            if blocks_by_shared_memory == Some(resident_blocks) {
                limiting_resources.push(NativeOccupancyLimitingResource::SharedMemory);
            }
            if limits.max_blocks_per_multiprocessor == Some(resident_blocks) {
                limiting_resources.push(NativeOccupancyLimitingResource::ArchitecturalBlocks);
            }

            output.push(NativeTheoreticalOccupancyReport {
                pass_name: pass.name.clone(),
                backend: Backend::Cuda,
                compute_unit_count: limits.multiprocessor_count,
                subgroup_size: limits.warp_size,
                workgroup_threads,
                subgroups_per_workgroup,
                max_subgroups_per_compute_unit,
                blocks_limited_by_thread_subgroups: blocks_by_threads,
                blocks_limited_by_registers: blocks_by_registers,
                blocks_limited_by_shared_memory: blocks_by_shared_memory,
                architectural_blocks_per_compute_unit: limits.max_blocks_per_multiprocessor,
                resident_blocks_per_compute_unit_upper_bound: resident_blocks,
                resident_subgroups_per_compute_unit_upper_bound: resident_subgroups,
                occupancy_basis_points_upper_bound: occupancy_basis_points,
                limiting_resources,
            });
        }
        Ok(output)
    }

    pub fn clear_runtime_caches(&self) -> Result<()> {
        check_cuda(
            unsafe { (self.driver.ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )?;
        check_cuda(
            unsafe { (self.driver.ctx_synchronize)() },
            "cuCtxSynchronize",
        )?;
        {
            let mut modules = self
                .module_cache
                .lock()
                .map_err(|_| runtime_error(Backend::Cuda, "CUDA module cache lock is poisoned"))?;
            for (_, (module, _)) in modules.drain() {
                check_cuda(
                    unsafe { (self.driver.module_unload)(module) },
                    "cuModuleUnload",
                )?;
            }
        }
        {
            let mut luts = self
                .lut_cache
                .lock()
                .map_err(|_| runtime_error(Backend::Cuda, "CUDA LUT cache lock is poisoned"))?;
            for (_, ptr) in luts.drain() {
                check_cuda(unsafe { (self.driver.mem_free)(ptr) }, "cuMemFree_v2")?;
            }
        }
        {
            let mut pool = self.transient_buffer_pool.lock().map_err(|_| {
                runtime_error(Backend::Cuda, "CUDA transient buffer pool lock is poisoned")
            })?;
            for ptr in pool.drain().flat_map(|(_, buffers)| buffers) {
                check_cuda(unsafe { (self.driver.mem_free)(ptr) }, "cuMemFree_v2")?;
            }
        }
        {
            self.host_output_pool
                .lock()
                .map_err(|_| {
                    runtime_error(Backend::Cuda, "CUDA host output pool lock is poisoned")
                })?
                .clear();
        }
        {
            let mut streams = self
                .stream_pool
                .lock()
                .map_err(|_| runtime_error(Backend::Cuda, "CUDA stream pool lock is poisoned"))?;
            for stream in streams.drain(..) {
                check_cuda(
                    unsafe { (self.driver.stream_destroy)(stream) },
                    "cuStreamDestroy_v2",
                )?;
            }
        }
        self.ptx_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA PTX cache lock is poisoned"))?
            .clear();
        Ok(())
    }

    pub fn compile_ptx(&self, shader: &NativeShaderSource) -> Result<Vec<u8>> {
        if shader.backend != Backend::Cuda {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA runtime can compile only CUDA native shaders",
            ));
        }
        shader.validate()?;
        if let Some(ptx) = self
            .ptx_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA PTX cache lock is poisoned"))?
            .get(&shader.source)
            .cloned()
        {
            return Ok(ptx);
        }
        let mut failures = Vec::<String>::new();
        for compiler in self
            .nvrtc
            .iter()
            .filter(|compiler| compiler.is_driver_compatible(self.driver_cuda_version))
        {
            match compiler.compile_ptx(&shader.source, self.compute_capability) {
                Ok(ptx) => {
                    self.ptx_cache
                        .lock()
                        .map_err(|_| {
                            runtime_error(Backend::Cuda, "CUDA PTX cache lock is poisoned")
                        })?
                        .entry(shader.source.clone())
                        .or_insert_with(|| ptx.clone());
                    return Ok(ptx);
                }
                Err(error) => failures.push(format!(
                    "NVRTC {}.{} ({}): {error}",
                    compiler.version.0, compiler.version.1, compiler.library_name
                )),
            }
        }
        Err(VkFftError::ShaderCompilation(format!(
            "all driver-compatible NVRTC compilers failed for compute_{}{} on CUDA driver {}.{}: {}",
            self.compute_capability.0,
            self.compute_capability.1,
            self.driver_cuda_version.0,
            self.driver_cuda_version.1,
            failures.join(" | ")
        )))
    }

    fn validate_program_complex32_source(&self, source: &NativeProgramSource) -> Result<()> {
        source.validate()?;
        if source.backend != Backend::Cuda {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA F32-storage execution requires a CUDA native program",
            ));
        }
        if source.program.scalar == ScalarType::F64 && !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "f64 compute with f32 storage",
            });
        }
        Ok(())
    }

    fn validate_resident_chain_transition(
        &self,
        producer: &NativeProgramSource,
        consumer: &NativeProgramSource,
    ) -> Result<()> {
        let output = producer.program.output_resource()?;
        let input = consumer.program.input_resource()?;
        if output.scalar != input.scalar
            || output.elements != input.elements
            || output.external_layout != input.external_layout
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA resident-chain adjacent external resources are incompatible",
            ));
        }
        Ok(())
    }

    fn validate_dense_f32_chain_boundary(
        &self,
        resource: &crate::program_ir::ProgramResource,
        logical_elements: usize,
    ) -> Result<()> {
        let layout = resource.external_layout.ok_or(VkFftError::InvalidKernelIr(
            "CUDA direct resident-chain boundary is missing its external layout",
        ))?;
        if resource.scalar != ScalarType::F32
            || layout.element_shape != crate::program_ir::ProgramElementShape::Complex
            || layout.physical_stride != layout.logical_len
            || layout.logical_elements()? != logical_elements
            || resource.elements != logical_elements
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA direct resident-chain boundary requires dense contiguous Complex32/F32 storage",
            ));
        }
        Ok(())
    }

    pub fn submit_program_complex32<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<CudaProgramTicket32<'a>> {
        self.validate_program_complex32_source(source)?;
        let prepared = prepare_program_complex32(Backend::Cuda, &source.program, input)?;
        Ok(CudaProgramTicket32 {
            pending: self.submit_prepared(source, prepared)?,
        })
    }

    pub fn execute_program_complex32(
        &self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        self.submit_program_complex32(source, input)?.wait()
    }

    /// Execute a sequence of CUDA ProgramIr graphs while keeping every intermediate
    /// external output resident on the device. The first program consumes the caller's
    /// host input and the final program alone performs D2H readback; adjacent programs
    /// must have identical external scalar/layout/physical byte contracts.
    pub fn execute_program_chain_complex32(
        &self,
        sources: &[&NativeProgramSource],
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        let (first, remaining) = sources.split_first().ok_or(VkFftError::InvalidKernelIr(
            "CUDA resident program chain requires at least one program",
        ))?;
        for source in sources {
            self.validate_program_complex32_source(source)?;
        }
        for adjacent in sources.windows(2) {
            self.validate_resident_chain_transition(adjacent[0], adjacent[1])?;
        }

        let mut pending = self.submit_program_complex32(first, input)?.pending;
        for source in remaining {
            let resident_input = pending.into_resident_output()?;
            let prepared =
                prepare_program_complex32_resident_input(Backend::Cuda, &source.program)?;
            pending = self.submit_prepared_with_resident_input(source, prepared, resident_input)?;
        }
        CudaProgramTicket32 { pending }.wait()
    }

    /// Execute a dense-F32 CUDA ProgramIr chain directly from caller input into caller
    /// output storage. This is the allocation-free host-boundary companion to
    /// [`Self::execute_program_chain_complex32`]: intermediate programs stay resident,
    /// the first H2D reads the caller's `Complex32` slice directly, and the final D2H
    /// writes the caller's output slice directly. Formatted/padded/F16 external layouts
    /// remain on the existing prepared-storage path.
    pub fn execute_program_chain_complex32_into(
        &self,
        sources: &[&NativeProgramSource],
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> Result<()> {
        let (first, remaining) = sources.split_first().ok_or(VkFftError::InvalidKernelIr(
            "CUDA resident program chain requires at least one program",
        ))?;
        for source in sources {
            self.validate_program_complex32_source(source)?;
        }
        for adjacent in sources.windows(2) {
            self.validate_resident_chain_transition(adjacent[0], adjacent[1])?;
        }
        self.validate_dense_f32_chain_boundary(first.program.input_resource()?, input.len())?;
        self.validate_dense_f32_chain_boundary(
            sources
                .last()
                .expect("resident chain is non-empty")
                .program
                .output_resource()?,
            output.len(),
        )?;

        let prepared = prepare_program_complex32_resident_input(Backend::Cuda, &first.program)?;
        let mut pending = self.submit_prepared_with_host_input(first, prepared, input)?;
        for source in remaining {
            let resident_input = pending.into_resident_output()?;
            let prepared =
                prepare_program_complex32_resident_input(Backend::Cuda, &source.program)?;
            pending = self.submit_prepared_with_resident_input(source, prepared, resident_input)?;
        }
        pending.finish_into_complex32(output)
    }

    pub fn submit_program_complex64<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<CudaProgramTicket64<'a>> {
        source.validate()?;
        if source.backend != Backend::Cuda || source.program.scalar != ScalarType::F64 {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA F64 execution requires a CUDA/F64 native program",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "f64",
            });
        }
        let prepared = prepare_program_complex64(Backend::Cuda, &source.program, input)?;
        Ok(CudaProgramTicket64 {
            pending: self.submit_prepared(source, prepared)?,
        })
    }

    pub fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        self.submit_program_complex64(source, input)?.wait()
    }

    pub fn execute_program_double_double(
        &self,
        source: &NativeProgramSource,
        input: &[ComplexDoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>> {
        source.validate()?;
        if source.backend != Backend::Cuda || source.program.scalar != ScalarType::DoubleDouble {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA double-double execution requires a CUDA/DD native program",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let prepared = prepare_program_double_double(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(source, prepared)?;
        pending.finish()?;
        pending.prepared.output_double_double()
    }

    pub fn execute_program_double_double_f64_storage(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        source.validate()?;
        if source.backend != Backend::Cuda || source.program.scalar != ScalarType::DoubleDouble {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD/F64-storage execution requires a CUDA/DD native program",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let prepared = prepare_program_complex64(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(source, prepared)?;
        pending.finish()?;
        pending.prepared.output_complex64()
    }

    pub fn execute_double_double_r2c(
        &self,
        ir: &crate::DoubleDoubleRealFftIr,
        input: &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>> {
        if ir.kind != crate::RealFftKind::RealToComplex
            || ir.external_storage != crate::PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD R2C runtime requires a full-DD R2C IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_real(ir)?;
        let prepared = prepare_program_double_double_scalar(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        pending.prepared.output_double_double()
    }

    pub fn execute_double_double_r2c_f64_storage(
        &self,
        ir: &crate::DoubleDoubleRealFftIr,
        input: &[f64],
    ) -> Result<Vec<Complex64>> {
        if ir.kind != crate::RealFftKind::RealToComplex
            || ir.external_storage != crate::PrecisionStorage::F64
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD/F64 R2C runtime requires an F64-storage R2C IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_real(ir)?;
        let prepared = prepare_program_f64_scalar(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        pending.prepared.output_complex64()
    }

    pub fn execute_double_double_c2r(
        &self,
        ir: &crate::DoubleDoubleRealFftIr,
        input: &[ComplexDoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        if ir.kind != crate::RealFftKind::ComplexToReal
            || ir.external_storage != crate::PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD C2R runtime requires a full-DD C2R IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_real(ir)?;
        let prepared = prepare_program_double_double(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        pending.prepared.output_double_double_scalar()
    }

    pub fn execute_double_double_c2r_f64_storage(
        &self,
        ir: &crate::DoubleDoubleRealFftIr,
        input: &[Complex64],
    ) -> Result<Vec<f64>> {
        if ir.kind != crate::RealFftKind::ComplexToReal
            || ir.external_storage != crate::PrecisionStorage::F64
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD/F64 C2R runtime requires an F64-storage C2R IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_real(ir)?;
        let prepared = prepare_program_complex64(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        pending.prepared.output_f64_scalar()
    }

    pub fn execute_double_double_nd_r2c(
        &self,
        ir: &crate::DoubleDoubleNdRealFftIr,
        input: &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>> {
        if ir.kind != crate::RealFftKind::RealToComplex
            || ir.external_storage != crate::PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD ND R2C runtime requires a full-DD R2C IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_nd_real(ir)?;
        let physical_input = ir.pack_formatted_input(input)?;
        let prepared =
            prepare_program_double_double_scalar(Backend::Cuda, &source.program, &physical_input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        let physical_output = pending.prepared.output_double_double()?;
        ir.unpack_formatted_output(&physical_output)
    }

    pub fn execute_double_double_nd_r2c_f64_storage(
        &self,
        ir: &crate::DoubleDoubleNdRealFftIr,
        input: &[f64],
    ) -> Result<Vec<Complex64>> {
        if ir.kind != crate::RealFftKind::RealToComplex
            || ir.external_storage != crate::PrecisionStorage::F64
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD/F64 ND R2C runtime requires an F64-storage R2C IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_nd_real(ir)?;
        let physical_input = ir.pack_formatted_input(input)?;
        let prepared = prepare_program_f64_scalar(Backend::Cuda, &source.program, &physical_input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        let physical_output = pending.prepared.output_complex64()?;
        ir.unpack_formatted_output(&physical_output)
    }

    pub fn execute_double_double_nd_c2r(
        &self,
        ir: &crate::DoubleDoubleNdRealFftIr,
        input: &[ComplexDoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        if ir.kind != crate::RealFftKind::ComplexToReal
            || ir.external_storage != crate::PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD ND C2R runtime requires a full-DD C2R IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_nd_real(ir)?;
        let physical_input = ir.pack_formatted_input(input)?;
        let prepared =
            prepare_program_double_double(Backend::Cuda, &source.program, &physical_input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        let physical_output = pending.prepared.output_double_double_scalar()?;
        ir.unpack_formatted_output(&physical_output)
    }

    pub fn execute_double_double_nd_c2r_f64_storage(
        &self,
        ir: &crate::DoubleDoubleNdRealFftIr,
        input: &[Complex64],
    ) -> Result<Vec<f64>> {
        if ir.kind != crate::RealFftKind::ComplexToReal
            || ir.external_storage != crate::PrecisionStorage::F64
        {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA DD/F64 ND C2R runtime requires an F64-storage C2R IR",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_nd_real(ir)?;
        let physical_input = ir.pack_formatted_input(input)?;
        let prepared = prepare_program_complex64(Backend::Cuda, &source.program, &physical_input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        let physical_output = pending.prepared.output_f64_scalar()?;
        ir.unpack_formatted_output(&physical_output)
    }

    pub fn execute_double_double_r2r(
        &self,
        ir: &crate::DoubleDoubleR2rIr,
        input: &[DoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        if ir.external_storage != crate::PrecisionStorage::DoubleDouble {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA double-double R2R runtime",
                precision: "IR uses F64 external storage",
            });
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_r2r(ir)?;
        let prepared = prepare_program_double_double_scalar(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        pending.prepared.output_double_double_scalar()
    }

    pub fn execute_double_double_r2r_f64_storage(
        &self,
        ir: &crate::DoubleDoubleR2rIr,
        input: &[f64],
    ) -> Result<Vec<f64>> {
        if ir.external_storage != crate::PrecisionStorage::F64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA double-double R2R runtime",
                precision: "IR uses double-double external storage",
            });
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_r2r(ir)?;
        let prepared = prepare_program_f64_scalar(Backend::Cuda, &source.program, input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        pending.prepared.output_f64_scalar()
    }

    pub fn execute_double_double_nd_r2r(
        &self,
        ir: &crate::DoubleDoubleNdR2rIr,
        input: &[DoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        if ir.external_storage != crate::PrecisionStorage::DoubleDouble {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA double-double ND R2R runtime",
                precision: "IR uses F64 external storage",
            });
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_nd_r2r(ir)?;
        let physical_input = ir.pack_formatted_input(input)?;
        let prepared =
            prepare_program_double_double_scalar(Backend::Cuda, &source.program, &physical_input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        let physical_output = pending.prepared.output_double_double_scalar()?;
        ir.unpack_formatted_output(&physical_output)
    }

    pub fn execute_double_double_nd_r2r_f64_storage(
        &self,
        ir: &crate::DoubleDoubleNdR2rIr,
        input: &[f64],
    ) -> Result<Vec<f64>> {
        if ir.external_storage != crate::PrecisionStorage::F64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA double-double ND R2R runtime",
                precision: "IR uses double-double external storage",
            });
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "CUDA runtime",
                precision: "double-double",
            });
        }
        let source = NativeSourceBackend::new(Backend::Cuda).lower_double_double_nd_r2r(ir)?;
        let physical_input = ir.pack_formatted_input(input)?;
        let prepared = prepare_program_f64_scalar(Backend::Cuda, &source.program, &physical_input)?;
        let mut pending = self.submit_prepared(&source, prepared)?;
        pending.finish()?;
        let physical_output = pending.prepared.output_f64_scalar()?;
        ir.unpack_formatted_output(&physical_output)
    }

    pub fn execute_transform_double_double_r2r(
        &self,
        ir: &TransformIr,
        input: &[DoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        match ir {
            TransformIr::RealToRealDoubleDouble(r2r) => self.execute_double_double_r2r(r2r, input),
            TransformIr::RealToRealNdDoubleDouble(r2r) => {
                self.execute_double_double_nd_r2r(r2r, input)
            }
            _ => Err(VkFftError::UnsupportedKernelPath(
                "high-level CUDA DD R2R execution requires a double-double DCT/DST transform",
            )),
        }
    }

    pub fn execute_transform_double_double_r2r_f64_storage(
        &self,
        ir: &TransformIr,
        input: &[f64],
    ) -> Result<Vec<f64>> {
        match ir {
            TransformIr::RealToRealDoubleDouble(r2r) => {
                self.execute_double_double_r2r_f64_storage(r2r, input)
            }
            TransformIr::RealToRealNdDoubleDouble(r2r) => {
                self.execute_double_double_nd_r2r_f64_storage(r2r, input)
            }
            _ => Err(VkFftError::UnsupportedKernelPath(
                "high-level CUDA DD/F64 R2R execution requires a double-double DCT/DST transform",
            )),
        }
    }

    pub fn execute_transform_double_double(
        &self,
        ir: &TransformIr,
        input: &[ComplexDoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>> {
        let source = NativeSourceBackend::new(Backend::Cuda).lower_transform(ir)?;
        if let TransformIr::ComplexNdDoubleDouble(nd) = ir {
            let physical_input = nd.pack_formatted_input(input)?;
            let physical_output = self.execute_program_double_double(&source, &physical_input)?;
            nd.unpack_formatted_output(&physical_output)
        } else {
            self.execute_program_double_double(&source, input)
        }
    }

    pub fn execute_transform_double_double_f64_storage(
        &self,
        ir: &TransformIr,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        let source = NativeSourceBackend::new(Backend::Cuda).lower_transform(ir)?;
        if let TransformIr::ComplexNdDoubleDouble(nd) = ir {
            let physical_input = nd.pack_formatted_input(input)?;
            let physical_output =
                self.execute_program_double_double_f64_storage(&source, &physical_input)?;
            nd.unpack_formatted_output(&physical_output)
        } else {
            self.execute_program_double_double_f64_storage(&source, input)
        }
    }

    pub fn submit_transform_complex32<'a>(
        &'a self,
        ir: &TransformIr,
        input: &[Complex32],
    ) -> Result<CudaProgramTicket32<'a>> {
        let source = NativeSourceBackend::new(Backend::Cuda).lower_transform(ir)?;
        self.submit_program_complex32(&source, input)
    }

    pub fn execute_transform_complex32(
        &self,
        ir: &TransformIr,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        self.submit_transform_complex32(ir, input)?.wait()
    }

    pub fn submit_transform_complex64<'a>(
        &'a self,
        ir: &TransformIr,
        input: &[Complex64],
    ) -> Result<CudaProgramTicket64<'a>> {
        let source = NativeSourceBackend::new(Backend::Cuda).lower_transform(ir)?;
        self.submit_program_complex64(&source, input)
    }

    pub fn execute_transform_complex64(
        &self,
        ir: &TransformIr,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        self.submit_transform_complex64(ir, input)?.wait()
    }

    pub fn submit_transform_f32<'a>(
        &'a self,
        ir: &TransformIr,
        input: NativeTransformInput32<'_>,
    ) -> Result<CudaTransformTicket32<'a>> {
        let source = NativeSourceBackend::new(Backend::Cuda).lower_transform(ir)?;
        let complex_input = prepare_transform_input32(ir, input)?;
        Ok(CudaTransformTicket32 {
            ir: ir.clone(),
            ticket: self.submit_program_complex32(&source, &complex_input)?,
        })
    }

    pub fn execute_transform_f32(
        &self,
        ir: &TransformIr,
        input: NativeTransformInput32<'_>,
    ) -> Result<NativeTransformOutput32> {
        self.submit_transform_f32(ir, input)?.wait()
    }

    pub fn submit_transform_f64<'a>(
        &'a self,
        ir: &TransformIr,
        input: NativeTransformInput64<'_>,
    ) -> Result<CudaTransformTicket64<'a>> {
        let source = NativeSourceBackend::new(Backend::Cuda).lower_transform(ir)?;
        let complex_input = prepare_transform_input64(ir, input)?;
        Ok(CudaTransformTicket64 {
            ir: ir.clone(),
            ticket: self.submit_program_complex64(&source, &complex_input)?,
        })
    }

    pub fn execute_transform_f64(
        &self,
        ir: &TransformIr,
        input: NativeTransformInput64<'_>,
    ) -> Result<NativeTransformOutput64> {
        self.submit_transform_f64(ir, input)?.wait()
    }

    fn submit_prepared<'a>(
        &'a self,
        source: &NativeProgramSource,
        prepared: PreparedProgramStorage,
    ) -> Result<CudaPendingProgram<'a>> {
        self.submit_prepared_impl(source, prepared, None, None)
    }

    fn submit_prepared_with_resident_input<'a>(
        &'a self,
        source: &NativeProgramSource,
        prepared: PreparedProgramStorage,
        resident_input: CudaResidentProgramOutput<'a>,
    ) -> Result<CudaPendingProgram<'a>> {
        self.submit_prepared_impl(source, prepared, Some(resident_input), None)
    }

    fn submit_prepared_with_host_input<'a>(
        &'a self,
        source: &NativeProgramSource,
        prepared: PreparedProgramStorage,
        host_input: &[Complex32],
    ) -> Result<CudaPendingProgram<'a>> {
        self.submit_prepared_impl(source, prepared, None, Some(host_input))
    }

    fn submit_prepared_impl<'a>(
        &'a self,
        source: &NativeProgramSource,
        prepared: PreparedProgramStorage,
        mut resident_input: Option<CudaResidentProgramOutput<'a>>,
        host_input: Option<&[Complex32]>,
    ) -> Result<CudaPendingProgram<'a>> {
        check_cuda(
            unsafe { (self.driver.ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )?;
        let stream = self.take_stream()?;

        let resident_input_allocation =
            if let Some(resident) = resident_input.as_ref() {
                let input_resource = source.program.input_resource()?;
                let input_layout =
                    input_resource
                        .external_layout
                        .ok_or(VkFftError::InvalidKernelIr(
                            "CUDA resident-chain input is missing its external layout",
                        ))?;
                let allocation_id = prepared.memory_plan.allocation_for(input_resource.id)?;
                let allocation = prepared
                    .memory_plan
                    .allocations
                    .get(allocation_id.0)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "CUDA resident-chain input references a missing allocation",
                    ))?;
                let prepared_allocation = prepared.allocations.get(allocation_id.0).ok_or(
                    VkFftError::InvalidKernelIr(
                        "CUDA resident-chain input is missing prepared storage",
                    ),
                )?;
                if allocation.kind != ProgramAllocationKind::ExternalInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "CUDA resident-chain input must map to ExternalInput",
                    ));
                }
                if resident.scalar != input_resource.scalar || resident.layout != input_layout {
                    return Err(VkFftError::InvalidKernelIr(
                        "CUDA resident-chain adjacent external layouts are incompatible",
                    ));
                }
                if resident.memory.bytes != prepared_allocation.byte_len {
                    return Err(VkFftError::InvalidKernelIr(
                        "CUDA resident-chain adjacent physical allocation sizes differ",
                    ));
                }
                if prepared_allocation.host_bytes.is_some() {
                    return Err(VkFftError::InvalidKernelIr(
                        "CUDA resident-chain input unexpectedly materialized host bytes",
                    ));
                }
                Some(allocation_id.0)
            } else {
                None
            };

        if resident_input.is_some() && host_input.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA program cannot consume both resident and direct host input",
            ));
        }
        let host_input_allocation =
            if let Some(host_input) = host_input {
                let input_resource = source.program.input_resource()?;
                let allocation_id = prepared.memory_plan.allocation_for(input_resource.id)?;
                let allocation = prepared
                    .memory_plan
                    .allocations
                    .get(allocation_id.0)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "CUDA direct host input references a missing allocation",
                    ))?;
                let prepared_allocation = prepared.allocations.get(allocation_id.0).ok_or(
                    VkFftError::InvalidKernelIr(
                        "CUDA direct host input is missing prepared storage",
                    ),
                )?;
                if allocation.kind != ProgramAllocationKind::ExternalInput
                    || input_resource.scalar != ScalarType::F32
                    || prepared_allocation.host_bytes.is_some()
                    || std::mem::size_of_val(host_input) != prepared_allocation.byte_len
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "CUDA direct host input does not match the dense F32 external allocation",
                    ));
                }
                Some(allocation_id.0)
            } else {
                None
            };

        let mut pending_luts = Vec::new();
        let mut allocations = Vec::with_capacity(prepared.allocations.len());
        for (index, (allocation, prepared_allocation)) in prepared
            .memory_plan
            .allocations
            .iter()
            .zip(&prepared.allocations)
            .enumerate()
        {
            let byte_len = prepared_allocation.byte_len;
            let host_bytes = prepared_allocation.host_bytes.as_deref();
            if Some(index) == resident_input_allocation {
                let CudaResidentProgramOutput { memory, .. } = resident_input
                    .take()
                    .expect("resident input allocation validated above");
                allocations.push(memory);
            } else if Some(index) == host_input_allocation {
                allocations.push(self.take_transient_and_upload_complex32(
                    host_input.expect("direct host input validated above"),
                )?);
            } else if allocation.kind == ProgramAllocationKind::LookupTable {
                let bytes = host_bytes.ok_or(VkFftError::InvalidKernelIr(
                    "CUDA lookup-table allocation is missing initialization bytes",
                ))?;
                let cached = self
                    .lut_cache
                    .lock()
                    .map_err(|_| runtime_error(Backend::Cuda, "CUDA LUT cache lock is poisoned"))?
                    .get(bytes)
                    .copied();
                if let Some(ptr) = cached {
                    allocations.push(CudaDeviceMemory {
                        api: &self.driver,
                        ptr,
                        bytes: byte_len,
                        owned: false,
                    });
                } else {
                    allocations.push(self.allocate_and_upload(bytes)?);
                    pending_luts.push((index, bytes.to_vec()));
                }
            } else if let Some(bytes) = host_bytes {
                allocations.push(self.take_transient_and_upload(bytes)?);
            } else {
                allocations.push(self.take_transient(byte_len)?);
            }
        }
        if resident_input.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA resident-chain input allocation was not consumed",
            ));
        }
        let modules = source
            .shaders
            .iter()
            .map(|shader| self.compile_and_load(shader))
            .collect::<Result<Vec<_>>>()?;

        for ((pass, shader), module) in source
            .program
            .passes
            .iter()
            .zip(&source.shaders)
            .zip(&modules)
        {
            let mut ordered = pass.bindings.iter().collect::<Vec<_>>();
            ordered.sort_by_key(|binding| binding.binding);
            let mut argument_values = ordered
                .iter()
                .map(|binding| {
                    let allocation_id = prepared.memory_plan.allocation_for(binding.resource)?;
                    allocations
                        .get(allocation_id.0)
                        .map(|allocation| allocation.ptr)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "CUDA pass references a missing device allocation",
                        ))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut argument_pointers = argument_values
                .iter_mut()
                .map(|value| (value as *mut CuDevicePtr).cast::<c_void>())
                .collect::<Vec<_>>();
            check_cuda(
                unsafe {
                    (self.driver.launch_kernel)(
                        module.function,
                        shader.dispatch.x,
                        shader.dispatch.y,
                        shader.dispatch.z,
                        shader.workgroup_size.x,
                        shader.workgroup_size.y,
                        shader.workgroup_size.z,
                        0,
                        stream.stream,
                        argument_pointers.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "cuLaunchKernel",
            )?;
        }

        Ok(CudaPendingProgram {
            stream,
            context: self,
            prepared,
            allocations,
            pending_luts,
            completed: false,
        })
    }

    fn take_stream(&self) -> Result<CudaOwnedStream<'_>> {
        if let Some(stream) = self
            .stream_pool
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA stream pool lock is poisoned"))?
            .pop()
        {
            return Ok(CudaOwnedStream {
                api: &self.driver,
                stream,
                owned: true,
            });
        }
        let mut stream = ptr::null_mut();
        check_cuda(
            unsafe { (self.driver.stream_create)(&mut stream, 0) },
            "cuStreamCreate",
        )?;
        Ok(CudaOwnedStream {
            api: &self.driver,
            stream,
            owned: true,
        })
    }

    fn take_host_output_buffer(&self, bytes: usize) -> Result<Vec<u8>> {
        if let Some(buffer) = self
            .host_output_pool
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA host output pool lock is poisoned"))?
            .remove(&bytes)
        {
            return Ok(buffer);
        }
        // A non-zero fill commits newly allocated anonymous pages on the CPU before the
        // CUDA driver writes into them. Large cold calloc-backed buffers otherwise make
        // synchronous D2H copies pay page-fault/COW costs inside `cuMemcpyDtoH_v2`.
        // The staging bytes are fully overwritten before decode.
        Ok(vec![0xa5; bytes])
    }

    fn recycle_host_output_buffer(&self, buffer: Vec<u8>) {
        let bytes = buffer.len();
        if let Ok(mut pool) = self.host_output_pool.lock() {
            pool.entry(bytes).or_insert(buffer);
        }
    }

    fn allocate_device<'a>(&'a self, bytes: usize) -> Result<CudaDeviceMemory<'a>> {
        let mut ptr = 0;
        check_cuda(
            unsafe { (self.driver.mem_alloc)(&mut ptr, bytes) },
            "cuMemAlloc_v2",
        )?;
        Ok(CudaDeviceMemory {
            api: &self.driver,
            ptr,
            bytes,
            owned: true,
        })
    }

    fn allocate_and_upload<'a>(&'a self, bytes: &[u8]) -> Result<CudaDeviceMemory<'a>> {
        let allocation = self.allocate_device(bytes.len())?;
        check_cuda(
            unsafe {
                (self.driver.memcpy_htod)(allocation.ptr, bytes.as_ptr().cast(), bytes.len())
            },
            "cuMemcpyHtoD_v2",
        )?;
        Ok(allocation)
    }

    fn take_transient<'a>(&'a self, bytes: usize) -> Result<CudaDeviceMemory<'a>> {
        let pooled = self
            .transient_buffer_pool
            .lock()
            .map_err(|_| {
                runtime_error(Backend::Cuda, "CUDA transient buffer pool lock is poisoned")
            })?
            .get_mut(&bytes)
            .and_then(Vec::pop);
        if let Some(ptr) = pooled {
            Ok(CudaDeviceMemory {
                api: &self.driver,
                ptr,
                bytes,
                owned: true,
            })
        } else {
            self.allocate_device(bytes)
        }
    }

    fn take_transient_and_upload<'a>(&'a self, bytes: &[u8]) -> Result<CudaDeviceMemory<'a>> {
        let allocation = self.take_transient(bytes.len())?;
        check_cuda(
            unsafe {
                (self.driver.memcpy_htod)(allocation.ptr, bytes.as_ptr().cast(), bytes.len())
            },
            "cuMemcpyHtoD_v2",
        )?;
        Ok(allocation)
    }

    fn take_transient_and_upload_complex32<'a>(
        &'a self,
        values: &[Complex32],
    ) -> Result<CudaDeviceMemory<'a>> {
        let bytes = std::mem::size_of_val(values);
        let allocation = self.take_transient(bytes)?;
        check_cuda(
            unsafe { (self.driver.memcpy_htod)(allocation.ptr, values.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD_v2",
        )?;
        Ok(allocation)
    }

    fn compile_and_load<'a>(&'a self, shader: &NativeShaderSource) -> Result<CudaLoadedModule<'a>> {
        if let Some((module, function)) = self
            .module_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Cuda, "CUDA module cache lock is poisoned"))?
            .get(&shader.source)
            .copied()
        {
            return Ok(CudaLoadedModule {
                api: &self.driver,
                module,
                function,
                owned: false,
            });
        }
        let ptx = self.compile_ptx(shader)?;
        let mut module = ptr::null_mut();
        check_cuda(
            unsafe { (self.driver.module_load_data)(&mut module, ptx.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        let mut function = ptr::null_mut();
        let entry = CString::new(shader.entry_point).expect("static CUDA entry point");
        if let Err(error) = check_cuda(
            unsafe { (self.driver.module_get_function)(&mut function, module, entry.as_ptr()) },
            "cuModuleGetFunction",
        ) {
            unsafe {
                (self.driver.module_unload)(module);
            }
            return Err(error);
        }
        let mut cache = match self.module_cache.lock() {
            Ok(cache) => cache,
            Err(_) => {
                unsafe {
                    (self.driver.module_unload)(module);
                }
                return Err(runtime_error(
                    Backend::Cuda,
                    "CUDA module cache lock is poisoned",
                ));
            }
        };
        cache.insert(shader.source.clone(), (module, function));
        Ok(CudaLoadedModule {
            api: &self.driver,
            module,
            function,
            owned: false,
        })
    }
}

impl crate::backend::native_runtime::NativeProgramTicket32 for CudaProgramTicket32<'_> {
    fn wait(self) -> Result<Vec<Complex32>> {
        CudaProgramTicket32::wait(self)
    }
}

impl crate::backend::native_runtime::NativeProgramTicket64 for CudaProgramTicket64<'_> {
    fn wait(self) -> Result<Vec<Complex64>> {
        CudaProgramTicket64::wait(self)
    }
}

impl crate::backend::native_runtime::NativeAsyncRuntime for CudaExecutionContext {
    type Ticket32<'a> = CudaProgramTicket32<'a>;
    type Ticket64<'a> = CudaProgramTicket64<'a>;

    fn submit_program_complex32_async<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Self::Ticket32<'a>> {
        self.submit_program_complex32(source, input)
    }

    fn submit_program_complex64_async<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Self::Ticket64<'a>> {
        self.submit_program_complex64(source, input)
    }
}

impl crate::backend::native_runtime::NativeRuntime for CudaExecutionContext {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn device_profile(&self) -> DeviceProfile {
        CudaExecutionContext::device_profile(self)
    }

    fn device_name(&self) -> &str {
        CudaExecutionContext::device_name(self)
    }

    fn compiled_pass_resource_reports(
        &self,
        source: &NativeProgramSource,
    ) -> Result<Vec<NativeCompiledPassResourceReport>> {
        source.validate()?;
        if source.backend != Backend::Cuda {
            return Err(VkFftError::InvalidKernelIr(
                "CUDA compiled-resource reporting requires a CUDA native program",
            ));
        }
        check_cuda(
            unsafe { (self.driver.ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )?;
        let mut reports = Vec::with_capacity(source.shaders.len());
        for (pass, shader) in source.program.passes.iter().zip(&source.shaders) {
            let loaded = self.compile_and_load(shader)?;
            reports.push(NativeCompiledPassResourceReport {
                pass_name: pass.name.clone(),
                metrics: self.compiled_function_resource_metrics(loaded.function)?,
            });
        }
        Ok(reports)
    }

    fn theoretical_occupancy_reports(
        &self,
        source: &NativeProgramSource,
        compiled_reports: &[NativeCompiledPassResourceReport],
    ) -> Result<Vec<NativeTheoreticalOccupancyReport>> {
        self.theoretical_occupancy_reports_from_compiled(source, compiled_reports)
    }

    fn execute_program_complex32(
        &self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        CudaExecutionContext::execute_program_complex32(self, source, input)
    }

    fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        CudaExecutionContext::execute_program_complex64(self, source, input)
    }
}

impl Drop for CudaExecutionContext {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe {
                (self.driver.ctx_set_current)(self.context);
            }
            let modules = match self.module_cache.get_mut() {
                Ok(cache) => cache,
                Err(poisoned) => poisoned.into_inner(),
            };
            for (_, (module, _)) in modules.drain() {
                unsafe {
                    (self.driver.module_unload)(module);
                }
            }
            let luts = match self.lut_cache.get_mut() {
                Ok(cache) => cache,
                Err(poisoned) => poisoned.into_inner(),
            };
            for (_, ptr) in luts.drain() {
                unsafe {
                    (self.driver.mem_free)(ptr);
                }
            }
            let pool = match self.transient_buffer_pool.get_mut() {
                Ok(pool) => pool,
                Err(poisoned) => poisoned.into_inner(),
            };
            for ptr in pool.drain().flat_map(|(_, buffers)| buffers) {
                unsafe {
                    (self.driver.mem_free)(ptr);
                }
            }
            let host_outputs = match self.host_output_pool.get_mut() {
                Ok(pool) => pool,
                Err(poisoned) => poisoned.into_inner(),
            };
            host_outputs.clear();
            let streams = match self.stream_pool.get_mut() {
                Ok(pool) => pool,
                Err(poisoned) => poisoned.into_inner(),
            };
            for stream in streams.drain(..) {
                unsafe {
                    (self.driver.stream_destroy)(stream);
                }
            }
            unsafe {
                (self.driver.ctx_destroy)(self.context);
            }
            self.context = ptr::null_mut();
        }
    }
}

fn device_attribute(api: &CudaDriverApi, device: CuDevice, attribute: c_int) -> Result<c_int> {
    let mut value = 0;
    check_cuda(
        unsafe { (api.device_get_attribute)(&mut value, attribute, device) },
        "cuDeviceGetAttribute",
    )?;
    Ok(value)
}

fn device_attribute_usize(
    api: &CudaDriverApi,
    device: CuDevice,
    attribute: c_int,
    field: &'static str,
) -> Result<usize> {
    let value = device_attribute(api, device, attribute)?;
    let value = usize::try_from(value).map_err(|_| VkFftError::ValueOutOfRange { field })?;
    if value == 0 {
        return Err(VkFftError::ValueOutOfRange { field });
    }
    Ok(value)
}

fn optional_device_attribute_usize(
    api: &CudaDriverApi,
    device: CuDevice,
    attribute: c_int,
) -> Option<usize> {
    let mut value = 0;
    let result = unsafe { (api.device_get_attribute)(&mut value, attribute, device) };
    (result == CUDA_SUCCESS && value > 0).then_some(value as usize)
}

fn floor_power_of_two(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Direction, FftConfig, Precision};
    use std::ops::Deref;

    struct TestCudaExecutionContext {
        context: CudaExecutionContext,
        _guard: crate::backend::GpuTestContextGuard,
    }

    impl Deref for TestCudaExecutionContext {
        type Target = CudaExecutionContext;

        fn deref(&self) -> &Self::Target {
            &self.context
        }
    }

    fn new_test_context(device_index: usize) -> Result<TestCudaExecutionContext> {
        let guard = crate::backend::gpu_test_context_guard();
        let context = CudaExecutionContext::new(device_index)?;
        Ok(TestCudaExecutionContext {
            context,
            _guard: guard,
        })
    }

    fn new_test_context_with_ptx_cache_archive(
        device_index: usize,
        encoded: &[u8],
    ) -> Result<TestCudaExecutionContext> {
        let guard = crate::backend::gpu_test_context_guard();
        let context = CudaExecutionContext::new_with_ptx_cache_archive(device_index, encoded)?;
        Ok(TestCudaExecutionContext {
            context,
            _guard: guard,
        })
    }

    fn context_or_skip() -> Option<TestCudaExecutionContext> {
        match new_test_context(0) {
            Ok(context) => Some(context),
            Err(VkFftError::NativeUnavailable { .. }) => None,
            Err(error) => panic!("CUDA runtime was discovered but could not initialize: {error}"),
        }
    }

    fn max_error32(lhs: &[Complex32], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| {
                let dr = lhs.re as f64 - rhs.re;
                let di = lhs.im as f64 - rhs.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, |maximum, error| {
                if error.is_finite() {
                    maximum.max(error)
                } else {
                    f64::INFINITY
                }
            })
    }

    #[test]
    fn cuda_subgroup_rader_transpose_p152_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        assert_eq!(profile.subgroup.size, 32);
        assert!(profile.subgroup.supports_full_subgroup_shuffle_compute());

        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![152])
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            let subgroup_passes = source
                .shaders
                .iter()
                .filter(|shader| shader.source.contains("__shfl_sync"))
                .collect::<Vec<_>>();
            assert_eq!(subgroup_passes.len(), 2);
            for shader in subgroup_passes {
                assert_eq!(shader.sequence_len, 18);
                assert_eq!(shader.workgroup_size.x, 32);
                assert_eq!(shader.workgroup_size.y, 1);
                assert_eq!(shader.required_shared_memory_bytes, 0);
                assert!(shader.source.contains("__activemask()"));
            }
        }

        let mut input = vec![Complex32::new(0.0, 0.0); 152];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f32::consts::TAU * k as f32 / 152.0;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 1.0e-4,
            "CUDA p152 subgroup Rader-transpose forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 1.0e-4,
            "CUDA p152 subgroup Rader-transpose round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_multi_subgroup_p257_exchange_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        assert_eq!(profile.subgroup.size, 32);
        assert!(profile.subgroup.supports_full_subgroup_shuffle_compute());

        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![1028])
                    .with_batch_count(2)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            let subgroup_passes = source
                .shaders
                .iter()
                .filter(|shader| shader.source.contains("__shfl_sync"))
                .collect::<Vec<_>>();
            assert_eq!(subgroup_passes.len(), 2);
            for shader in subgroup_passes {
                assert_eq!(shader.sequence_len, 256);
                assert_eq!(shader.workgroup_size.x, 64);
                assert_eq!(shader.workgroup_size.y, 1);
                assert_eq!(shader.dispatch.x, 2);
                assert_eq!(shader.required_shared_memory_bytes, 0);
                assert!(shader.source.contains("__activemask()"));
                assert!(shader.source.contains("& 31u"));
                assert!(shader.source.contains(">> 5u"));
            }
        }

        let mut input = vec![Complex32::new(0.0, 0.0); 1028 * 2];
        for batch in 0..2 {
            input[batch * 1028 + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % 1028;
                let angle = -std::f32::consts::TAU * k as f32 / 1028.0;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 1.0e-4,
            "CUDA N1028 multi-subgroup p257 forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 1.0e-4,
            "CUDA N1028 multi-subgroup p257 round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_f64_multi_subgroup_p257_exchange_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64 {
            return;
        }
        assert_eq!(profile.subgroup.size, 32);
        assert!(profile.subgroup.supports_full_subgroup_shuffle_compute());

        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![1028])
                    .with_batch_count(2)
                    .with_precision(Precision::F64)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            let subgroup_passes = source
                .shaders
                .iter()
                .filter(|shader| shader.source.contains("__shfl_sync"))
                .collect::<Vec<_>>();
            assert_eq!(subgroup_passes.len(), 2);
            for shader in subgroup_passes {
                assert_eq!(shader.scalar, ScalarType::F64);
                assert_eq!(shader.sequence_len, 256);
                assert_eq!(shader.workgroup_size.x, 64);
                assert_eq!(shader.workgroup_size.y, 1);
                assert_eq!(shader.dispatch.x, 2);
                assert_eq!(shader.required_shared_memory_bytes, 0);
                assert!(shader.source.contains("__activemask()"));
                assert!(shader.source.contains("& 31u"));
                assert!(shader.source.contains(">> 5u"));
            }
        }

        let mut input = vec![Complex64::new(0.0, 0.0); 1028 * 2];
        for batch in 0..2 {
            input[batch * 1028 + 1] = Complex64::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex64(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % 1028;
                let angle = -std::f64::consts::TAU * k as f64 / 1028.0;
                (*actual - Complex64::new(angle.cos(), angle.sin()))
                    .norm_sqr()
                    .sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            forward_error <= 1.0e-10,
            "CUDA F64 N1028 multi-subgroup p257 forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let restored = context
            .execute_transform_complex64(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f64, f64::max);
        assert!(
            round_trip_error <= 1.0e-10,
            "CUDA F64 N1028 multi-subgroup p257 round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_compiled_resource_report_covers_fused_n51_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let ir = TransformIr::build(
            FftConfig::new(vec![51]).with_batch_count(64),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::Cuda)
            .lower_transform(&ir)
            .unwrap();
        let reports =
            crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                &*context, &source,
            )
            .unwrap();
        assert_eq!(reports.len(), source.program.passes.len());
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].pass_name, source.program.passes[0].name);
        let NativeCompiledResourceMetrics::Cuda {
            registers_per_thread,
            static_shared_memory_bytes_per_block,
            local_memory_bytes_per_thread: _,
            max_threads_per_block,
        } = reports[0].metrics
        else {
            panic!("CUDA resource report returned the wrong backend metric variant");
        };
        assert!(registers_per_thread > 0);
        assert!(
            static_shared_memory_bytes_per_block >= source.shaders[0].required_shared_memory_bytes
        );
        let workgroup = source.shaders[0].workgroup_size;
        assert!(
            max_threads_per_block
                >= workgroup.x as usize * workgroup.y as usize * workgroup.z as usize
        );
        let occupancy =
            crate::backend::native_runtime::NativeRuntime::theoretical_occupancy_reports(
                &*context, &source, &reports,
            )
            .unwrap();
        assert_eq!(occupancy.len(), 1);
        let occupancy = &occupancy[0];
        assert_eq!(occupancy.pass_name, reports[0].pass_name);
        assert_eq!(occupancy.backend, Backend::Cuda);
        assert!(occupancy.compute_unit_count > 0);
        assert_eq!(
            occupancy.subgroup_size,
            context.device_profile().subgroup.size
        );
        assert!(occupancy.workgroup_threads > 0);
        assert!(occupancy.subgroups_per_workgroup > 0);
        assert!(occupancy.max_subgroups_per_compute_unit > 0);
        assert!(occupancy.resident_blocks_per_compute_unit_upper_bound > 0);
        assert!(occupancy.resident_subgroups_per_compute_unit_upper_bound > 0);
        assert!(occupancy.occupancy_basis_points_upper_bound <= 10_000);
        assert!(!occupancy.limiting_resources.is_empty());
    }

    #[test]
    fn cuda_real_gpu_double_double_nd_direct_rader_and_bluestein_axes_match_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for (prime, force_bluestein) in [(47usize, false), (103usize, true), (2_053usize, true)] {
            let dimensions = vec![2usize, prime];
            let mut tuning = crate::PlannerTuning::portable();
            if force_bluestein {
                tuning.max_rader_fft_prime = 100;
            }
            let forward = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap_or_else(|error| panic!("CUDA DD [2,{prime}] forward build failed: {error}"));
            let inverse = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_tuning(tuning),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
                panic!("CUDA DD prime-axis plan must use multidimensional DD IR");
            };
            match (&nd.axes[0].transform, force_bluestein) {
                (crate::DoubleDoubleOneDimIr::DirectRader(_), false) => {}
                (crate::DoubleDoubleOneDimIr::Bluestein(bluestein), true) => {
                    if prime == 2_053 {
                        assert_eq!(bluestein.convolution_len, 4_368);
                        assert!(matches!(
                            bluestein.forward_fft,
                            crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(_)
                        ));
                        assert!(matches!(
                            bluestein.inverse_fft,
                            crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(_)
                        ));
                    }
                }
                (other, _) => panic!("unexpected CUDA DD ND prime child: {other:?}"),
            }
            let input = (0..2 * prime)
                .map(|index| {
                    let x = index as f64;
                    crate::ComplexDoubleDouble::new(
                        crate::DoubleDouble::from_parts(
                            (0.037 * x).sin() + 0.0003 * x,
                            (index + 1) as f64 * 8.0e-32,
                        ),
                        crate::DoubleDouble::from_parts(
                            (0.019 * x).cos() - 0.0002 * x,
                            -(index as f64 + 1.0) * 4.0e-32,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let expected = forward.execute_double_double_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let dd_error = |actual: crate::ComplexDoubleDouble,
                            expected: crate::ComplexDoubleDouble| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            };
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            let forward_tolerance = if prime == 2_053 { 2.0e-15 } else { 5.0e-18 };
            assert!(
                error <= forward_tolerance,
                "CUDA DD [2,{prime}] prime-axis mismatch on {}: {error:e}",
                context.device_name()
            );
            let restored = context
                .execute_transform_double_double(&inverse, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            let round_trip_tolerance = if prime == 2_053 { 5.0e-14 } else { 5.0e-16 };
            assert!(
                round_trip_error <= round_trip_tolerance,
                "CUDA DD [2,{prime}] round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );

            let f64_forward = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_tuning(tuning),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_inverse = TransformIr::build(
                FftConfig::new(dimensions)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(true)
                    .with_tuning(tuning),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(crate::ComplexDoubleDouble::to_complex64)
                .collect::<Vec<_>>();
            let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            let f64_tolerance = if prime == 2_053 { 5.0e-8 } else { 2.0e-10 };
            assert!(
                f64_error <= f64_tolerance,
                "CUDA DD/F64 [2,{prime}] mismatch on {}: {f64_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            let f64_round_trip_tolerance = if prime == 2_053 { 1.0e-8 } else { 5.0e-10 };
            assert!(
                f64_round_trip_error <= f64_round_trip_tolerance,
                "CUDA DD/F64 [2,{prime}] round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_nd_recursive_rader_axes_match_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for axis_len in [17usize * 17, 17usize * 19] {
            let dimensions = vec![2usize, axis_len];
            let forward = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(7)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let inverse = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(7)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
                panic!("CUDA DD [2,{axis_len}] must use multidimensional DD IR");
            };
            assert!(matches!(
                nd.axes[0].transform,
                crate::DoubleDoubleOneDimIr::Recursive(_)
            ));
            let crate::DoubleDoubleOneDimIr::Recursive(recursive_axis) = &nd.axes[0].transform
            else {
                unreachable!("validated CUDA recursive DD axis");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive_axis.root
            else {
                panic!("CUDA grouped DD recursive axis should keep a Cooley root");
            };
            let axis_plan = crate::FftPlan::build_for_device(
                FftConfig::new(vec![axis_len])
                    .with_batch_count(recursive_axis.batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                context.device_profile(),
            )
            .unwrap();
            let expected_axis = crate::DoubleDoubleRecursiveFftIr::build_for_device(
                &axis_plan,
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                expected_root,
            ) = &expected_axis.root
            else {
                panic!("CUDA independent DD recursive Rader axis should keep a Cooley root");
            };
            let expected_axis_block = expected_root.pack_right.axis_batch_block.unwrap();
            let axis_block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(axis_block, expected_axis_block);
            assert_eq!(axis_block.grouped_batch, 3);
            let parent_block = crate::WorkgroupSize {
                x: u32::try_from(axis_block.local_size_x).unwrap(),
                y: u32::try_from(axis_block.local_size_y).unwrap(),
                z: 1,
            };
            assert_eq!(root.pack_right.workgroup_size, parent_block);
            assert_eq!(root.twiddle_transpose.workgroup_size, parent_block);
            assert_eq!(root.scatter_output.workgroup_size, parent_block);
            assert_eq!(root.pack_right.dispatch.x, 5);
            assert_eq!(nd.batch_count, 7);
            assert!(
                nd.axes
                    .iter()
                    .all(|axis| { axis.grouped_batch == 3 && axis.transform.grouped_batch() == 3 })
            );
            assert_eq!(
                nd.axes[0]
                    .transform
                    .batch_count()
                    .div_ceil(nd.axes[0].transform.grouped_batch()),
                5
            );
            assert_eq!(
                nd.axes[1]
                    .transform
                    .batch_count()
                    .div_ceil(nd.axes[1].transform.grouped_batch()),
                (7 * axis_len).div_ceil(3)
            );
            let input = (0..7 * 2 * axis_len)
                .map(|index| {
                    let x = index as f64;
                    crate::ComplexDoubleDouble::new(
                        crate::DoubleDouble::from_parts(
                            (0.031 * x).sin() + 0.00011 * x,
                            (index + 1) as f64 * 7.0e-32,
                        ),
                        crate::DoubleDouble::from_parts(
                            (0.017 * x).cos() - 0.00009 * x,
                            -(index as f64 + 1.0) * 3.0e-32,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let expected = forward.execute_double_double_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let dd_error = |actual: crate::ComplexDoubleDouble,
                            expected: crate::ComplexDoubleDouble| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            };
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                error <= 5.0e-16,
                "CUDA DD ND [2,{axis_len}] mismatch on {}: {error:e}",
                context.device_name()
            );
            let restored = context
                .execute_transform_double_double(&inverse, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 5.0e-14,
                "CUDA DD ND [2,{axis_len}] round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );

            let f64_forward = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(7)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_inverse = TransformIr::build(
                FftConfig::new(dimensions)
                    .with_batch_count(7)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(crate::ComplexDoubleDouble::to_complex64)
                .collect::<Vec<_>>();
            let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_error <= 5.0e-9,
                "CUDA DD/F64 ND [2,{axis_len}] mismatch on {}: {f64_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_round_trip_error <= 5.0e-10,
                "CUDA DD/F64 ND [2,{axis_len}] round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_nd_3x17_fft_rader_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![3usize, 17usize];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone()).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [3,17] plan must use multidimensional DD IR");
        };
        assert!(matches!(
            nd.axes[0].transform,
            crate::DoubleDoubleOneDimIr::FftRader(_)
        ));
        let input = (0..51usize)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.071 * x).sin() + 0.0007 * x,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.043 * x).cos() - 0.0004 * x,
                        -(index as f64 + 1.0) * 6.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 2.0e-18,
            "CUDA DD [3,17] FFT-Rader-axis mismatch on {}: {error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-16,
            "CUDA DD [3,17] round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            FftConfig::new(dimensions).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11,
            "CUDA DD/F64 [3,17] mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn double_double_nd_real_odd_even_match_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for last_len in [7usize, 8usize] {
            let dimensions = vec![3usize, last_len];
            let elements = dimensions.iter().product::<usize>() * batch_count;
            let r2c = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let c2r = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::RealNdDoubleDouble(r2c_ir) = &r2c else {
                panic!("DD ND R2C must build RealNdDoubleDouble IR");
            };
            let TransformIr::RealNdDoubleDouble(c2r_ir) = &c2r else {
                panic!("DD ND C2R must build RealNdDoubleDouble IR");
            };
            // Fixed upstream keeps these small N7/N8 real axes full-complex;
            // bigSequenceEvenR2C is a device-scored large-sequence fallback.
            assert!(!r2c_ir.real_axis.even_half_size);
            let input = (0..elements)
                .map(|index| {
                    let x = index as f64;
                    DoubleDouble::from_parts(
                        (0.083 * x).sin() + 0.17 * (0.031 * x).cos() + 0.001 * x,
                        (index + 1) as f64 * 6.0e-32,
                    )
                })
                .collect::<Vec<_>>();
            let expected = r2c.execute_double_double_r2c_reference(&input).unwrap();
            let actual = context
                .execute_double_double_nd_r2c(r2c_ir, &input)
                .unwrap();
            let complex_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let re = (actual.re - expected.re).abs();
                    let im = (actual.im - expected.im).abs();
                    re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                complex_error <= 2.0e-17,
                "CUDA DD ND [3,{last_len}] R2C mismatch on {}: {complex_error:e}",
                context.device_name()
            );
            let restored = context
                .execute_double_double_nd_c2r(c2r_ir, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 2.0e-16,
                "CUDA DD ND [3,{last_len}] real round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );

            let r2c_f64 = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let c2r_f64 = TransformIr::build(
                FftConfig::new(dimensions)
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::RealNdDoubleDouble(r2c_f64_ir) = &r2c_f64 else {
                panic!("DD/F64 ND R2C must build RealNdDoubleDouble IR");
            };
            let TransformIr::RealNdDoubleDouble(c2r_f64_ir) = &c2r_f64 else {
                panic!("DD/F64 ND C2R must build RealNdDoubleDouble IR");
            };
            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_expected = r2c_f64.execute_r2c_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_double_double_nd_r2c_f64_storage(r2c_f64_ir, &f64_input)
                .unwrap();
            let f64_forward_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_forward_error <= 3.0e-11,
                "CUDA DD/F64 ND [3,{last_len}] R2C mismatch on {}: {f64_forward_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_double_double_nd_c2r_f64_storage(c2r_f64_ir, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_round_trip_error <= 3.0e-12,
                "CUDA DD/F64 ND [3,{last_len}] round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn double_double_nd_real_padding_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![3usize, 8usize];
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let elements = dimensions.iter().product::<usize>() * batch_count;
        let dd_input = (0..elements)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.079 * x).sin() + 0.13 * (0.027 * x).cos() + 0.0011 * x,
                    (index + 1) as f64 * 5.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let build = |transform, precision| {
            let config = FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_transform(transform)
                .with_precision(precision)
                .with_inverse_normalization(transform == crate::TransformKind::ComplexToReal)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_grouped_batch(1, grouped_batch)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap();
            TransformIr::build(
                config,
                if transform == crate::TransformKind::RealToComplex {
                    Direction::Forward
                } else {
                    Direction::Inverse
                },
                context.device_profile(),
            )
            .unwrap()
        };
        let r2c = build(crate::TransformKind::RealToComplex, Precision::DoubleDouble);
        let c2r = build(crate::TransformKind::ComplexToReal, Precision::DoubleDouble);
        let TransformIr::RealNdDoubleDouble(r2c_ir) = &r2c else {
            panic!("padded DD ND R2C must build RealNdDoubleDouble IR");
        };
        let TransformIr::RealNdDoubleDouble(c2r_ir) = &c2r else {
            panic!("padded DD ND C2R must build RealNdDoubleDouble IR");
        };
        let expected = r2c.execute_double_double_r2c_reference(&dd_input).unwrap();
        let actual = context
            .execute_double_double_nd_r2c(r2c_ir, &dd_input)
            .unwrap();
        let complex_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            complex_error <= 2.0e-17,
            "CUDA padded DD ND R2C mismatch: {complex_error:e}"
        );
        let expected_restored = c2r.execute_double_double_c2r_reference(&expected).unwrap();
        let restored = context
            .execute_double_double_nd_c2r(c2r_ir, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-16,
            "CUDA padded DD ND C2R mismatch: {round_trip_error:e}"
        );

        let r2c_f64 = build(
            crate::TransformKind::RealToComplex,
            Precision::DoubleDoubleF64Storage,
        );
        let c2r_f64 = build(
            crate::TransformKind::ComplexToReal,
            Precision::DoubleDoubleF64Storage,
        );
        let TransformIr::RealNdDoubleDouble(r2c_f64_ir) = &r2c_f64 else {
            panic!("padded DD/F64 ND R2C must build RealNdDoubleDouble IR");
        };
        let TransformIr::RealNdDoubleDouble(c2r_f64_ir) = &c2r_f64 else {
            panic!("padded DD/F64 ND C2R must build RealNdDoubleDouble IR");
        };
        let f64_input = dd_input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_expected = r2c_f64.execute_r2c_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_double_double_nd_r2c_f64_storage(r2c_f64_ir, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 3.0e-11,
            "CUDA padded DD/F64 ND R2C mismatch: {f64_forward_error:e}"
        );
        let f64_expected_restored = c2r_f64.execute_c2r_reference(&f64_expected).unwrap();
        let f64_restored = context
            .execute_double_double_nd_c2r_f64_storage(c2r_f64_ir, &f64_actual)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_expected_restored)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 3.0e-12,
            "CUDA padded DD/F64 ND C2R mismatch: {f64_round_trip_error:e}"
        );
    }

    #[test]
    fn double_double_real_padding_odd_even_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let batch_count = 7usize;
        for length in [15usize, 16usize] {
            let build = |precision, transform| {
                let config = FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(transform)
                    .with_precision(precision)
                    .with_inverse_normalization(transform == crate::TransformKind::ComplexToReal)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_zero_padding(0, 3, 7)
                    .unwrap();
                TransformIr::build(
                    config,
                    if transform == crate::TransformKind::RealToComplex {
                        Direction::Forward
                    } else {
                        Direction::Inverse
                    },
                    context.device_profile(),
                )
                .unwrap()
            };
            let r2c = build(Precision::DoubleDouble, crate::TransformKind::RealToComplex);
            let c2r = build(Precision::DoubleDouble, crate::TransformKind::ComplexToReal);
            let TransformIr::RealDoubleDouble(r2c_ir) = &r2c else {
                panic!("padded DD R2C must build RealDoubleDouble IR");
            };
            let TransformIr::RealDoubleDouble(c2r_ir) = &c2r else {
                panic!("padded DD C2R must build RealDoubleDouble IR");
            };
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    crate::DoubleDouble::from_parts(
                        (0.107 * x).sin() + 0.15 * (0.039 * x).cos() + 0.0013 * x,
                        (index + 1) as f64 * 6.0e-32,
                    )
                })
                .collect::<Vec<_>>();
            let expected = r2c.execute_double_double_r2c_reference(&input).unwrap();
            let actual = context.execute_double_double_r2c(r2c_ir, &input).unwrap();
            let complex_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let re = (actual.re - expected.re).abs();
                    let im = (actual.im - expected.im).abs();
                    re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                complex_error <= 2.0e-17,
                "CUDA padded DD real N={length} R2C mismatch on {}: {complex_error:e}",
                context.device_name()
            );
            let expected_restored = c2r.execute_double_double_c2r_reference(&expected).unwrap();
            let restored = context.execute_double_double_c2r(c2r_ir, &actual).unwrap();
            let real_error = restored
                .iter()
                .copied()
                .zip(expected_restored.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                real_error <= 2.0e-16,
                "CUDA padded DD real N={length} C2R mismatch on {}: {real_error:e}",
                context.device_name()
            );

            let r2c_f64 = build(
                Precision::DoubleDoubleF64Storage,
                crate::TransformKind::RealToComplex,
            );
            let c2r_f64 = build(
                Precision::DoubleDoubleF64Storage,
                crate::TransformKind::ComplexToReal,
            );
            let TransformIr::RealDoubleDouble(r2c_f64_ir) = &r2c_f64 else {
                panic!("padded DD/F64 R2C must build RealDoubleDouble IR");
            };
            let TransformIr::RealDoubleDouble(c2r_f64_ir) = &c2r_f64 else {
                panic!("padded DD/F64 C2R must build RealDoubleDouble IR");
            };
            let f64_input = input
                .iter()
                .copied()
                .map(crate::DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_expected = r2c_f64.execute_r2c_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_double_double_r2c_f64_storage(r2c_f64_ir, &f64_input)
                .unwrap();
            let f64_forward_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_forward_error <= 3.0e-11,
                "CUDA padded DD/F64 real N={length} R2C mismatch on {}: {f64_forward_error:e}",
                context.device_name()
            );
            let f64_expected_restored = c2r_f64.execute_c2r_reference(&f64_expected).unwrap();
            let f64_restored = context
                .execute_double_double_c2r_f64_storage(c2r_f64_ir, &f64_actual)
                .unwrap();
            let f64_real_error = f64_restored
                .iter()
                .zip(&f64_expected_restored)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_real_error <= 3.0e-12,
                "CUDA padded DD/F64 real N={length} C2R mismatch on {}: {f64_real_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_even_real_n16_half_size_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 16usize;
        let batch_count = 2usize;
        let r2c = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let c2r = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealDoubleDouble(r2c_ir) = &r2c else {
            panic!("DD N=15 R2C must build RealDoubleDouble IR");
        };
        let TransformIr::RealDoubleDouble(c2r_ir) = &c2r else {
            panic!("DD N=15 C2R must build RealDoubleDouble IR");
        };
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::DoubleDouble::from_parts(
                    (0.13 * x).sin() + 0.002 * x,
                    (index + 1) as f64 * 7.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let expected = r2c.execute_double_double_r2c_reference(&input).unwrap();
        let actual = context.execute_double_double_r2c(r2c_ir, &input).unwrap();
        let complex_dd_error =
            |actual: crate::ComplexDoubleDouble, expected: crate::ComplexDoubleDouble| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| complex_dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-20,
            "CUDA DD even R2C mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context.execute_double_double_c2r(c2r_ir, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-18,
            "CUDA DD even real round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let r2c_f64 = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let c2r_f64 = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealDoubleDouble(r2c_f64_ir) = &r2c_f64 else {
            panic!("DD/F64 N=15 R2C must build RealDoubleDouble IR");
        };
        let TransformIr::RealDoubleDouble(c2r_f64_ir) = &c2r_f64 else {
            panic!("DD/F64 N=15 C2R must build RealDoubleDouble IR");
        };
        let f64_input = input
            .iter()
            .copied()
            .map(crate::DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_expected = r2c_f64.execute_r2c_reference(&f64_input).unwrap();
        let f64_spectrum = context
            .execute_double_double_r2c_f64_storage(r2c_f64_ir, &f64_input)
            .unwrap();
        let f64_forward_error = f64_spectrum
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 2.0e-12,
            "CUDA DD/F64 even R2C mismatch on {}: {f64_forward_error:e}",
            context.device_name()
        );
        let f64_restored = context
            .execute_double_double_c2r_f64_storage(c2r_f64_ir, &f64_spectrum)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 2.0e-13,
            "CUDA DD/F64 even real round trip mismatch on {}: {f64_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_odd_real_n15_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 15usize;
        let batch_count = 2usize;
        let r2c = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let c2r = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealDoubleDouble(r2c_ir) = &r2c else {
            panic!("DD N=15 R2C must build RealDoubleDouble IR");
        };
        let TransformIr::RealDoubleDouble(c2r_ir) = &c2r else {
            panic!("DD N=15 C2R must build RealDoubleDouble IR");
        };
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::DoubleDouble::from_parts(
                    (0.13 * x).sin() + 0.002 * x,
                    (index + 1) as f64 * 7.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let expected = r2c.execute_double_double_r2c_reference(&input).unwrap();
        let actual = context.execute_double_double_r2c(r2c_ir, &input).unwrap();
        let complex_dd_error =
            |actual: crate::ComplexDoubleDouble, expected: crate::ComplexDoubleDouble| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| complex_dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-20,
            "CUDA DD odd R2C mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context.execute_double_double_c2r(c2r_ir, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-18,
            "CUDA DD odd real round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let r2c_f64 = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let c2r_f64 = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealDoubleDouble(r2c_f64_ir) = &r2c_f64 else {
            panic!("DD/F64 N=15 R2C must build RealDoubleDouble IR");
        };
        let TransformIr::RealDoubleDouble(c2r_f64_ir) = &c2r_f64 else {
            panic!("DD/F64 N=15 C2R must build RealDoubleDouble IR");
        };
        let f64_input = input
            .iter()
            .copied()
            .map(crate::DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_expected = r2c_f64.execute_r2c_reference(&f64_input).unwrap();
        let f64_spectrum = context
            .execute_double_double_r2c_f64_storage(r2c_f64_ir, &f64_input)
            .unwrap();
        let f64_forward_error = f64_spectrum
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 2.0e-12,
            "CUDA DD/F64 odd R2C mismatch on {}: {f64_forward_error:e}",
            context.device_name()
        );
        let f64_restored = context
            .execute_double_double_c2r_f64_storage(c2r_f64_ir, &f64_spectrum)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 2.0e-13,
            "CUDA DD/F64 odd real round trip mismatch on {}: {f64_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_nd_3x4_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![3usize, 4usize];
        let batch_count = 2usize;
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [3,4] plan must use multidimensional DD IR");
        };
        assert_eq!(nd.axes.len(), 2);
        let input = (0..24usize)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.17 * x).sin() + 0.003 * x,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.11 * x).cos() - 0.002 * x,
                        -(index as f64 + 1.0) * 7.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 5.0e-20,
            "CUDA DD [3,4] forward mismatch on {}: {error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-18,
            "CUDA DD [3,4] round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            FftConfig::new(dimensions)
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-12,
            "CUDA DD/F64 [3,4] mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_nd_padding_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![3usize, 4usize];
        let batch_count = 2usize;
        let build = |precision, direction| {
            let config = FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_zero_padding(1, 1, 3)
                .unwrap();
            TransformIr::build(config, direction, context.device_profile()).unwrap()
        };
        let forward = build(Precision::DoubleDouble, Direction::Forward);
        let inverse = build(Precision::DoubleDouble, Direction::Inverse);
        let input = (0..24usize)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.103 * x).sin() + 0.0019 * x,
                        (index + 1) as f64 * 8.0e-32,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.071 * x).cos() - 0.0013 * x,
                        -(index as f64 + 1.0) * 5.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-20,
            "CUDA padded DD [3,4] forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let expected_restored = inverse.execute_double_double_reference(&expected).unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 5.0e-18,
            "CUDA padded DD [3,4] inverse mismatch on {}: {inverse_error:e}",
            context.device_name()
        );

        let f64_forward = build(Precision::DoubleDoubleF64Storage, Direction::Forward);
        let f64_inverse = build(Precision::DoubleDoubleF64Storage, Direction::Inverse);
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 2.0e-12,
            "CUDA padded DD/F64 [3,4] forward mismatch on {}: {f64_forward_error:e}",
            context.device_name()
        );
        let f64_expected_restored = f64_inverse
            .execute_complex_reference(&f64_expected)
            .unwrap();
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_inverse_error = f64_restored
            .iter()
            .zip(&f64_expected_restored)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_inverse_error <= 2.0e-12,
            "CUDA padded DD/F64 [3,4] inverse mismatch on {}: {f64_inverse_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_1024_multi_pass_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 1024usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Stockham(
            stockham,
        )) = &forward
        else {
            panic!("CUDA DD 1024 plan must remain Stockham");
        };
        assert!(stockham.sequence_len > 64);
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00007 * x,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.009 * x).cos() - 0.00003 * x,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 2.0e-17,
            "CUDA DD 1024 multi-pass forward mismatch on {}: {error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-14,
            "CUDA DD 1024 multi-pass round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 3.0e-11 * length as f64,
            "CUDA DD/F64 1024 multi-pass mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_prime_convolutions_above_256_match_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for (length, force_bluestein, expected_convolution) in
            [(401usize, false, 400usize), (257usize, true, 625usize)]
        {
            let mut tuning = crate::PlannerTuning::portable();
            if force_bluestein {
                tuning.max_rader_fft_prime = 100;
            }
            let forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            match &forward {
                TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(
                    rader,
                )) if !force_bluestein => assert_eq!(rader.convolution_len, expected_convolution),
                TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                    bluestein,
                )) if force_bluestein => {
                    assert_eq!(bluestein.convolution_len, expected_convolution)
                }
                other => panic!("unexpected CUDA DD prime algorithm: {other:?}"),
            }
            let input = (0..length)
                .map(|index| {
                    let x = index as f64;
                    crate::ComplexDoubleDouble::new(
                        crate::DoubleDouble::from_parts(
                            (0.013 * x).sin() + 0.00007 * x,
                            (index + 1) as f64 * 2.0e-31,
                        ),
                        crate::DoubleDouble::from_parts(
                            (0.009 * x).cos() - 0.00003 * x,
                            -(index as f64 + 1.0) * 1.0e-31,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let expected = forward.execute_double_double_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let re = (actual.re - expected.re).abs();
                    let im = (actual.im - expected.im).abs();
                    re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                error <= 5.0e-18,
                "CUDA DD prime length {length} mismatch on {}: {error:e}",
                context.device_name()
            );

            let f64_ir = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_tuning(tuning),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(crate::ComplexDoubleDouble::to_complex64)
                .collect::<Vec<_>>();
            let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
                .unwrap();
            let f64_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_error <= 2.0e-10,
                "CUDA DD/F64 prime length {length} mismatch on {}: {f64_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_n32768_wide_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 32 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        let length = 32_768usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N32768 must use recursive upload scheduling");
        };
        assert_eq!(
            ir.stockham_upload_schedule
                .as_ref()
                .expect("CUDA DD N32768 must retain upload metadata")
                .axis_split,
            vec![512, 64]
        );
        assert!(ir.two_upload_four_step_plan.is_some());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(ir)
                .unwrap()
                .passes
                .len(),
            2
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-11,
            "CUDA DD N32768 wide two-upload impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-13,
            "CUDA DD N32768 wide two-upload round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11,
            "CUDA DD/F64 N32768 wide two-upload mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_n4116_non_power_of_two_upload_matches_analytic_or_skip()
    {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 4_116usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N4116 must use recursive upload scheduling");
        };
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("CUDA DD N4116 must retain a Quad upload schedule");
        assert!(matches!(schedule.upload_count, 2 | 3));
        assert_eq!(schedule.axis_split.iter().product::<usize>(), length);
        assert!(
            schedule
                .axis_split
                .iter()
                .all(|component| *component <= 4_096)
        );
        assert!(ir.two_upload_four_step_plan.is_some());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(ir)
                .unwrap()
                .passes
                .len(),
            2
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-11,
            "CUDA DD N4116 upload impulse spectrum mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-13,
            "CUDA DD N4116 fused Four-step round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11,
            "CUDA DD/F64 N4116 fused Four-step mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_n4116_constrained_three_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 672;
        device.shared_memory_pow2_bytes = 672;
        let length = 4_116usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA constrained DD N4116 must use recursive upload scheduling");
        };
        assert_eq!(
            ir.stockham_upload_schedule
                .as_ref()
                .expect("CUDA constrained DD N4116 must retain upload metadata")
                .axis_split,
            vec![14, 21, 14]
        );
        assert!(ir.three_upload_four_step_plan.is_some());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(ir)
                .unwrap()
                .passes
                .len(),
            3
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-11,
            "CUDA DD N4116 constrained three-upload impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-13,
            "CUDA DD N4116 constrained three-upload round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11,
            "CUDA DD/F64 N4116 constrained three-upload mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_n1679616_two_upload_wide_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64
            || actual_device.vendor != crate::GpuVendor::Nvidia
            || actual_device.shared_memory_bytes < 48 * 1024
        {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = device.shared_memory_pow2_bytes.min(32 * 1024);
        let length = 1_679_616usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N1679616 must use recursive upload scheduling");
        };
        assert_eq!(
            ir.stockham_upload_schedule
                .as_ref()
                .expect("CUDA DD N1679616 must retain upload metadata")
                .axis_split,
            vec![1296, 1296]
        );
        assert!(ir.two_upload_four_step_plan.is_some());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(ir)
                .unwrap()
                .passes
                .len(),
            2
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 1.0e-9,
            "CUDA DD N1679616 wide two-upload impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 1.0e-11,
            "CUDA DD N1679616 wide two-upload round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 1.0e-9,
            "CUDA DD/F64 N1679616 wide two-upload mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_n823543_three_upload_wide_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 16 * 1024;
        device.shared_memory_pow2_bytes = 16 * 1024;
        let length = 823_543usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N823543 must use recursive upload scheduling");
        };
        assert_eq!(
            ir.stockham_upload_schedule
                .as_ref()
                .expect("CUDA DD N823543 must retain upload metadata")
                .axis_split,
            vec![343, 49, 49]
        );
        assert!(ir.three_upload_four_step_plan.is_some());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(ir)
                .unwrap()
                .passes
                .len(),
            3
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 5.0e-10,
            "CUDA DD N823543 wide three-upload impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-12,
            "CUDA DD N823543 wide three-upload round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 5.0e-10,
            "CUDA DD/F64 N823543 wide three-upload mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n1922_three_upload_fft_high_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 1024;
        device.shared_memory_pow2_bytes = 1024;
        device.max_threads_per_block = 32;
        let length = 2usize * 31 * 31;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N1922 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N1922 must retain forced-Rader upload metadata")
                .axis_split,
            vec![2, 31, 31]
        );
        assert!(
            ir.forced_rader_three_upload_mapped_components()
                .unwrap()
                .is_some()
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 5.0e-10,
            "CUDA DD N1922 three-upload fft-high impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-12,
            "CUDA DD N1922 three-upload fft-high round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 5.0e-10,
            "CUDA DD/F64 N1922 three-upload fft-high mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n3196_three_upload_direct_middle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 3 * 1024;
        device.shared_memory_pow2_bytes = 3 * 1024;
        device.max_threads_per_block = 128;
        let length = 4usize * 17 * 47;
        let config = || {
            FftConfig::new(vec![length])
                .with_tuning(crate::PlannerTuning::portable())
                .with_precision(Precision::DoubleDouble)
        };
        let forward = TransformIr::build(config(), Direction::Forward, device).unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N3196 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N3196 must retain forced-Rader upload metadata")
                .axis_split,
            vec![4, 47, 17]
        );
        assert!(
            ir.forced_rader_three_upload_mapped_components()
                .unwrap()
                .is_some()
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 5.0e-10,
            "CUDA DD N3196 three-upload direct-low impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            config().with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-12,
            "CUDA DD N3196 three-upload direct-low round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_tuning(crate::PlannerTuning::portable())
                .with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 5.0e-10,
            "CUDA DD/F64 N3196 three-upload direct-low mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_composite_direct_rader_n94_through_n564_type1_threads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for (length, threads_per_transform, batch_count, grouped_batch, shared_bytes) in [
            (2usize * 47, 48usize, 5usize, 3usize, 9_024usize),
            (3usize * 47, 72usize, 5usize, 3usize, 13_536usize),
            (4usize * 47, 96usize, 5usize, 3usize, 18_048usize),
            (5usize * 47, 120usize, 5usize, 3usize, 22_560usize),
            (6usize * 47, 144usize, 5usize, 3usize, 27_072usize),
            (7usize * 47, 168usize, 5usize, 3usize, 31_584usize),
            (8usize * 47, 192usize, 5usize, 3usize, 36_096usize),
            (9usize * 47, 216usize, 5usize, 3usize, 40_608usize),
            (10usize * 47, 240usize, 5usize, 3usize, 45_120usize),
            (12usize * 47, 96usize, 1usize, 1usize, 18_048usize),
        ] {
            let build = |direction| {
                TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_precision(Precision::DoubleDouble)
                        .with_tuning(crate::PlannerTuning::portable())
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    context.device_profile(),
                )
                .unwrap()
            };
            let forward = build(Direction::Forward);
            let inverse = build(Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                    transform
                else {
                    panic!("CUDA DD N{length} composite direct-Rader should remain recursive C2C");
                };
                let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                    root,
                ) = &ir.root
                else {
                    panic!("CUDA DD N{length} composite direct-Rader should keep a Cooley root");
                };
                assert!(matches!(
                    root.right,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(
                        _
                    )
                ));
                let block = root.pack_right.axis_batch_block.unwrap();
                assert_eq!(block.threads_per_transform, threads_per_transform);
                assert_eq!(block.grouped_batch, grouped_batch);
                assert_eq!(
                    [block.local_size_x, block.local_size_y],
                    [threads_per_transform, grouped_batch]
                );
                assert_eq!(
                    root.pack_right.dispatch.x as usize,
                    batch_count.div_ceil(grouped_batch)
                );
                assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
                assert_eq!(root.scatter_output.axis_batch_block, Some(block));
                let fused = root
                    .fused_small_direct_rader_stockham()
                    .unwrap()
                    .unwrap_or_else(|| {
                        panic!("CUDA DD N{length} should use the fused Direct-Rader component")
                    });
                assert_eq!(fused.required_shared_memory_bytes().unwrap(), shared_bytes);
                let program = crate::ProgramIr::double_double_recursive(ir).unwrap();
                assert_eq!(program.passes.len(), 1);
                assert_eq!(program.passes[0].name, fused.name());
                let source = NativeSourceBackend::new(Backend::Cuda)
                    .lower_transform(transform)
                    .unwrap();
                let reports =
                    crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                        &*context, &source,
                    )
                    .unwrap();
                assert_eq!(reports.len(), 1);
                let NativeCompiledResourceMetrics::Cuda {
                    registers_per_thread,
                    static_shared_memory_bytes_per_block,
                    local_memory_bytes_per_thread,
                    max_threads_per_block: _,
                } = reports[0].metrics
                else {
                    panic!(
                        "CUDA DD N{length} resource report returned the wrong backend metric variant"
                    );
                };
                assert!(registers_per_thread > 0);
                assert!(static_shared_memory_bytes_per_block >= shared_bytes);
                eprintln!(
                    "CUDA DD N{length} fused resources: regs/thread={registers_per_thread}, shared/block={static_shared_memory_bytes_per_block}, local/thread={local_memory_bytes_per_thread}"
                );
            }

            let mut input = vec![crate::ComplexDoubleDouble::default(); length * batch_count];
            for batch in 0..batch_count {
                input[batch * length + 1] = crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_f64(1.0),
                    crate::DoubleDouble::default(),
                );
            }
            let expected = forward.execute_double_double_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let dd_error = |actual: crate::ComplexDoubleDouble,
                            expected: crate::ComplexDoubleDouble| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            };
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 5.0e-18,
                "CUDA DD N{length} composite direct-Rader mismatch on {}: {forward_error:e}",
                context.device_name()
            );
            let restored = context
                .execute_transform_double_double(&inverse, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 5.0e-17,
                "CUDA DD N{length} composite direct-Rader round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_composite_direct_rader_n94_type1_threads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 2usize * 47;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA N94 composite direct-Rader should remain recursive C2C");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
            panic!("CUDA N94 composite direct-Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 48);
        assert_eq!(block.grouped_batch, grouped_batch);
        assert_eq!([block.local_size_x, block.local_size_y], [3, 48]);
        assert_eq!(root.pack_right.dispatch.x, 2);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 5.0e-4,
            "CUDA N94 type-1 composite direct-Rader forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = build(Direction::Inverse);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 5.0e-4,
            "CUDA N94 type-1 composite direct-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_fft_rader_cooley_left_n51_fusion_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        let length = 3usize * 17;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive = crate::RecursiveFftIr::build(&plan, direction, profile).unwrap();
            TransformIr::Complex1d(crate::OneDimFftIr::Recursive(Box::new(recursive)))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N51 FFT-Rader Cooley fusion probe should remain recursive");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("CUDA N51 should keep a 3 x p17 Cooley root");
            };
            assert!(matches!(
                root.right,
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 17
            ));
            assert!(root.fused_fft_rader_left_stockham().unwrap().is_some());
            assert!(root.fused_small_fft_rader_stockham().unwrap().is_some());
            assert_eq!(crate::ProgramIr::recursive_fft(ir).unwrap().passes.len(), 1);
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for k in [0usize, 1, 7, 17, length / 2, length - 1] {
            let angle = -std::f32::consts::TAU * k as f32 / length as f32;
            let actual = spectrum[k];
            assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 1.0e-3);
        }
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(error <= 1.0e-3);
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_n51_fusion_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let profile = context.device_profile();
        let length = 3usize * 17;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive =
                crate::DoubleDoubleRecursiveFftIr::build_for_device(&plan, direction, profile)
                    .unwrap();
            TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N51 FFT-Rader fusion probe should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N51 should keep a 3 x p17 Cooley root");
            };
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                    ref rader
                ) if rader.prime == 17
            ));
            let fused = root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .expect("CUDA DD N51 should use the fused FFT-Rader component");
            assert_eq!(fused.required_shared_memory_bytes().unwrap(), 3_168);
            assert_eq!(
                crate::ProgramIr::double_double_recursive(ir)
                    .unwrap()
                    .passes
                    .len(),
                1
            );
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            assert_eq!(source.program.passes.len(), 1);
            assert_eq!(source.shaders[0].required_shared_memory_bytes, 3_168);
            let reports =
                crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                    &*context, &source,
                )
                .unwrap();
            assert_eq!(reports.len(), 1);
            let NativeCompiledResourceMetrics::Cuda {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                ..
            } = reports[0].metrics
            else {
                panic!("CUDA DD N51 resource report returned the wrong metric variant");
            };
            eprintln!(
                "CUDA DD N51 fused FFT-Rader resources: regs/thread={registers_per_thread}, shared/block={static_shared_memory_bytes_per_block}, local/thread={local_memory_bytes_per_thread}"
            );
            assert_eq!(static_shared_memory_bytes_per_block, 3_168);
        }

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD N51 fused FFT-Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-17,
            "CUDA DD N51 fused FFT-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_n68_radix4_fusion_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let profile = context.device_profile();
        let length = 4usize * 17;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive =
                crate::DoubleDoubleRecursiveFftIr::build_for_device(&plan, direction, profile)
                    .unwrap();
            TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N68 radix-4 FFT-Rader fusion probe should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N68 should keep a 4 x p17 Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (4, 17));
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                    ref rader
                ) if rader.prime == 17
            ));
            let fused = root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .expect("CUDA DD N68 should use the fused radix-4 FFT-Rader component");
            assert_eq!(fused.required_shared_memory_bytes().unwrap(), 4_224);
            assert_eq!(
                crate::ProgramIr::double_double_recursive(ir)
                    .unwrap()
                    .passes
                    .len(),
                1
            );
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            assert_eq!(source.program.passes.len(), 1);
            assert_eq!(source.shaders[0].required_shared_memory_bytes, 4_224);
            let reports =
                crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                    &*context, &source,
                )
                .unwrap();
            assert_eq!(reports.len(), 1);
            let NativeCompiledResourceMetrics::Cuda {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                ..
            } = reports[0].metrics
            else {
                panic!("CUDA DD N68 resource report returned the wrong metric variant");
            };
            eprintln!(
                "CUDA DD N68 fused radix-4 FFT-Rader resources: regs/thread={registers_per_thread}, shared/block={static_shared_memory_bytes_per_block}, local/thread={local_memory_bytes_per_thread}"
            );
            assert_eq!(static_shared_memory_bytes_per_block, 4_224);
        }

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD N68 fused radix-4 FFT-Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-17,
            "CUDA DD N68 fused radix-4 FFT-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_n85_radix5_fusion_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let profile = context.device_profile();
        let length = 5usize * 17;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive =
                crate::DoubleDoubleRecursiveFftIr::build_for_device(&plan, direction, profile)
                    .unwrap();
            TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N85 radix-5 FFT-Rader fusion probe should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N85 should keep a 5 x p17 Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (5, 17));
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                    ref rader
                ) if rader.prime == 17
            ));
            let fused = root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .expect("CUDA DD N85 should use the fused radix-5 FFT-Rader component");
            assert_eq!(fused.required_shared_memory_bytes().unwrap(), 5_280);
            assert_eq!(
                crate::ProgramIr::double_double_recursive(ir)
                    .unwrap()
                    .passes
                    .len(),
                1
            );
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            assert_eq!(source.program.passes.len(), 1);
            assert_eq!(source.shaders[0].required_shared_memory_bytes, 5_280);
            let reports =
                crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                    &*context, &source,
                )
                .unwrap();
            assert_eq!(reports.len(), 1);
            let NativeCompiledResourceMetrics::Cuda {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                ..
            } = reports[0].metrics
            else {
                panic!("CUDA DD N85 resource report returned the wrong metric variant");
            };
            eprintln!(
                "CUDA DD N85 fused radix-5 FFT-Rader resources: regs/thread={registers_per_thread}, shared/block={static_shared_memory_bytes_per_block}, local/thread={local_memory_bytes_per_thread}"
            );
            assert_eq!(static_shared_memory_bytes_per_block, 5_280);
        }

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD N85 fused radix-5 FFT-Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-17,
            "CUDA DD N85 fused radix-5 FFT-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    fn run_cuda_dd_fft_rader_composite_fusion_case(
        left_len: usize,
        right_prime: usize,
        expected_shared: usize,
    ) {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let profile = context.device_profile();
        let length = left_len * right_prime;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive =
                crate::DoubleDoubleRecursiveFftIr::build_for_device(&plan, direction, profile)
                    .unwrap();
            TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N{length} FFT-Rader fusion probe should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N{length} should keep a {left_len} x p{right_prime} Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (left_len, right_prime));
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                    ref rader
                ) if rader.prime == right_prime
            ));
            let actual_shared = if matches!(
                root.left,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(_)
            ) {
                root.fused_dual_fft_rader_supported()
                    .unwrap()
                    .expect("CUDA DD dual FFT-Rader root should use an explicitly supported fused component")
                    .required_shared_memory_bytes()
                    .unwrap()
            } else {
                root.fused_small_fft_rader_stockham()
                    .unwrap()
                    .expect("CUDA DD p17 composite should use the fused FFT-Rader component")
                    .required_shared_memory_bytes()
                    .unwrap()
            };
            assert_eq!(actual_shared, expected_shared);
            assert_eq!(
                crate::ProgramIr::double_double_recursive(ir)
                    .unwrap()
                    .passes
                    .len(),
                1
            );
            let source = NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(transform)
                .unwrap();
            assert_eq!(source.program.passes.len(), 1);
            assert_eq!(
                source.shaders[0].required_shared_memory_bytes,
                expected_shared
            );
            let reports =
                crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                    &*context, &source,
                )
                .unwrap();
            assert_eq!(reports.len(), 1);
            let NativeCompiledResourceMetrics::Cuda {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                ..
            } = reports[0].metrics
            else {
                panic!("CUDA DD N{length} resource report returned the wrong metric variant");
            };
            eprintln!(
                "CUDA DD N{length} fused {left_len}xp{right_prime} FFT-Rader resources: regs/thread={registers_per_thread}, shared/block={static_shared_memory_bytes_per_block}, local/thread={local_memory_bytes_per_thread}"
            );
            assert_eq!(static_shared_memory_bytes_per_block, expected_shared);
        }

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD N{length} fused {left_len}xp{right_prime} FFT-Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-17,
            "CUDA DD N{length} fused {left_len}xp{right_prime} FFT-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_n102_radix6_fusion_or_skip() {
        run_cuda_dd_fft_rader_composite_fusion_case(6, 17, 6_336);
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_p17_radix7_through16_fusion_or_skip() {
        for radix in 7usize..=16 {
            run_cuda_dd_fft_rader_composite_fusion_case(radix, 17, 1_056 * radix);
        }
    }

    #[test]
    fn cuda_dd_fft_rader_n289_dual_rader_fusion_or_skip() {
        run_cuda_dd_fft_rader_composite_fusion_case(17, 17, 19_040);
    }

    #[test]
    fn cuda_dd_fft_rader_n323_dual_rader_fusion_or_skip() {
        run_cuda_dd_fft_rader_composite_fusion_case(17, 19, 21_344);
    }

    #[test]
    fn cuda_dd_fft_rader_n493_dual_rader_fusion_or_skip() {
        run_cuda_dd_fft_rader_composite_fusion_case(17, 29, 32_864);
    }

    #[test]
    fn cuda_dd_fft_rader_n527_dual_rader_fusion_or_skip() {
        run_cuda_dd_fft_rader_composite_fusion_case(17, 31, 35_168);
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_n306_multistage_fusion_or_skip() {
        run_cuda_dd_fft_rader_composite_fusion_case(18, 17, 28_800);
    }

    #[test]
    fn cuda_dd_fft_rader_cooley_left_n340_through_n510_multistage_fusion_or_skip() {
        for (radix, expected_shared) in [
            (20usize, 32_000usize),
            (21, 33_600),
            (22, 35_200),
            (24, 38_400),
            (25, 40_000),
            (26, 41_600),
            (27, 43_200),
            (28, 44_800),
            (30, 48_000),
        ] {
            run_cuda_dd_fft_rader_composite_fusion_case(radix, 17, expected_shared);
        }
    }

    #[test]
    fn cuda_n4089_direct_and_fft_rader_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut profile = context.device_profile();
        if profile.max_threads_per_block < 128 || profile.max_workgroup_size[0] < 128 {
            return;
        }
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        let length = 3usize * 29 * 47;
        for fft_rader_low in [false, true] {
            let build = |direction| {
                let mut tuning = crate::PlannerTuning::portable();
                if !fft_rader_low {
                    tuning.min_rader_fft_prime = 53;
                }
                tuning.validate().unwrap();
                TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_tuning(tuning)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    profile,
                )
                .unwrap()
            };
            let forward = build(Direction::Forward);
            let inverse = build(Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                    panic!("CUDA N4089 normal-capacity probe should remain recursive");
                };
                assert!(!ir.rader_forced_two_upload);
                let schedule = ir.rader_forced_upload_schedule.as_ref().expect(
                    "CUDA N4089 normal capacity must materialize Rader multi-upload metadata",
                );
                assert_eq!(schedule.axis_split, vec![87, 47]);
                assert_eq!(schedule.upload_count, 2);
                assert!(ir.four_step_plan.is_some());
                let uploads = ir
                    .four_step_rader_upload_nodes()
                    .unwrap()
                    .expect("CUDA N4089 normal capacity must materialize two Four-step uploads");
                assert_eq!(uploads.len(), 2);
                assert!(matches!(
                    uploads[0],
                    crate::recursive_ir::RecursiveFftNodeIr::DirectRader(ref rader) if rader.prime == 47
                ));
                let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(low) = &uploads[1] else {
                    panic!("CUDA N4089 low upload should remain N87 Cooley");
                };
                if fft_rader_low {
                    assert!(matches!(
                        low.right,
                        crate::recursive_ir::RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 29
                    ));
                    let fused = low
                        .fused_small_fft_rader_stockham()
                        .unwrap()
                        .expect("CUDA N4089 p29 upload should stay monolithic");
                    assert!(!fused.single_shared_register_convolution().unwrap());
                    assert_eq!(fused.shared_stripe_count().unwrap(), 2);
                    assert_eq!(fused.uniform_barrier_count().unwrap(), 7);
                    assert!(
                        fused.required_shared_memory_bytes().unwrap()
                            <= profile.shared_memory_bytes
                    );
                } else {
                    assert!(matches!(
                        low.right,
                        crate::recursive_ir::RecursiveFftNodeIr::DirectRader(ref rader) if rader.prime == 29
                    ));
                    assert!(low.fused_small_direct_rader_stockham().unwrap().is_some());
                }
                assert_eq!(crate::ProgramIr::recursive_fft(ir).unwrap().passes.len(), 2);
            }
            let mut input = vec![Complex32::new(0.0, 0.0); length];
            input[1] = Complex32::new(1.0, 0.0);
            let spectrum = context
                .execute_transform_complex32(&forward, &input)
                .unwrap();
            for k in [0usize, 1, 29, 47, 257, length / 2, length - 1] {
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                let actual = spectrum[k];
                assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 2.0e-3);
            }
            let restored = context
                .execute_transform_complex32(&inverse, &spectrum)
                .unwrap();
            let error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0f32, f32::max);
            assert!(error <= 2.0e-3);
        }
    }

    #[test]
    fn cuda_forced_rader_pass_local_fft_parent_n5100_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut profile = context.device_profile();
        if profile.max_threads_per_block < 128 || profile.max_workgroup_size[0] < 128 {
            return;
        }
        profile.max_threads_per_block = 128;
        profile.max_workgroup_size[0] = 128;
        let length = 17usize * 300;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N5100 forced-Rader probe should remain recursive");
            };
            assert_eq!(
                ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
                vec![68, 75]
            );
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(low) = &uploads[1] else {
                panic!("CUDA N5100 upload0 should remain 4 x p17 Cooley");
            };
            let block = low.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 5);
            assert_eq!(low.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(low.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for k in [0usize, 1, 17, 75, 257, length / 2, length - 1] {
            let angle = -std::f32::consts::TAU * k as f32 / length as f32;
            let actual = spectrum[k];
            assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 2.0e-3);
        }
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(error <= 2.0e-3);
    }

    #[test]
    fn cuda_forced_rader_upload0_global_scale_n69632_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut profile = context.device_profile();
        if profile.max_threads_per_block < 64 || profile.max_workgroup_size[0] < 64 {
            return;
        }
        profile.max_threads_per_block = 64;
        profile.max_workgroup_size[0] = 64;
        let length = (16usize * 17) * 256;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N69632 forced-Rader probe should remain recursive");
            };
            assert_eq!(
                ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
                vec![272, 256]
            );
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            assert!(matches!(
                uploads[0],
                crate::recursive_ir::RecursiveFftNodeIr::Stockham(_)
            ));
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(low) = &uploads[1] else {
                panic!("CUDA N69632 upload0 should remain 16 x p17 Cooley");
            };
            let block = low.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 17);
            assert!(block.grouped_batch > 1);
            assert!(block.grouped_batch * block.threads_per_transform <= 64);
            assert!(block.transforms_on_x);
            assert!(block.axis_swapped);
            assert_eq!(block.local_size_x, block.grouped_batch);
            assert_eq!(block.local_size_y, 17);
            assert_eq!(low.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(low.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for k in [0usize, 1, 17, 272, 1024, length / 2, length - 1] {
            let angle = -std::f32::consts::TAU * k as f32 / length as f32;
            let actual = spectrum[k];
            assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 5.0e-3);
        }
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(error <= 5.0e-3);
    }

    #[test]
    fn cuda_multi_fft_rader_n551_type0_threads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 19usize * 29;
        let batch_count = 2usize;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = transform else {
                panic!("CUDA N551 multi FFT-Rader should remain recursive C2C");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
                panic!("CUDA N551 should keep a Cooley root");
            };
            assert!(matches!(
                root.left,
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(_)
            ));
            assert!(matches!(
                root.right,
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 92);
            assert_eq!([block.local_size_x, block.local_size_y], [92, 1]);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(forward_error <= 1.5e-3);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(round_trip_error <= 1.5e-3);
    }

    #[test]
    fn cuda_pure_multi_fft_rader_n8303_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        let length = 19usize * 19 * 23;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = transform else {
                panic!("CUDA N8303 repeated FFT-Rader should remain recursive C2C");
            };
            assert!(!recursive.rader_forced_two_upload);
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA N8303 normal capacity must materialize Rader multi-upload metadata");
            assert_eq!(schedule.axis_split, vec![361, 23]);
            assert_eq!(schedule.upload_count, 2);
            assert!(recursive.four_step_plan.is_some());
            let uploads = recursive
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("CUDA N8303 normal capacity must materialize two Four-step uploads");
            assert_eq!(uploads.len(), 2);
            assert!(matches!(
                uploads[0],
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 23
            ));
            assert!(matches!(
                uploads[1],
                crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(_)
            ));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(forward_error <= 5.0e-3);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(round_trip_error <= 5.0e-3);
    }

    #[test]
    fn cuda_mixed_direct_multi_fft_rader_n7429_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        let length = 17usize * 19 * 23;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 17;
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = transform else {
                panic!("CUDA N7429 mixed multi-Rader should remain recursive C2C");
            };
            assert!(!recursive.rader_forced_two_upload);
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA N7429 normal capacity must materialize Rader multi-upload metadata");
            assert_eq!(schedule.axis_split, vec![323, 23]);
            assert_eq!(schedule.upload_count, 2);
            assert!(recursive.four_step_plan.is_some());
            let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
            assert_eq!(uploads.len(), 2);
            assert_eq!(uploads[0].logical_len(), 23);
            assert_eq!(uploads[1].logical_len(), 323);
            assert!(matches!(
                uploads[0],
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 23
            ));
            assert!(matches!(
                uploads[1],
                crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(_)
            ));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 4.0e-3,
            "CUDA N7429 mixed multi-Rader forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 4.0e-3,
            "CUDA N7429 mixed multi-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_mixed_direct_multi_fft_rader_n9367_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != crate::GpuVendor::Nvidia
            || profile.max_threads_per_block < 63
            || profile.max_workgroup_size[0] < 63
        {
            return;
        }
        let length = 17usize * 19 * 29;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 17;
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = transform else {
                panic!("CUDA N9367 mixed multi-Rader should remain recursive C2C");
            };
            assert!(!recursive.rader_forced_two_upload);
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA N9367 normal capacity must materialize Rader multi-upload metadata");
            assert_eq!(schedule.axis_split, vec![323, 29]);
            assert_eq!(schedule.upload_count, 2);
            assert!(recursive.four_step_plan.is_some());
            let uploads = recursive
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("CUDA N9367 normal capacity must materialize two Four-step uploads");
            assert_eq!(uploads.len(), 2);
            assert!(matches!(
                uploads[0],
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 29
            ));
            assert!(matches!(
                uploads[1],
                crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(_)
            ));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 5.0e-3,
            "CUDA N9367 mixed multi-Rader forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 5.0e-3,
            "CUDA N9367 mixed multi-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_mixed_direct_fft_rader_n527_type1_threads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 17usize * 31;
        let batch_count = 2usize;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = transform else {
                panic!("CUDA N527 mixed Rader should remain recursive C2C");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
                panic!("CUDA N527 mixed Rader should keep a Cooley root");
            };
            assert!(matches!(
                root.left,
                crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
            ));
            assert!(matches!(
                root.right,
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 108);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [108, 1]);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 1.2e-3,
            "CUDA N527 mixed Rader forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 1.2e-3,
            "CUDA N527 mixed Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        if !context.device_profile().supports_f64 {
            return;
        }
        let build_dd = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let dd_forward = build_dd(Direction::Forward);
        let dd_inverse = build_dd(Direction::Inverse);
        for transform in [&dd_forward, &dd_inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N527 mixed Rader should remain recursive C2C");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N527 mixed Rader should keep a Cooley root");
            };
            assert!(matches!(
                root.left,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 108);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [108, 1]);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        }
        let mut dd_input = vec![crate::ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            dd_input[batch * length + 1] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(1.0),
                crate::DoubleDouble::default(),
            );
        }
        let dd_expected = dd_forward
            .execute_double_double_reference(&dd_input)
            .unwrap();
        let dd_actual = context
            .execute_transform_double_double(&dd_forward, &dd_input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let dd_forward_error = dd_actual
            .iter()
            .copied()
            .zip(dd_expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            dd_forward_error <= 2.0e-17,
            "CUDA DD N527 mixed Rader mismatch on {}: {dd_forward_error:e}",
            context.device_name()
        );
        let dd_restored = context
            .execute_transform_double_double(&dd_inverse, &dd_actual)
            .unwrap();
        let dd_round_trip_error = dd_restored
            .iter()
            .copied()
            .zip(dd_input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            dd_round_trip_error <= 2.0e-16,
            "CUDA DD N527 mixed Rader round trip mismatch on {}: {dd_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_mixed_rader_n7905_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        let length = 15usize * 17 * 31;
        let tuning = || {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            tuning
        };
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_tuning(tuning())
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };

        let forward = build(Precision::F32, Direction::Forward);
        let inverse = build(Precision::F32, Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N7905 mixed Rader should remain recursive");
            };
            assert!(!ir.rader_forced_two_upload);
            let schedule = ir
                .rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA N7905 normal capacity must materialize Rader multi-upload metadata");
            assert_eq!(schedule.axis_split, vec![93, 85]);
            assert_eq!(schedule.upload_count, 2);
            assert!(ir.four_step_plan.is_some());
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            assert_eq!(uploads.len(), 2);
            assert_eq!(uploads[0].logical_len(), 85);
            assert_eq!(uploads[1].logical_len(), 93);
            assert!(matches!(
                uploads[0],
                crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(_)
            ));
            assert!(matches!(
                uploads[1],
                crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(_)
            ));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 4.0e-3,
            "CUDA N7905 ordinary mixed Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 4.0e-3,
            "CUDA N7905 ordinary mixed Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        if !profile.supports_f64 {
            return;
        }
        let dd_forward = build(Precision::DoubleDouble, Direction::Forward);
        let dd_inverse = build(Precision::DoubleDouble, Direction::Inverse);
        for transform in [&dd_forward, &dd_inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N7905 mixed Rader should remain recursive");
            };
            let schedule = ir.rader_forced_upload_schedule.as_ref().expect(
                "CUDA DD N7905 normal capacity must materialize Rader multi-upload metadata",
            );
            assert_eq!(schedule.axis_split, vec![93, 85]);
            assert_eq!(schedule.upload_count, 2);
            let mapped_high = ir
                .forced_rader_two_upload_mapped_high_component()
                .unwrap()
                .unwrap();
            let mapped_low = ir
                .forced_rader_two_upload_mapped_low_component()
                .unwrap()
                .unwrap();
            assert_eq!(mapped_high.logical_len(), 85);
            assert_eq!(mapped_low.logical_len(), 93);
            assert!(matches!(
                mapped_high,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_)
            ));
            assert!(matches!(
                mapped_low,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_)
            ));
        }
        let mut dd_input = vec![crate::ComplexDoubleDouble::default(); length];
        dd_input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let dd_expected = dd_forward
            .execute_double_double_reference(&dd_input)
            .unwrap();
        let dd_actual = context
            .execute_transform_double_double(&dd_forward, &dd_input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let dd_forward_error = dd_actual
            .iter()
            .copied()
            .zip(dd_expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            dd_forward_error <= 5.0e-17,
            "CUDA DD N7905 mixed Rader mismatch on {}: {dd_forward_error:e}",
            context.device_name()
        );
        let dd_restored = context
            .execute_transform_double_double(&dd_inverse, &dd_actual)
            .unwrap();
        let dd_round_trip_error = dd_restored
            .iter()
            .copied()
            .zip(dd_input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            dd_round_trip_error <= 5.0e-16,
            "CUDA DD N7905 mixed Rader round trip mismatch on {}: {dd_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_mixed_multi_fft_rader_n9367_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64
            || profile.vendor != crate::GpuVendor::Nvidia
            || profile.shared_memory_bytes < 48 * 1024
            || profile.shared_memory_pow2_bytes < 32 * 1024
        {
            return;
        }
        let mut device = profile;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        let length = 17usize * 19 * 29;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 17;
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive =
                crate::DoubleDoubleRecursiveFftIr::build_for_device(&plan, direction, device)
                    .unwrap();
            TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N9367 mixed multi-Rader should remain recursive");
            };
            let schedule = ir.rader_forced_upload_schedule.as_ref().expect(
                "CUDA DD N9367 normal capacity must materialize Rader multi-upload metadata",
            );
            assert_eq!(schedule.axis_split, vec![323, 29]);
            assert_eq!(schedule.upload_count, 2);
            assert!(matches!(
                ir.forced_rader_two_upload_mapped_high_component().unwrap(),
                Some(crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(ref rader)) if rader.prime == 29
            ));
            assert!(matches!(
                ir.forced_rader_two_upload_mapped_low_component().unwrap(),
                Some(
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                        _
                    )
                )
            ));
        }
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0f64, f64::max);
        assert!(forward_error <= 3.0e-12);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0f64, f64::max);
        assert!(round_trip_error <= 8.0e-16);
    }

    #[test]
    fn cuda_double_double_mixed_multi_fft_rader_n9367_forced256_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64
            || profile.vendor != crate::GpuVendor::Nvidia
            || profile.shared_memory_bytes < 48 * 1024
            || profile.shared_memory_pow2_bytes < 32 * 1024
            || profile.max_threads_per_block < 256
            || profile.max_workgroup_size[0] < 162
            || profile.max_workgroup_size[1] < 5
        {
            return;
        }
        let mut device = profile;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 256;
        let length = 17usize * 19 * 29;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 17;
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let recursive =
                crate::DoubleDoubleRecursiveFftIr::build_for_device(&plan, direction, device)
                    .unwrap();
            TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive))
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N9367 forced256 should remain recursive");
            };
            assert_eq!(
                ir.rader_forced_upload_schedule
                    .as_ref()
                    .expect("CUDA DD N9367 forced256 must use forced Rader uploads")
                    .axis_split,
                vec![323, 29]
            );
            let mapped_high = ir
                .forced_rader_two_upload_mapped_high_component()
                .unwrap()
                .unwrap();
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                high_rader,
            ) = mapped_high
            else {
                panic!("CUDA DD N9367 forced256 upload1 should remain p29 FFT-Rader");
            };
            assert_eq!(high_rader.prime, 29);
            let high_block = high_rader.caller_axis_batch_block.unwrap();
            assert_eq!(high_block.threads_per_transform, 5);
            assert_eq!(high_block.grouped_batch, 24);
            assert_eq!([high_block.local_size_x, high_block.local_size_y], [24, 5]);
            let mapped_low = ir
                .forced_rader_two_upload_mapped_low_component()
                .unwrap()
                .unwrap();
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) =
                mapped_low
            else {
                panic!("CUDA DD N9367 forced256 upload0 should remain p17+p19 Cooley");
            };
            let block = low.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 162);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [162, 1]);
            assert_eq!(low.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(low.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0f64, f64::max);
        assert!(forward_error <= 3.0e-12);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0f64, f64::max);
        assert!(round_trip_error <= 8.0e-16);
    }

    #[test]
    fn cuda_double_double_direct_rader_n1012_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut profile = context.device_profile();
        if !profile.supports_f64 {
            return;
        }
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        let length = 4usize * 11 * 23;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N1012 direct-Rader reserve probe should remain recursive");
            };
            let schedule = ir.rader_forced_upload_schedule.as_ref().expect(
                "CUDA DD N1012 normal capacity must materialize Rader multi-upload metadata",
            );
            assert_eq!(schedule.axis_split, vec![44, 23]);
            assert_eq!(schedule.upload_count, 2);
            assert!(ir.two_upload_four_step_plan.is_none());
            assert!(ir.three_upload_four_step_plan.is_none());
            assert!(matches!(
                ir.forced_rader_two_upload_mapped_high_component().unwrap(),
                Some(crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(ref rader)) if rader.prime == 23
            ));
            assert!(
                ir.forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .is_some()
            );
        }
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0f64, f64::max);
        assert!(forward_error <= 3.0e-12);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0f64, f64::max);
        assert!(round_trip_error <= 8.0e-16);
    }

    #[test]
    fn cuda_double_double_repeated_fft_rader_n8303_normal_capacity_two_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64 {
            return;
        }
        let length = 19usize * 19 * 23;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N8303 repeated FFT-Rader should remain recursive");
            };
            let schedule = ir.rader_forced_upload_schedule.as_ref().expect(
                "CUDA DD N8303 normal capacity must materialize Rader multi-upload metadata",
            );
            assert_eq!(schedule.axis_split, vec![361, 23]);
            assert_eq!(schedule.upload_count, 2);
            assert!(matches!(
                ir.forced_rader_two_upload_mapped_high_component().unwrap(),
                Some(crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(ref rader)) if rader.prime == 23
            ));
            assert!(matches!(
                ir.forced_rader_two_upload_mapped_low_component().unwrap(),
                Some(
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                        _
                    )
                )
            ));
        }
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(k, actual)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0f64, f64::max);
        assert!(forward_error <= 3.0e-12);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0f64, f64::max);
        assert!(round_trip_error <= 8.0e-16);
    }

    #[test]
    fn cuda_repeated_direct_rader_n289_multiplier_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 17usize * 17;
        let batch_count = 2usize;
        let tuning = || {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 29;
            tuning.validate().unwrap();
            tuning
        };

        let build_f32 = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_tuning(tuning())
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build_f32(Direction::Forward);
        let inverse = build_f32(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = transform else {
                panic!("CUDA N289 repeated direct-Rader should remain recursive C2C");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
                panic!("CUDA N289 repeated direct-Rader should keep a Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (17, 17));
            assert!(matches!(
                root.left,
                crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
            ));
            assert!(matches!(
                root.right,
                crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 153);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [153, 1]);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 8.0e-4,
            "CUDA N289 repeated direct-Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 8.0e-4,
            "CUDA N289 repeated direct-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        if !context.device_profile().supports_f64 {
            return;
        }
        let build_dd = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning())
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let dd_forward = build_dd(Direction::Forward);
        let dd_inverse = build_dd(Direction::Inverse);
        for transform in [&dd_forward, &dd_inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N289 repeated direct-Rader should remain recursive C2C");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N289 repeated direct-Rader should keep a Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (17, 17));
            assert!(matches!(
                root.left,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 153);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [153, 1]);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        }
        let mut dd_input = vec![crate::ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            dd_input[batch * length + 1] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(1.0),
                crate::DoubleDouble::default(),
            );
        }
        let dd_expected = dd_forward
            .execute_double_double_reference(&dd_input)
            .unwrap();
        let dd_actual = context
            .execute_transform_double_double(&dd_forward, &dd_input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let dd_forward_error = dd_actual
            .iter()
            .copied()
            .zip(dd_expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            dd_forward_error <= 1.0e-17,
            "CUDA DD N289 repeated direct-Rader mismatch on {}: {dd_forward_error:e}",
            context.device_name()
        );
        let dd_restored = context
            .execute_transform_double_double(&dd_inverse, &dd_actual)
            .unwrap();
        let dd_round_trip_error = dd_restored
            .iter()
            .copied()
            .zip(dd_input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            dd_round_trip_error <= 1.0e-16,
            "CUDA DD N289 repeated direct-Rader round trip mismatch on {}: {dd_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_multi_direct_rader_n2491_type1_threads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 47usize * 53;
        let batch_count = 2usize;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 89;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        let expected_threads =
            crate::scheduler::plan_gpu_double_double_axis0_composite_direct_rader_threads_for_primes(
                length,
                &[47, 53],
                batch_count,
                context.device_profile(),
            )
            .unwrap()
            .expect("CUDA DD N2491 type-1 coupling should produce a physical thread count");
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N2491 all-direct Rader should remain recursive C2C");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N2491 all-direct Rader should keep a Cooley root");
            };
            assert!(matches!(
                root.left,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            assert!(matches!(
                root.right,
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [expected_threads, 1]
            );
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        }

        let mut input = vec![crate::ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(1.0),
                crate::DoubleDouble::default(),
            );
        }
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-17,
            "CUDA DD N2491 multi-direct type-1 mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-16,
            "CUDA DD N2491 multi-direct type-1 round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_multi_direct_rader_n2491_type1_threads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 47usize * 53;
        let batch_count = 2usize;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 89;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA N2491 all-direct Rader should remain recursive C2C");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
            panic!("CUDA N2491 all-direct Rader should keep a Cooley root");
        };
        assert!(matches!(
            root.left,
            crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
        ));
        assert!(matches!(
            root.right,
            crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
        ));
        let expected_threads =
            crate::scheduler::plan_gpu_axis0_composite_direct_rader_threads_for_primes(
                length,
                &[47, 53],
                batch_count,
                context.device_profile(),
            )
            .unwrap()
            .expect("CUDA N2491 type-1 coupling should produce a physical thread count");
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, expected_threads);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!(
            [block.local_size_x, block.local_size_y],
            [expected_threads, 1]
        );
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 2.0e-3,
            "CUDA N2491 multi-direct type-1 forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = build(Direction::Inverse);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            round_trip_error <= 2.0e-3,
            "CUDA N2491 multi-direct type-1 round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n102272_three_upload_direct_middle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        let length = 17usize * 47 * 128;
        let config = || {
            FftConfig::new(vec![length])
                .with_tuning(crate::PlannerTuning::portable())
                .with_precision(Precision::DoubleDouble)
        };
        let forward = TransformIr::build(config(), Direction::Forward, device).unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N102272 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N102272 must retain forced-Rader upload metadata")
                .axis_split,
            vec![64, 47, 34]
        );
        let components = ir
            .forced_rader_three_upload_mapped_components()
            .unwrap()
            .expect("CUDA DD N102272 must materialize mapped three-upload components");
        let middle = match &components[1] {
            crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 1,
                ir: crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(direct),
            } if direct.prime == 47
                && matches!(direct.io_mapping, crate::StockhamIoMapping::FourStepThreeUpload1(_)) => direct,
            _ => panic!("CUDA DD N102272 upload1 must be mapped p47 direct-Rader"),
        };
        let crate::StockhamIoMapping::FourStepThreeUpload1(mapping) = &middle.io_mapping else {
            unreachable!();
        };
        let [a, b, _c] = mapping.axis_split;
        let ab = a * b;
        let tile_mapping = crate::ThreeUploadFourStepMapping {
            logical_len: ab * 2,
            axis_split: [a, b, 2],
            outer_batch_count: 1,
        };
        let mut middle_tile = middle.clone();
        middle_tile.batch_count = a * 2;
        middle_tile.io_mapping = crate::StockhamIoMapping::FourStepThreeUpload1(tile_mapping);
        middle_tile.validate().unwrap();
        let n1 = 7usize;
        let batch = n1;
        let mut component_input = vec![crate::ComplexDoubleDouble::default(); ab * 2];
        component_input[batch * b + 1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let source = NativeSourceBackend::new(Backend::Cuda)
            .lower_double_double_direct_rader(&middle_tile)
            .unwrap();
        let actual = context
            .execute_program_double_double(&source, &component_input)
            .unwrap();
        let mut expected = vec![crate::ComplexDoubleDouble::default(); ab * 2];
        for k2 in 0..b {
            let angle =
                -std::f64::consts::TAU * (k2 as f64 / b as f64 + (n1 * k2) as f64 / ab as f64);
            expected[n1 + a * k2] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(angle.cos()),
                crate::DoubleDouble::from_f64(angle.sin()),
            );
        }
        let max_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 5.0e-12,
            "CUDA DD N102272 upload1 direct-Rader A*B phase mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            config().with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(inverse_ir)) =
            &inverse
        else {
            panic!("CUDA inverse DD N102272 must use forced-Rader recursive scheduling");
        };
        let inverse_components = inverse_ir
            .forced_rader_three_upload_mapped_components()
            .unwrap()
            .expect("CUDA inverse DD N102272 must materialize mapped components");
        let inverse_middle = match &inverse_components[1] {
            crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 1,
                ir: crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(direct),
            } => direct,
            _ => panic!("CUDA inverse DD N102272 upload1 must be direct-Rader"),
        };
        let mut inverse_middle_tile = inverse_middle.clone();
        inverse_middle_tile.batch_count = a * 2;
        inverse_middle_tile.io_mapping =
            crate::StockhamIoMapping::FourStepThreeUpload1(tile_mapping);
        inverse_middle_tile.validate().unwrap();
        let inverse_source = NativeSourceBackend::new(Backend::Cuda)
            .lower_double_double_direct_rader(&inverse_middle_tile)
            .unwrap();
        let inverse_actual = context
            .execute_program_double_double(&inverse_source, &component_input)
            .unwrap();
        let mut inverse_expected = vec![crate::ComplexDoubleDouble::default(); ab * 2];
        let scale = 1.0 / b as f64;
        for k2 in 0..b {
            let angle =
                std::f64::consts::TAU * (k2 as f64 / b as f64 + (n1 * k2) as f64 / ab as f64);
            inverse_expected[n1 + a * k2] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(scale * angle.cos()),
                crate::DoubleDouble::from_f64(scale * angle.sin()),
            );
        }
        let inverse_error = inverse_actual
            .iter()
            .copied()
            .zip(inverse_expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 5.0e-12,
            "CUDA DD N102272 inverse upload1 direct-Rader A*B phase mismatch on {}: {inverse_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n5797_fft_high_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 24 * 1024;
        device.shared_memory_pow2_bytes = 24 * 1024;
        device.max_threads_per_block = 128;
        let length = 11usize * 17 * 31;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N5797 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N5797 must retain forced-Rader upload metadata")
                .axis_split,
            vec![187, 31]
        );
        let mapped_high = ir
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("CUDA DD N5797 FFT-Rader high upload must own Four-step input");
        assert!(matches!(
            mapped_high,
            crate::DoubleDoubleRecursiveFftNodeIr::FftRader(ref fft)
                if fft.prime == 31
                    && matches!(fft.io_mapping, crate::StockhamIoMapping::FourStepRight(_))
        ));

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-10,
            "CUDA DD N5797 fft-high forced-Rader impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-12,
            "CUDA DD N5797 fft-high forced-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-10,
            "CUDA DD/F64 N5797 fft-high forced-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n8789_direct_high_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        let length = 11usize * 17 * 47;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 11;
        tuning.max_rader_direct_prime = 89;
        tuning.validate().unwrap();
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_tuning(tuning)
                .with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N8789 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N8789 must retain forced-Rader upload metadata")
                .axis_split,
            vec![187, 47]
        );
        let mapped_high = ir
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("CUDA DD N8789 direct-Rader high upload must own Four-step input");
        assert!(matches!(
            mapped_high,
            crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(ref direct)
                if direct.prime == 47
                    && matches!(direct.io_mapping, crate::StockhamIoMapping::FourStepRight(_))
        ));

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-10,
            "CUDA DD N8789 direct-high forced-Rader impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_tuning(crate::PlannerTuning::portable())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-12,
            "CUDA DD N8789 direct-high forced-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_tuning(crate::PlannerTuning::portable())
                .with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-10,
            "CUDA DD/F64 N8789 direct-high forced-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n544_monolithic_low_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 16;
        let length = 17usize * 32;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N544 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N544 must retain forced-Rader upload metadata")
                .axis_split,
            vec![32, 17]
        );
        assert!(
            ir.forced_rader_two_upload_mapped_high_component()
                .unwrap()
                .is_some()
        );
        assert!(
            ir.forced_rader_two_upload_mapped_low_stockham()
                .unwrap()
                .is_some()
        );
        let program = crate::ProgramIr::double_double_recursive(ir).unwrap();
        assert!(
            program
                .passes
                .last()
                .unwrap()
                .name
                .ends_with("_four_step_left")
        );

        let (low, mapping) = ir
            .forced_rader_two_upload_mapped_low_stockham()
            .unwrap()
            .expect("CUDA N544 low Stockham mapping");
        let mut low_source = NativeSourceBackend::new(Backend::Cuda)
            .lower_double_double_recursive(ir)
            .unwrap();
        low_source.shaders = vec![low_source.shaders.last().unwrap().clone()];
        low_source.program.passes = vec![low_source.program.passes.last().unwrap().clone()];
        let original_input = low_source
            .program
            .resources
            .iter()
            .position(|resource| resource.kind == crate::program_ir::ProgramResourceKind::Input)
            .unwrap();
        let exchange = low_source
            .program
            .resources
            .iter()
            .position(|resource| resource.name == "double_double_rader_four_step_exchange")
            .unwrap();
        low_source.program.resources[original_input].kind =
            crate::program_ir::ProgramResourceKind::Scratch;
        low_source.program.resources[original_input].initialization =
            crate::program_ir::ProgramResourceInitialization::Zeroed;
        low_source.program.resources[original_input].external_layout = None;
        low_source.program.resources[exchange].kind = crate::program_ir::ProgramResourceKind::Input;
        low_source.program.resources[exchange].initialization =
            crate::program_ir::ProgramResourceInitialization::ExternalInput;
        low_source.program.resources[exchange].external_layout =
            Some(crate::program_ir::ExternalBufferLayout {
                logical_len: low.sequence_len,
                physical_stride: low.sequence_len,
                batch_count: low.batch_count,
                element_shape: crate::program_ir::ProgramElementShape::Complex,
            });
        low_source.validate().unwrap();

        let exchange_input = (0..length)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_f64((0.017 * x).sin() + 0.0003 * x),
                    crate::DoubleDouble::from_f64((0.013 * x).cos() - 0.0002 * x),
                )
            })
            .collect::<Vec<_>>();
        let low_expected = crate::execute_double_double_stockham_ir(low, &exchange_input).unwrap();
        let mut expected = vec![crate::ComplexDoubleDouble::default(); length];
        for batch in 0..low.batch_count {
            let outer_batch = batch / mapping.right_len;
            let k2 = batch % mapping.right_len;
            for k1 in 0..low.sequence_len {
                expected[outer_batch * mapping.logical_len + k2 + mapping.right_len * k1] =
                    low_expected[batch * low.sequence_len + k1];
            }
        }
        let actual_low = context
            .execute_program_double_double(&low_source, &exchange_input)
            .unwrap();
        let mapped_error = actual_low
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            mapped_error <= 2.0e-12,
            "CUDA DD N544 mapped monolithic-low shader mismatch on {}: {mapped_error:e}",
            context.device_name()
        );
        if std::env::var_os("VKFFT_RUN_HEAVY_N544_FORCED_RADER").is_none() {
            return;
        }

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-10,
            "CUDA DD N544 monolithic-low forced-Rader impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-12,
            "CUDA DD N544 monolithic-low forced-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-10,
            "CUDA DD/F64 N544 monolithic-low forced-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_r2r_forced_two_upload_n4352_recursive_high_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64
            || actual_device.vendor != crate::GpuVendor::Nvidia
            || actual_device.shared_memory_bytes < 48 * 1024
            || actual_device.max_threads_per_block < 128
        {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 128;
        let length = 17usize * 256;
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [0usize, 1, 17, 33, 64, 67, 68, 255, length / 2, length - 1];
        let impulse_index = 321usize;

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N4352 DCT-II/III must build 1D R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N4352 DCT-II/III must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N4352 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("CUDA DD N4352 forced-two schedule")
                        .axis_split,
                    vec![64, 68]
                );
                let mapped_high = recursive
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .expect("CUDA DD N4352 mapped high component");
                assert!(matches!(
                    mapped_high,
                    crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ref high)
                        if matches!(
                            high.pack_right.input_modifier,
                            crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepRight(_)
                        )
                ));
                let (low, _) = recursive
                    .forced_rader_two_upload_mapped_low_stockham()
                    .unwrap()
                    .expect("CUDA DD N4352 mapped N64 low Stockham");
                assert_eq!(low.sequence_len, 64);
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
            }

            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let expected = impulse * two * cosine;
                        let delta = (spectrum[k] - expected).abs();
                        let error = delta.hi.abs() + delta.lo.abs();
                        assert!(
                            error <= 5.0e-10,
                            "CUDA DD N4352 forced-two R2R DCT-II bin {k} error {error:e}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 5.0e-11);
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        let expected = 2.0 * impulse * angle.cos();
                        assert!((spectrum[k] - expected).abs() <= 2.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 2.0e-10);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n4352_recursive_high_stockham_low_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 128;
        let length = 17usize * 256;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N4352 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N4352 must retain forced-Rader upload metadata")
                .axis_split,
            vec![64, 68]
        );
        assert!(
            ir.forced_rader_two_upload_mapped_high_component()
                .unwrap()
                .is_some()
        );
        let (mapped_low, _) = ir
            .forced_rader_two_upload_mapped_low_stockham()
            .unwrap()
            .expect("CUDA DD N4352 N64 Stockham low upload should own FourStepLeft");
        assert_eq!(mapped_low.sequence_len, 64);
        let program = crate::ProgramIr::double_double_recursive(ir).unwrap();
        assert!(program.name.contains("mapped_rader_four_step"));
        assert_eq!(
            program.passes.last().unwrap().name,
            format!("{}_four_step_left", mapped_low.name)
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-10,
            "CUDA DD N4352 Stockham-high forced-Rader impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-12,
            "CUDA DD N4352 Stockham-high forced-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-10,
            "CUDA DD/F64 N4352 Stockham-high forced-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_dd_r2r_forced_two_upload_n5100_recursive_low_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64
            || actual_device.vendor != crate::GpuVendor::Nvidia
            || actual_device.shared_memory_bytes < 48 * 1024
            || actual_device.max_threads_per_block < 128
        {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 128;
        let length = 17usize * 300;
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [
            0usize,
            1,
            17,
            33,
            67,
            68,
            74,
            75,
            299,
            length / 2,
            length - 1,
        ];
        let impulse_index = 379usize;

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N5100 DCT-II/III must build 1D R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N5100 DCT-II/III must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N5100 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("CUDA DD N5100 forced-two schedule")
                        .axis_split,
                    vec![68, 75]
                );
                let (high, mapping) = recursive
                    .forced_rader_two_upload_mapped_high_stockham()
                    .unwrap()
                    .expect("CUDA DD N5100 N75 Stockham high boundary");
                assert_eq!(high.sequence_len, 75);
                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("CUDA DD N5100 mapped recursive low boundary");
                assert!(matches!(
                    mapped_low,
                    crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ref low)
                        if matches!(
                            low.scatter_output.output_modifier,
                            crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                                if low_mapping == mapping
                        )
                ));
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
            }

            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let expected = impulse * two * cosine;
                        let delta = (spectrum[k] - expected).abs();
                        let error = delta.hi.abs() + delta.lo.abs();
                        assert!(
                            error <= 5.0e-10,
                            "CUDA DD N5100 recursive-output R2R DCT-II bin {k} error {error:e}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 5.0e-11);
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        let expected = 2.0 * impulse * angle.cos();
                        assert!((spectrum[k] - expected).abs() <= 2.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 2.0e-10);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_forced_two_upload_n5797_fft_high_recursive_low_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 24 * 1024;
        device.shared_memory_pow2_bytes = 24 * 1024;
        device.max_threads_per_block = 128;
        let length = 11usize * 17 * 31;
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [0usize, 1, 11, 17, 30, 31, 186, 187, length / 2, length - 1];
        let impulse_index = 379usize;

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N5797 DCT-II/III must build 1D R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N5797 DCT-II/III must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N5797 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("CUDA DD N5797 forced-two schedule")
                        .axis_split,
                    vec![187, 31]
                );
                let mapped_high = recursive
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .expect("CUDA DD N5797 mapped FFT-Rader high boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::FftRader(high) = mapped_high else {
                    panic!("CUDA DD N5797 high upload must remain p31 FFT-Rader");
                };
                let crate::StockhamIoMapping::FourStepRight(mapping) = high.io_mapping else {
                    panic!("CUDA DD N5797 p31 high upload must own FourStepRight input");
                };
                assert_eq!(high.prime, 31);
                assert_eq!(
                    high.caller_axis_batch_block
                        .expect("CUDA DD N5797 p31 caller block")
                        .threads_per_transform,
                    7
                );
                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("CUDA DD N5797 mapped recursive low boundary");
                assert!(matches!(
                    mapped_low,
                    crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ref low)
                        if low.logical_len == 187
                            && matches!(
                                low.scatter_output.output_modifier,
                                crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                                    if low_mapping == mapping
                            )
                ));
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
            }

            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let expected = impulse * two * cosine;
                        let delta = (spectrum[k] - expected).abs();
                        let error = delta.hi.abs() + delta.lo.abs();
                        assert!(
                            error <= 5.0e-10,
                            "CUDA DD N5797 FFT-Rader-input R2R DCT-II bin {k} error {error:e}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 5.0e-11);
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        let expected = 2.0 * impulse * angle.cos();
                        assert!((spectrum[k] - expected).abs() <= 2.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 2.0e-10);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_forced_two_upload_n345_stockham_high_direct_low_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        let length = 345usize;
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [0usize, 1, 14, 15, 22, 23, length / 2, length - 1];
        let impulse_index = 37usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N345 must build R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N345 must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N345 must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .unwrap()
                        .axis_split,
                    vec![23, 15]
                );
                let (high, mapping) = recursive
                    .forced_rader_two_upload_mapped_high_stockham()
                    .unwrap()
                    .unwrap();
                assert_eq!(high.sequence_len, 15);
                assert_eq!(high.axis_batch_block.unwrap().threads_per_transform, 5);
                let low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .unwrap();
                let crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(low) = low else {
                    panic!("CUDA DD N345 low must be p23 Direct-Rader");
                };
                assert_eq!(low.prime, 23);
                assert!(
                    matches!(low.io_mapping, crate::StockhamIoMapping::FourStepLeft(m) if m == mapping)
                );
                assert_eq!(low.axis_batch_block.unwrap().threads_per_transform, 12);
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(
                    program
                        .resources
                        .iter()
                        .all(|r| !r.name.contains("double_double_r2r_fft_input")
                            && !r.name.contains("double_double_r2r_fft_output"))
                );
            }
            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let delta = (spectrum[k] - impulse * two * cosine).abs();
                        assert!(
                            delta.hi.abs() + delta.lo.abs() <= 5.0e-10,
                            "CUDA DD N345 output bin {k}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(a, e)| {
                            let d = (a - e).abs();
                            d.hi.abs() + d.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(error <= 5.0e-11);
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        assert!((spectrum[k] - 2.0 * impulse * angle.cos()).abs() <= 2.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .zip(&input)
                        .map(|(a, e)| (a - e).abs())
                        .fold(0.0, f64::max);
                    assert!(error <= 2.0e-10);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_whole_axis_bluestein_n103_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let device = context.device_profile();
        if !device.supports_f64 || device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let length = 103usize;
        let impulse_index = 17usize;
        let bins = [0usize, 1, 17, 51, length - 1];
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let build = |direction| {
                let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
                tuning.max_rader_fft_prime = 100;
                TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                        .with_precision(precision)
                        .with_tuning(tuning)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap()
            };
            let forward = build(Direction::Forward);
            let inverse = build(Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N103 Bluestein probe did not build R2R");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N103 Bluestein probe must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("CUDA DD N103 R2R child must remain whole-axis Bluestein");
                };
                assert_eq!(bluestein.convolution_len, 256);
                let child_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
            }
            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.75, 2.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let delta = (spectrum[k] - impulse * two * cosine).abs();
                        assert!(
                            delta.hi.abs() + delta.lo.abs() <= 5.0e-9,
                            "CUDA DD N103 Bluestein output bin {k}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        error <= 5.0e-10,
                        "CUDA DD N103 Bluestein round trip {error:e}"
                    );
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.75f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        assert!((spectrum[k] - 2.0 * impulse * angle.cos()).abs() <= 5.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        error <= 5.0e-10,
                        "CUDA DD/F64 N103 Bluestein round trip {error:e}"
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_whole_axis_bluestein_n103_dst_signs_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let device = context.device_profile();
        if !device.supports_f64 || device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let length = 103usize;
        let impulse_index = 17usize;
        let bins = [0usize, 1, 17, 51, length - 1];
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let build = |direction| {
                let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
                tuning.max_rader_fft_prime = 100;
                TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(crate::TransformKind::Dst(crate::DstType::II))
                        .with_precision(precision)
                        .with_tuning(tuning)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap()
            };
            let forward = build(Direction::Forward);
            let inverse = build(Direction::Inverse);
            for (transform, effective) in [
                (&forward, crate::R2rTransform::Dst(crate::DstType::II)),
                (&inverse, crate::R2rTransform::Dst(crate::DstType::III)),
            ] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N103 DST Bluestein probe did not build R2R");
                };
                assert_eq!(r2r.effective_transform, effective);
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N103 DST Bluestein probe must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("CUDA DD N103 DST child must remain whole-axis Bluestein");
                };
                assert_eq!(bluestein.convolution_len, 256);
                let child_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
            }
            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.75, 2.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64((k + 1) as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (sine, _) = angle.sin_cos();
                        let delta = (spectrum[k] - impulse * two * sine).abs();
                        assert!(
                            delta.hi.abs() + delta.lo.abs() <= 5.0e-9,
                            "CUDA DD N103 DST-II Bluestein output bin {k}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(error <= 5.0e-10, "CUDA DD N103 DST round trip {error:e}");
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.75f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle =
                            std::f64::consts::PI * (impulse_index as f64 + 0.5) * (k + 1) as f64
                                / length as f64;
                        assert!((spectrum[k] - 2.0 * impulse * angle.sin()).abs() <= 5.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        error <= 5.0e-10,
                        "CUDA DD/F64 N103 DST round trip {error:e}"
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_whole_axis_bluestein_n103_grouped_padding_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let device = context.device_profile();
        if !device.supports_f64 || device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let length = 103usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let pad_left = 20usize;
        let pad_right = 25usize;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
            tuning.max_rader_fft_prime = 100;
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_zero_padding(0, pad_left, pad_right)
                    .unwrap()
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                panic!("CUDA grouped padded DD N103 must build R2R IR");
            };
            assert_eq!(r2r.batch_count, batch_count);
            assert_eq!(r2r.grouped_batch, grouped_batch);
            assert_eq!(r2r.zero_padding.unwrap().left, pad_left);
            assert_eq!(r2r.zero_padding.unwrap().right, pad_right);
            let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                panic!("CUDA grouped padded DD N103 must use FFT reduction");
            };
            let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                panic!("CUDA grouped padded DD N103 child must remain whole-axis Bluestein");
            };
            assert_eq!(bluestein.convolution_len, 256);
            assert_eq!(bluestein.batch_count, batch_count);
            assert_eq!(bluestein.grouped_batch, grouped_batch);
            assert!(bluestein.zero_padding.is_none());
            let child_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
            let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
            assert_eq!(program.passes.len(), child_program.passes.len());
        }
        let input = (0..batch_count * length)
            .map(|index| DoubleDouble::from_f64(((index * 37 % 101) as f64 - 50.0) / 31.0))
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double_r2r(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-8,
            "CUDA grouped padded DD N103 Bluestein forward mismatch: {forward_error:e}"
        );
        let inverse_expected = inverse
            .execute_double_double_r2r_reference(&actual)
            .unwrap();
        let inverse_actual = context
            .execute_transform_double_double_r2r(&inverse, &actual)
            .unwrap();
        let inverse_error = inverse_actual
            .iter()
            .copied()
            .zip(inverse_expected.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 2.0e-8,
            "CUDA grouped padded DD N103 Bluestein inverse mismatch: {inverse_error:e}"
        );
        for batch in 0..batch_count {
            for index in pad_left..pad_right {
                let value = inverse_actual[batch * length + index].abs();
                assert!(value.hi.abs() + value.lo.abs() <= 2.0e-12);
            }
        }
    }

    #[test]
    fn cuda_dd_nd_r2r_higher_axis_bluestein_n103_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let device = context.device_profile();
        if !device.supports_f64 || device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let dimensions = vec![103usize, 8usize];
        let batch_count = 2usize;
        let elements = dimensions.iter().product::<usize>() * batch_count;
        let build = |precision, direction| {
            let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
            tuning.max_rader_fft_prime = 100;
            TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_precision(precision)
                    .with_tuning(tuning)
                    .with_grouped_batch(0, 2)
                    .unwrap()
                    .with_grouped_batch(1, 2)
                    .unwrap()
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                    panic!("CUDA DD ND [103,8] must build ND R2R");
                };
                let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                assert_eq!(outer.axis_len, 103);
                assert_eq!(outer.inner_stride, 8);
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } =
                    &outer.transform.algorithm
                else {
                    panic!("CUDA DD ND [103,8] higher axis must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("CUDA DD ND [103,8] higher axis must remain whole-axis Bluestein");
                };
                assert_eq!(bluestein.convolution_len, 256);
                assert_eq!(bluestein.batch_count, 16);
                assert_eq!(bluestein.grouped_batch, 2);
                assert!(bluestein.zero_padding.is_none());
                let child_program = crate::ProgramIr::double_double_r2r(&outer.transform).unwrap();
                let program = crate::ProgramIr::double_double_nd_r2r(nd).unwrap();
                let axis0_passes = program
                    .passes
                    .iter()
                    .filter(|pass| pass.name.contains("vkfft_dd_nd_r2r_axis_0_"))
                    .count();
                assert_eq!(axis0_passes, child_program.passes.len() + 2);
                assert!(program.resources.iter().all(|resource| {
                    !resource
                        .name
                        .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_input")
                        && !resource
                            .name
                            .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_output")
                }));
            }
            match precision {
                Precision::DoubleDouble => {
                    let input = (0..elements)
                        .map(|index| {
                            let x = index as f64;
                            DoubleDouble::from_parts(
                                (0.017 * x).sin() + 0.11 * (0.009 * x).cos() + 0.0002 * x,
                                (index + 1) as f64 * 2.0e-32,
                            )
                        })
                        .collect::<Vec<_>>();
                    let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
                    let actual = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let forward_error = actual
                        .iter()
                        .copied()
                        .zip(expected.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        forward_error <= 5.0e-8,
                        "CUDA DD ND [103,8] Bluestein forward mismatch: {forward_error:e}"
                    );
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &actual)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        round_trip_error <= 5.0e-8,
                        "CUDA DD ND [103,8] Bluestein round trip mismatch: {round_trip_error:e}"
                    );
                }
                Precision::DoubleDoubleF64Storage => {
                    let input = (0..elements)
                        .map(|index| {
                            let x = index as f64;
                            (0.017 * x).sin() + 0.11 * (0.009 * x).cos() + 0.0002 * x
                        })
                        .collect::<Vec<_>>();
                    let expected = forward.execute_r2r_reference(&input).unwrap();
                    let actual = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    let forward_error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        forward_error <= 5.0e-8,
                        "CUDA DD/F64 ND [103,8] Bluestein forward mismatch: {forward_error:e}"
                    );
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &actual)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        round_trip_error <= 5.0e-8,
                        "CUDA DD/F64 ND [103,8] Bluestein round trip mismatch: {round_trip_error:e}"
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_nd_r2r_higher_axis_bluestein_n103_padding_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let device = context.device_profile();
        if !device.supports_f64 || device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let dimensions = vec![103usize, 8usize];
        let batch_count = 2usize;
        let pad_left = 20usize;
        let pad_right = 25usize;
        let elements = dimensions.iter().product::<usize>() * batch_count;
        let build = |precision, direction| {
            let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
            tuning.max_rader_fft_prime = 100;
            TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_precision(precision)
                    .with_tuning(tuning)
                    .with_grouped_batch(0, 2)
                    .unwrap()
                    .with_grouped_batch(1, 2)
                    .unwrap()
                    .with_zero_padding(0, pad_left, pad_right)
                    .unwrap()
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                    panic!("CUDA padded DD ND [103,8] must build ND R2R");
                };
                assert_eq!(nd.zero_padding[0].unwrap().left, pad_left);
                assert_eq!(nd.zero_padding[0].unwrap().right, pad_right);
                let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } =
                    &outer.transform.algorithm
                else {
                    panic!("CUDA padded DD ND [103,8] higher axis must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!(
                        "CUDA padded DD ND [103,8] higher axis must remain whole-axis Bluestein"
                    );
                };
                assert_eq!(bluestein.convolution_len, 256);
                assert_eq!(bluestein.batch_count, 16);
                assert_eq!(bluestein.grouped_batch, 2);
                assert!(bluestein.zero_padding.is_none());
                assert!(outer.transform.zero_padding.is_none());
                let child_program = crate::ProgramIr::double_double_r2r(&outer.transform).unwrap();
                let program = crate::ProgramIr::double_double_nd_r2r(nd).unwrap();
                let axis0_passes = program
                    .passes
                    .iter()
                    .filter(|pass| pass.name.contains("vkfft_dd_nd_r2r_axis_0_"))
                    .count();
                assert_eq!(axis0_passes, child_program.passes.len() + 2);
                assert!(program.resources.iter().all(|resource| {
                    !resource
                        .name
                        .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_input")
                        && !resource
                            .name
                            .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_output")
                }));
            }
            match precision {
                Precision::DoubleDouble => {
                    let input = (0..elements)
                        .map(|index| {
                            let x = index as f64;
                            DoubleDouble::from_parts(
                                (0.017 * x).sin() + 0.11 * (0.009 * x).cos() + 0.0002 * x,
                                (index + 1) as f64 * 2.0e-32,
                            )
                        })
                        .collect::<Vec<_>>();
                    let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
                    let actual = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let forward_error = actual
                        .iter()
                        .copied()
                        .zip(expected.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        forward_error <= 5.0e-8,
                        "CUDA padded DD ND [103,8] Bluestein forward mismatch: {forward_error:e}"
                    );
                    let expected_restored = inverse
                        .execute_double_double_r2r_reference(&actual)
                        .unwrap();
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &actual)
                        .unwrap();
                    let inverse_error = restored
                        .iter()
                        .copied()
                        .zip(expected_restored.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        inverse_error <= 5.0e-8,
                        "CUDA padded DD ND [103,8] Bluestein inverse mismatch: {inverse_error:e}"
                    );
                    for batch in 0..batch_count {
                        let base = batch * dimensions[0] * dimensions[1];
                        for n0 in pad_left..pad_right {
                            for n1 in 0..dimensions[1] {
                                let value = restored[base + n0 * dimensions[1] + n1].abs();
                                assert!(value.hi.abs() + value.lo.abs() <= 5.0e-8);
                            }
                        }
                    }
                }
                Precision::DoubleDoubleF64Storage => {
                    let input = (0..elements)
                        .map(|index| {
                            let x = index as f64;
                            (0.017 * x).sin() + 0.11 * (0.009 * x).cos() + 0.0002 * x
                        })
                        .collect::<Vec<_>>();
                    let expected = forward.execute_r2r_reference(&input).unwrap();
                    let actual = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    let forward_error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        forward_error <= 5.0e-8,
                        "CUDA padded DD/F64 ND [103,8] Bluestein forward mismatch: {forward_error:e}"
                    );
                    let expected_restored = inverse.execute_r2r_reference(&actual).unwrap();
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &actual)
                        .unwrap();
                    let inverse_error = restored
                        .iter()
                        .zip(&expected_restored)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        inverse_error <= 5.0e-8,
                        "CUDA padded DD/F64 ND [103,8] Bluestein inverse mismatch: {inverse_error:e}"
                    );
                    for batch in 0..batch_count {
                        let base = batch * dimensions[0] * dimensions[1];
                        for n0 in pad_left..pad_right {
                            for n1 in 0..dimensions[1] {
                                assert!(restored[base + n0 * dimensions[1] + n1].abs() <= 5.0e-8);
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_forced_two_upload_n285_stockham_high_fft_rader_low_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        let length = 285usize;
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [0usize, 1, 14, 15, 18, 19, length / 2, length - 1];
        let impulse_index = 37usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N285 must build R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N285 must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N285 must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .unwrap()
                        .axis_split,
                    vec![19, 15]
                );
                let (high, mapping) = recursive
                    .forced_rader_two_upload_mapped_high_stockham()
                    .unwrap()
                    .unwrap();
                assert_eq!(high.sequence_len, 15);
                assert!(high.axis_batch_block.is_some());
                let low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .unwrap();
                let crate::DoubleDoubleRecursiveFftNodeIr::FftRader(low) = low else {
                    panic!("CUDA DD N285 low must be p19 FFT-Rader");
                };
                assert_eq!(low.prime, 19);
                assert!(matches!(
                    low.io_mapping,
                    crate::StockhamIoMapping::FourStepLeft(m) if m == mapping
                ));
                assert!(low.caller_axis_batch_block.is_some());
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(
                    program
                        .resources
                        .iter()
                        .all(|r| !r.name.contains("double_double_r2r_fft_input")
                            && !r.name.contains("double_double_r2r_fft_output"))
                );
            }
            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let delta = (spectrum[k] - impulse * two * cosine).abs();
                        assert!(
                            delta.hi.abs() + delta.lo.abs() <= 5.0e-10,
                            "CUDA DD N285 output bin {k}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(a, e)| {
                            let d = (a - e).abs();
                            d.hi.abs() + d.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(error <= 5.0e-11);
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        assert!((spectrum[k] - 2.0 * impulse * angle.cos()).abs() <= 2.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let error = restored
                        .iter()
                        .zip(&input)
                        .map(|(a, e)| (a - e).abs())
                        .fold(0.0, f64::max);
                    assert!(error <= 2.0e-10);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_dd_r2r_forced_two_upload_n8789_direct_high_recursive_low_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        let length = 11usize * 17 * 47;
        let build = |precision, direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 11;
            tuning.max_rader_direct_prime = 89;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [0usize, 1, 11, 17, 46, 47, 186, 187, length / 2, length - 1];
        let impulse_index = 379usize;

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N8789 DCT-II/III must build 1D R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N8789 DCT-II/III must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N8789 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("CUDA DD N8789 forced-two schedule")
                        .axis_split,
                    vec![187, 47]
                );
                let mapped_high = recursive
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .expect("CUDA DD N8789 mapped Direct-Rader high boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(high) = mapped_high else {
                    panic!("CUDA DD N8789 high upload must remain p47 Direct-Rader");
                };
                let crate::StockhamIoMapping::FourStepRight(mapping) = high.io_mapping else {
                    panic!("CUDA DD N8789 p47 high upload must own FourStepRight input");
                };
                assert_eq!(high.prime, 47);
                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("CUDA DD N8789 mapped recursive low boundary");
                assert!(matches!(
                    mapped_low,
                    crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ref low)
                        if low.logical_len == 187
                            && matches!(
                                low.scatter_output.output_modifier,
                                crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                                    if low_mapping == mapping
                            )
                ));
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
            }

            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let expected = impulse * two * cosine;
                        let delta = (spectrum[k] - expected).abs();
                        let error = delta.hi.abs() + delta.lo.abs();
                        assert!(
                            error <= 5.0e-10,
                            "CUDA DD N8789 Direct-Rader-input R2R DCT-II bin {k} error {error:e}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 5.0e-11);
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        let expected = 2.0 * impulse * angle.cos();
                        assert!((spectrum[k] - expected).abs() <= 2.0e-9);
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(round_trip_error <= 2.0e-10);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_forced_rader_n5100_mapped_high_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 128;
        let length = 17usize * 300;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            &forward
        else {
            panic!("CUDA DD N5100 must use forced-Rader recursive scheduling");
        };
        assert_eq!(
            ir.rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD N5100 must retain forced-Rader upload metadata")
                .axis_split,
            vec![68, 75]
        );
        assert!(
            ir.forced_rader_two_upload_mapped_high_stockham()
                .unwrap()
                .is_some()
        );
        assert!(
            ir.forced_rader_two_upload_mapped_low_component()
                .unwrap()
                .is_some()
        );

        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[1] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re.to_f64() - angle.cos()).hypot(actual.im.to_f64() - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-10,
            "CUDA DD N5100 mapped forced-Rader impulse mismatch on {}: {max_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-12,
            "CUDA DD N5100 mapped forced-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let angle = -std::f64::consts::TAU * index as f64 / length as f64;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-10,
            "CUDA DD/F64 N5100 mapped forced-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_recursive_rader_composition_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let grouped_batch = 3usize;
        for (length, batch_count) in [(17usize * 17, 7usize), (17usize * 19, 7usize)] {
            let forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            assert!(matches!(
                forward,
                TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(_))
            ));
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    crate::ComplexDoubleDouble::new(
                        crate::DoubleDouble::from_parts(
                            (0.013 * x).sin() + 0.00007 * x,
                            (index + 1) as f64 * 2.0e-31,
                        ),
                        crate::DoubleDouble::from_parts(
                            (0.009 * x).cos() - 0.00003 * x,
                            -(index as f64 + 1.0) * 1.0e-31,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let expected = forward.execute_double_double_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let dd_error = |actual: crate::ComplexDoubleDouble,
                            expected: crate::ComplexDoubleDouble| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            };
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 5.0e-17,
                "CUDA DD recursive length {length} forward mismatch on {}: {forward_error:e}",
                context.device_name()
            );
            let restored = context
                .execute_transform_double_double(&inverse, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 5.0e-15,
                "CUDA DD recursive length {length} round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );

            let f64_forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(crate::ComplexDoubleDouble::to_complex64)
                .collect::<Vec<_>>();
            let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_error <= 5.0e-10,
                "CUDA DD/F64 recursive length {length} mismatch on {}: {f64_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_round_trip_error <= 5.0e-11,
                "CUDA DD/F64 recursive length {length} round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_real_gpu_double_double_bluestein_p103_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 103usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let zero_left = 19usize;
        let zero_right = 31usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, zero_left, zero_right)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, zero_left, zero_right)
                .unwrap(),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        for transform in [&forward, &inverse] {
            let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                bluestein,
            )) = transform
            else {
                panic!("forced CUDA DD p103 plan must use Bluestein");
            };
            assert_eq!(bluestein.convolution_len, 256);
            assert_eq!(bluestein.batch_count, batch_count);
            assert_eq!(bluestein.grouped_batch, grouped_batch);
            assert_eq!(bluestein.batch_group_count(), 3);
            assert!(bluestein.has_spatial_zero_padding());
            let block = bluestein
                .stockham_convolution_axis_batch_block()
                .expect("CUDA DD p103/G3 should materialize M256 Quad geometry");
            assert_eq!(block.threads_per_transform, 32);
            assert_eq!(block.grouped_batch, grouped_batch);
            assert_eq!([block.local_size_x, block.local_size_y], [3, 32]);
            assert!(block.transforms_on_x);
            assert!(block.axis_swapped);
        }
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00007 * x + 17.0,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.009 * x).cos() - 0.00003 * x - 11.0,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let mut manual_zeroed = input.clone();
        for batch in 0..batch_count {
            let base = batch * length;
            for index in zero_left..zero_right {
                manual_zeroed[base + index] = crate::ComplexDoubleDouble::default();
            }
        }
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 5.0e-19,
            "CUDA DD p103 Bluestein forward mismatch on {}: {error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(manual_zeroed.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-17,
            "CUDA DD padded/grouped p103 Bluestein round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, zero_left, zero_right)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, zero_left, zero_right)
                .unwrap(),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        for transform in [&f64_ir, &f64_inverse] {
            let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                bluestein,
            )) = transform
            else {
                panic!("forced CUDA DD/F64 p103 plan must use Bluestein");
            };
            let block = bluestein
                .stockham_convolution_axis_batch_block()
                .expect("CUDA DD/F64 p103/G3 should materialize M256 Quad geometry");
            assert_eq!([block.local_size_x, block.local_size_y], [3, 32]);
            assert_eq!(block.grouped_batch, grouped_batch);
        }
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11 * length as f64,
            "CUDA DD/F64 p103 Bluestein mismatch on {}: {f64_error:e}",
            context.device_name()
        );
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_manual_zeroed = manual_zeroed
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_manual_zeroed)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 5.0e-10,
            "CUDA DD/F64 padded/grouped p103 Bluestein round trip mismatch on {}: {f64_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_n611_default_bluestein_m1225_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 611usize;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                bluestein,
            )) = transform
            else {
                panic!("CUDA device-default DD N611 must use Bluestein");
            };
            assert_eq!(bluestein.convolution_len, 1_225);
            assert!(matches!(
                bluestein.forward_fft,
                crate::DoubleDoubleBluesteinConvolutionIr::Stockham(_)
            ));
            assert!(matches!(
                bluestein.inverse_fft,
                crate::DoubleDoubleBluesteinConvolutionIr::Stockham(_)
            ));
            assert_eq!(
                crate::ProgramIr::double_double_bluestein(bluestein)
                    .unwrap()
                    .passes
                    .len(),
                10
            );
        }

        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.017 * x).sin() + 0.00011 * x + 3.0,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.011 * x).cos() - 0.00007 * x - 2.0,
                        -(index as f64 + 1.0) * 0.5e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-15,
            "CUDA DD N611/M1225 forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-14,
            "CUDA DD N611/M1225 round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let build_f64 = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let f64_forward = build_f64(Direction::Forward);
        let f64_inverse = build_f64(Direction::Inverse);
        for transform in [&f64_forward, &f64_inverse] {
            let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                bluestein,
            )) = transform
            else {
                panic!("CUDA device-default DD/F64 N611 must use Bluestein");
            };
            assert_eq!(bluestein.convolution_len, 1_225);
            assert_eq!(bluestein.external_storage, crate::PrecisionStorage::F64);
            assert_eq!(
                crate::ProgramIr::double_double_bluestein(bluestein)
                    .unwrap()
                    .passes
                    .len(),
                10
            );
        }
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11 * length as f64,
            "CUDA DD/F64 N611/M1225 mismatch on {}: {f64_error:e}",
            context.device_name()
        );
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 5.0e-10,
            "CUDA DD/F64 N611/M1225 round trip mismatch on {}: {f64_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_explicit_portable_recursive_p419_bluestein_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 419usize;
        let impulse_index = 23usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                bluestein,
            )) = transform
            else {
                panic!("CUDA DD p419 must use precision-specific Bluestein fallback");
            };
            assert!(bluestein.convolution_len >= 2 * length - 1);
        }

        let impulse = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_parts(1.25, 3.0e-31),
            crate::DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = impulse
                    * crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-15,
            "CUDA DD p419 precision-specific Bluestein mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-14,
            "CUDA DD p419 precision-specific Bluestein round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_bluestein_p2053_recursive_convolution_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 2_053usize;
        let impulse_index = 137usize;
        let mut tuning =
            crate::PlannerTuning::for_device(context.device_profile(), Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
            bluestein,
        )) = &forward
        else {
            panic!("forced CUDA DD p2053 plan must use Bluestein");
        };
        assert_eq!(bluestein.convolution_len, 4_368);
        assert!(matches!(
            &bluestein.forward_fft,
            crate::DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            &bluestein.inverse_fft,
            crate::DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let crate::DoubleDoubleBluesteinConvolutionIr::Recursive(recursive) =
            &bluestein.forward_fft
        else {
            unreachable!("validated CUDA DD p2053 recursive convolution");
        };
        assert_eq!(recursive.logical_len, 4_368);
        assert!(
            recursive.stockham_upload_schedule.is_none(),
            "CUDA DD M4368 is a mixed Quad-Stockham + p13 Rader tree"
        );
        let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
        child_program.validate().unwrap();
        assert!(child_program.passes.len() > 2);

        let impulse = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_parts(1.25, 3.0e-31),
            crate::DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = impulse
                    * crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-15,
            "CUDA DD p2053 recursive Bluestein mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-14,
            "CUDA DD p2053 recursive Bluestein round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = (impulse
                    * crate::double_double_unit_root(
                        impulse_index * k,
                        length,
                        Direction::Forward,
                    )
                    .unwrap())
                .to_complex64();
                (actual - expected).norm_sqr().sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 5.0e-8,
            "CUDA DD/F64 p2053 recursive Bluestein mismatch on {}: {f64_forward_error:e}",
            context.device_name()
        );
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 1.0e-8,
            "CUDA DD/F64 p2053 recursive Bluestein round trip mismatch on {}: {f64_round_trip_error:e}",
            context.device_name()
        );

        let real_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let real_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealDoubleDouble(real_forward_ir) = &real_forward else {
            panic!("CUDA DD p2053 R2C must use real DD IR");
        };
        let TransformIr::RealDoubleDouble(real_inverse_ir) = &real_inverse else {
            panic!("CUDA DD p2053 C2R must use real DD IR");
        };
        let real_impulse = crate::DoubleDouble::from_parts(1.25, 3.0e-31);
        let mut real_input = vec![crate::DoubleDouble::ZERO; length];
        real_input[impulse_index] = real_impulse;
        let real_spectrum = context
            .execute_double_double_r2c(real_forward_ir, &real_input)
            .unwrap();
        let real_forward_error = real_spectrum
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected =
                    crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap()
                        .scale_dd(real_impulse);
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            real_forward_error <= 2.0e-15,
            "CUDA DD real p2053 recursive Bluestein mismatch on {}: {real_forward_error:e}",
            context.device_name()
        );
        let real_restored = context
            .execute_double_double_c2r(real_inverse_ir, &real_spectrum)
            .unwrap();
        let real_round_trip_error = real_restored
            .iter()
            .copied()
            .zip(real_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            real_round_trip_error <= 5.0e-14,
            "CUDA DD real p2053 round trip mismatch on {}: {real_round_trip_error:e}",
            context.device_name()
        );

        let real_f64_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let real_f64_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealDoubleDouble(real_f64_forward_ir) = &real_f64_forward else {
            panic!("CUDA DD/F64 p2053 R2C must use real DD IR");
        };
        let TransformIr::RealDoubleDouble(real_f64_inverse_ir) = &real_f64_inverse else {
            panic!("CUDA DD/F64 p2053 C2R must use real DD IR");
        };
        let f64_real_input = real_input
            .iter()
            .copied()
            .map(crate::DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_real_spectrum = context
            .execute_double_double_r2c_f64_storage(real_f64_forward_ir, &f64_real_input)
            .unwrap();
        let f64_real_forward_error = f64_real_spectrum
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected =
                    crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap()
                        .scale_dd(real_impulse)
                        .to_complex64();
                (actual - expected).norm_sqr().sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            f64_real_forward_error <= 5.0e-8,
            "CUDA DD/F64 real p2053 mismatch on {}: {f64_real_forward_error:e}",
            context.device_name()
        );
        let f64_real_restored = context
            .execute_double_double_c2r_f64_storage(real_f64_inverse_ir, &f64_real_spectrum)
            .unwrap();
        let f64_real_round_trip_error = f64_real_restored
            .iter()
            .zip(&f64_real_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_real_round_trip_error <= 1.0e-8,
            "CUDA DD/F64 real p2053 round trip mismatch on {}: {f64_real_round_trip_error:e}",
            context.device_name()
        );

        let dct_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let dct_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let mut dct_input = vec![crate::DoubleDouble::ZERO; length];
        dct_input[impulse_index] = real_impulse;
        let dct_actual = context
            .execute_transform_double_double_r2r(&dct_forward, &dct_input)
            .unwrap();
        let dct_expected = |k: usize| {
            let angle = crate::DoubleDouble::PI
                * (crate::DoubleDouble::from_f64(impulse_index as f64)
                    + crate::DoubleDouble::from_f64(0.5))
                * crate::DoubleDouble::from_f64(k as f64)
                / crate::DoubleDouble::from_f64(length as f64);
            let (_, cosine) = angle.sin_cos();
            real_impulse * crate::DoubleDouble::from_f64(2.0) * cosine
        };
        for k in [0usize, 1, 17, length / 2, length - 1] {
            let delta = (dct_actual[k] - dct_expected(k)).abs();
            let error = delta.hi.abs() + delta.lo.abs();
            assert!(
                error <= 2.0e-14,
                "CUDA DD DCT-II p2053 bin {k} mismatch on {}: {error:e}",
                context.device_name()
            );
        }
        let dct_restored = context
            .execute_transform_double_double_r2r(&dct_inverse, &dct_actual)
            .unwrap();
        let dct_round_trip_error = dct_restored
            .iter()
            .copied()
            .zip(dct_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            dct_round_trip_error <= 2.0e-12,
            "CUDA DD DCT-II/III p2053 round trip mismatch on {}: {dct_round_trip_error:e}",
            context.device_name()
        );

        let dct_f64_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let dct_f64_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let f64_dct_input = dct_input
            .iter()
            .copied()
            .map(crate::DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_dct_actual = context
            .execute_transform_double_double_r2r_f64_storage(&dct_f64_forward, &f64_dct_input)
            .unwrap();
        for k in [0usize, 1, 17, length / 2, length - 1] {
            let error = (f64_dct_actual[k] - dct_expected(k).to_f64()).abs();
            assert!(
                error <= 2.0e-7,
                "CUDA DD/F64 DCT-II p2053 bin {k} mismatch on {}: {error:e}",
                context.device_name()
            );
        }
        let f64_dct_restored = context
            .execute_transform_double_double_r2r_f64_storage(&dct_f64_inverse, &f64_dct_actual)
            .unwrap();
        let f64_dct_round_trip_error = f64_dct_restored
            .iter()
            .zip(&f64_dct_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_dct_round_trip_error <= 5.0e-8,
            "CUDA DD/F64 DCT-II/III p2053 round trip mismatch on {}: {f64_dct_round_trip_error:e}",
            context.device_name()
        );

        let nd_dimensions = vec![2usize, length];
        let build_nd_real = |precision, transform| {
            TransformIr::build(
                FftConfig::new(nd_dimensions.clone())
                    .with_transform(transform)
                    .with_precision(precision)
                    .with_inverse_normalization(transform == crate::TransformKind::ComplexToReal)
                    .with_tuning(tuning),
                if transform == crate::TransformKind::RealToComplex {
                    Direction::Forward
                } else {
                    Direction::Inverse
                },
                context.device_profile(),
            )
            .unwrap()
        };
        let nd_r2c = build_nd_real(Precision::DoubleDouble, crate::TransformKind::RealToComplex);
        let nd_c2r = build_nd_real(Precision::DoubleDouble, crate::TransformKind::ComplexToReal);
        let TransformIr::RealNdDoubleDouble(nd_r2c_ir) = &nd_r2c else {
            panic!("CUDA DD [2,2053] R2C must use ND real DD IR");
        };
        let TransformIr::RealNdDoubleDouble(nd_c2r_ir) = &nd_c2r else {
            panic!("CUDA DD [2,2053] C2R must use ND real DD IR");
        };
        let mut nd_real_input = vec![crate::DoubleDouble::ZERO; 2 * length];
        nd_real_input[length + impulse_index] = real_impulse;
        let nd_real_spectrum = context
            .execute_double_double_nd_r2c(nd_r2c_ir, &nd_real_input)
            .unwrap();
        let compact_last = length / 2 + 1;
        let mut nd_real_forward_error = 0.0f64;
        for k0 in 0..2usize {
            let root0 = crate::double_double_unit_root(k0, 2, Direction::Forward).unwrap();
            for k1 in 0..compact_last {
                let root1 =
                    crate::double_double_unit_root(impulse_index * k1, length, Direction::Forward)
                        .unwrap();
                let expected = (root0 * root1).scale_dd(real_impulse);
                nd_real_forward_error = nd_real_forward_error
                    .max(dd_error(nd_real_spectrum[k0 * compact_last + k1], expected));
            }
        }
        assert!(
            nd_real_forward_error <= 5.0e-14,
            "CUDA DD ND-real [2,2053] mismatch on {}: {nd_real_forward_error:e}",
            context.device_name()
        );
        let nd_real_restored = context
            .execute_double_double_nd_c2r(nd_c2r_ir, &nd_real_spectrum)
            .unwrap();
        let nd_real_round_trip_error = nd_real_restored
            .iter()
            .copied()
            .zip(nd_real_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            nd_real_round_trip_error <= 2.0e-12,
            "CUDA DD ND-real [2,2053] round trip mismatch on {}: {nd_real_round_trip_error:e}",
            context.device_name()
        );

        let nd_f64_r2c = build_nd_real(
            Precision::DoubleDoubleF64Storage,
            crate::TransformKind::RealToComplex,
        );
        let nd_f64_c2r = build_nd_real(
            Precision::DoubleDoubleF64Storage,
            crate::TransformKind::ComplexToReal,
        );
        let TransformIr::RealNdDoubleDouble(nd_f64_r2c_ir) = &nd_f64_r2c else {
            panic!("CUDA DD/F64 [2,2053] R2C must use ND real DD IR");
        };
        let TransformIr::RealNdDoubleDouble(nd_f64_c2r_ir) = &nd_f64_c2r else {
            panic!("CUDA DD/F64 [2,2053] C2R must use ND real DD IR");
        };
        let nd_f64_real_input = nd_real_input
            .iter()
            .copied()
            .map(crate::DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let nd_f64_real_spectrum = context
            .execute_double_double_nd_r2c_f64_storage(nd_f64_r2c_ir, &nd_f64_real_input)
            .unwrap();
        let mut nd_f64_real_forward_error = 0.0f64;
        for k0 in 0..2usize {
            let root0 = crate::double_double_unit_root(k0, 2, Direction::Forward).unwrap();
            for k1 in 0..compact_last {
                let root1 =
                    crate::double_double_unit_root(impulse_index * k1, length, Direction::Forward)
                        .unwrap();
                let expected = (root0 * root1).scale_dd(real_impulse).to_complex64();
                nd_f64_real_forward_error = nd_f64_real_forward_error.max(
                    (nd_f64_real_spectrum[k0 * compact_last + k1] - expected)
                        .norm_sqr()
                        .sqrt(),
                );
            }
        }
        assert!(
            nd_f64_real_forward_error <= 2.0e-7,
            "CUDA DD/F64 ND-real [2,2053] mismatch on {}: {nd_f64_real_forward_error:e}",
            context.device_name()
        );
        let nd_f64_real_restored = context
            .execute_double_double_nd_c2r_f64_storage(nd_f64_c2r_ir, &nd_f64_real_spectrum)
            .unwrap();
        let nd_f64_real_round_trip_error = nd_f64_real_restored
            .iter()
            .zip(&nd_f64_real_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            nd_f64_real_round_trip_error <= 5.0e-8,
            "CUDA DD/F64 ND-real [2,2053] round trip mismatch on {}: {nd_f64_real_round_trip_error:e}",
            context.device_name()
        );

        let build_nd_dct = |precision, direction| {
            TransformIr::build(
                FftConfig::new(nd_dimensions.clone())
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_precision(precision)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_tuning(tuning),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let nd_dct_forward = build_nd_dct(Precision::DoubleDouble, Direction::Forward);
        let nd_dct_inverse = build_nd_dct(Precision::DoubleDouble, Direction::Inverse);
        let mut nd_dct_input = vec![crate::DoubleDouble::ZERO; 2 * length];
        nd_dct_input[length + impulse_index] = real_impulse;
        let nd_dct_actual = context
            .execute_transform_double_double_r2r(&nd_dct_forward, &nd_dct_input)
            .unwrap();
        let outer_coefficient = |k0: usize| {
            let angle = crate::DoubleDouble::PI
                * crate::DoubleDouble::from_f64(1.5)
                * crate::DoubleDouble::from_f64(k0 as f64)
                / crate::DoubleDouble::from_f64(2.0);
            let (_, cosine) = angle.sin_cos();
            crate::DoubleDouble::from_f64(2.0) * cosine
        };
        for k0 in 0..2usize {
            for k1 in [0usize, 1, 17, length / 2, length - 1] {
                let expected = dct_expected(k1) * outer_coefficient(k0);
                let delta = (nd_dct_actual[k0 * length + k1] - expected).abs();
                let error = delta.hi.abs() + delta.lo.abs();
                assert!(
                    error <= 5.0e-13,
                    "CUDA DD ND DCT-II [2,2053] bin ({k0},{k1}) mismatch on {}: {error:e}",
                    context.device_name()
                );
            }
        }
        let nd_dct_restored = context
            .execute_transform_double_double_r2r(&nd_dct_inverse, &nd_dct_actual)
            .unwrap();
        let nd_dct_round_trip_error = nd_dct_restored
            .iter()
            .copied()
            .zip(nd_dct_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            nd_dct_round_trip_error <= 5.0e-11,
            "CUDA DD ND DCT-II/III [2,2053] round trip mismatch on {}: {nd_dct_round_trip_error:e}",
            context.device_name()
        );

        let nd_f64_dct_forward =
            build_nd_dct(Precision::DoubleDoubleF64Storage, Direction::Forward);
        let nd_f64_dct_inverse =
            build_nd_dct(Precision::DoubleDoubleF64Storage, Direction::Inverse);
        let nd_f64_dct_input = nd_dct_input
            .iter()
            .copied()
            .map(crate::DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let nd_f64_dct_actual = context
            .execute_transform_double_double_r2r_f64_storage(&nd_f64_dct_forward, &nd_f64_dct_input)
            .unwrap();
        for k0 in 0..2usize {
            for k1 in [0usize, 1, 17, length / 2, length - 1] {
                let expected = (dct_expected(k1) * outer_coefficient(k0)).to_f64();
                let error = (nd_f64_dct_actual[k0 * length + k1] - expected).abs();
                assert!(
                    error <= 1.0e-6,
                    "CUDA DD/F64 ND DCT-II [2,2053] bin ({k0},{k1}) mismatch on {}: {error:e}",
                    context.device_name()
                );
            }
        }
        let nd_f64_dct_restored = context
            .execute_transform_double_double_r2r_f64_storage(
                &nd_f64_dct_inverse,
                &nd_f64_dct_actual,
            )
            .unwrap();
        let nd_f64_dct_round_trip_error = nd_f64_dct_restored
            .iter()
            .zip(&nd_f64_dct_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            nd_f64_dct_round_trip_error <= 2.0e-7,
            "CUDA DD/F64 ND DCT-II/III [2,2053] round trip mismatch on {}: {nd_f64_dct_round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_fft_rader_p257_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 257usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let caller_block =
            crate::scheduler::plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block(
                length,
                batch_count,
                Some(grouped_batch),
                context.device_profile(),
            )
            .unwrap();
        let child_block =
            crate::scheduler::plan_gpu_double_double_axis0_user_grouped_stockham_block(
                length - 1,
                batch_count,
                Some(grouped_batch),
                context.device_profile(),
            )
            .unwrap();
        let (Some(caller_block), Some(child_block)) = (
            caller_block.filter(|block| block.grouped_batch == grouped_batch),
            child_block.filter(|block| block.grouped_batch == grouped_batch),
        ) else {
            return;
        };
        let config = |precision, inverse_normalization| {
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_inverse_normalization(inverse_normalization)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
        };
        let forward = TransformIr::build(
            config(Precision::DoubleDouble, false),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            config(Precision::DoubleDouble, true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(ir)) =
                transform
            else {
                panic!("grouped CUDA DD p257 should remain FFT Rader");
            };
            assert_eq!(ir.caller_axis_batch_block, Some(caller_block));
            assert_eq!(
                ir.forward_fft.stockham_axis_batch_block(),
                Some(child_block)
            );
            assert_eq!(
                ir.inverse_fft.stockham_axis_batch_block(),
                Some(child_block)
            );
            assert_eq!(ir.batch_group_count(), 3);
        }
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00007 * x,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.009 * x).cos() - 0.00003 * x,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 2.0e-22,
            "CUDA DD p257 FFT-Rader forward mismatch on {}: {error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-20,
            "CUDA DD p257 FFT-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            config(Precision::DoubleDoubleF64Storage, false),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(f64_rader)) =
            &f64_ir
        else {
            panic!("grouped CUDA DD/F64 p257 should remain FFT Rader");
        };
        assert_eq!(f64_rader.caller_axis_batch_block, Some(caller_block));
        assert_eq!(
            f64_rader.forward_fft.stockham_axis_batch_block(),
            Some(child_block)
        );
        assert_eq!(
            f64_rader.inverse_fft.stockham_axis_batch_block(),
            Some(child_block)
        );
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-11 * length as f64,
            "CUDA DD/F64 p257 FFT-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_nested_direct_sub_rader_composite_n566_parent_floor_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        let length = 2usize * 283;
        let batch_count = 2usize;
        let Some(expected_threads) =
            crate::scheduler::plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
                length,
                &[(283, 1)],
                batch_count,
                profile,
            )
            .unwrap()
        else {
            return;
        };
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N566 nested direct sub-Rader should remain recursive C2C");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("CUDA N566 should keep smooth-2 x p283 Cooley root");
            };
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [expected_threads, 1]
            );
            let outer = match (&root.left, &root.right) {
                (crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader), _)
                    if rader.prime == 283 =>
                {
                    rader
                }
                (_, crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader))
                    if rader.prime == 283 =>
                {
                    rader
                }
                _ => panic!("CUDA N566 should contain p283 FFT-Rader child"),
            };
            fn contains_direct_prime(
                node: &crate::recursive_ir::RecursiveFftNodeIr,
                prime: usize,
            ) -> bool {
                match node {
                    crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) => {
                        direct.prime == prime
                    }
                    crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(cooley) => {
                        contains_direct_prime(&cooley.left, prime)
                            || contains_direct_prime(&cooley.right, prime)
                    }
                    _ => false,
                }
            }
            assert!(contains_direct_prime(
                &outer.forward_recursive().unwrap().root,
                47
            ));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(forward_error <= 4.0e-3);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(round_trip_error <= 4.0e-3);
    }

    #[test]
    fn cuda_nested_device_capped_sub_rader_n566_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut profile = context.device_profile();
        profile.max_threads_per_block = profile.max_threads_per_block.min(64);
        profile.max_workgroup_size[0] = profile.max_workgroup_size[0].min(64);
        let length = 2usize * 283;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let effective =
            crate::config::upstream_effective_rader_tuning(tuning, profile, crate::Precision::F32);
        if effective.max_rader_direct_prime != 31 {
            return;
        }
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA device-capped N566 should remain recursive C2C");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("CUDA device-capped N566 should keep smooth-2 x p283 Cooley root");
            };
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 19);
            assert_eq!([block.local_size_x, block.local_size_y], [19, 1]);
            let outer = match (&root.left, &root.right) {
                (crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader), _)
                    if rader.prime == 283 =>
                {
                    rader
                }
                (_, crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader))
                    if rader.prime == 283 =>
                {
                    rader
                }
                _ => panic!("CUDA device-capped N566 should contain p283 FFT-Rader child"),
            };
            fn contains_fft_prime(
                node: &crate::recursive_ir::RecursiveFftNodeIr,
                prime: usize,
            ) -> bool {
                match node {
                    crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) => {
                        rader.prime == prime
                    }
                    crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(cooley) => {
                        contains_fft_prime(&cooley.left, prime)
                            || contains_fft_prime(&cooley.right, prime)
                    }
                    _ => false,
                }
            }
            fn contains_direct_prime(
                node: &crate::recursive_ir::RecursiveFftNodeIr,
                prime: usize,
            ) -> bool {
                match node {
                    crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) => {
                        direct.prime == prime
                    }
                    crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(cooley) => {
                        contains_direct_prime(&cooley.left, prime)
                            || contains_direct_prime(&cooley.right, prime)
                    }
                    _ => false,
                }
            }
            let convolution_root = &outer.forward_recursive().unwrap().root;
            assert!(contains_fft_prime(convolution_root, 47));
            assert!(!contains_direct_prime(convolution_root, 47));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(forward_error <= 4.0e-3);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(round_trip_error <= 4.0e-3);
    }

    #[test]
    fn cuda_nested_sub_rader_composite_n214_parent_floor_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 2usize * 107;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N214 nested sub-Rader should remain recursive C2C");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("CUDA N214 should keep smooth-2 x p107 Cooley root");
            };
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 18);
            assert_eq!([block.local_size_x, block.local_size_y], [18, 1]);
            let outer = match (&root.left, &root.right) {
                (crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader), _)
                    if rader.prime == 107 =>
                {
                    rader
                }
                (_, crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader))
                    if rader.prime == 107 =>
                {
                    rader
                }
                _ => panic!("CUDA N214 should contain p107 FFT-Rader child"),
            };
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(conv) =
                &outer.forward_recursive().unwrap().root
            else {
                panic!("CUDA p107 convolution should remain 106=2x53");
            };
            assert!(matches!(
                conv.right,
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(ref sub) if sub.prime == 53
            ));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = Complex32::new(1.0, 0.0);
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = spectrum
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let k = index % length;
                let angle = -std::f32::consts::TAU * k as f32 / length as f32;
                (actual.re - angle.cos()).hypot(actual.im - angle.sin())
            })
            .fold(0.0f32, f32::max);
        assert!(forward_error <= 4.0e-3);
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(round_trip_error <= 4.0e-3);
    }

    #[test]
    fn cuda_double_double_device_default_min_direct_11_n106_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64 {
            return;
        }
        let length = 2usize * 53;
        let tuning = crate::PlannerTuning::for_device(profile, Precision::DoubleDouble)
            .with_recursive_fft_rader(true);
        tuning.validate().unwrap();
        assert_eq!(tuning.min_rader_direct_prime, 11);
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N106 should use recursive 2 x p53");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N106 should keep a Cooley root");
            };
            assert_eq!(
                root.pack_right
                    .axis_batch_block
                    .unwrap()
                    .threads_per_transform,
                9
            );
            let outer = match (&root.left, &root.right) {
                (
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                        rader,
                    ),
                    _,
                ) if rader.prime == 53 => rader,
                (
                    _,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                        rader,
                    ),
                ) if rader.prime == 53 => rader,
                _ => panic!("CUDA DD N106 should contain p53 FFT-Rader child"),
            };
            let crate::double_double_ir::DoubleDoubleBluesteinConvolutionIr::Recursive(conv) =
                &outer.forward_fft
            else {
                panic!("CUDA DD p53 convolution should remain recursive 52=4x13");
            };
            fn contains_direct_p13(
                node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            ) -> bool {
                match node {
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == 13,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => contains_direct_p13(&cooley.left) || contains_direct_p13(&cooley.right),
                    _ => false,
                }
            }
            assert!(contains_direct_p13(&conv.root));
        }
        let impulse = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_parts(1.25, 3.0e-31),
            crate::DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let impulse_index = 3usize;
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = spectrum
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = impulse
                    * crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(forward_error <= 2.0e-15);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(round_trip_error <= 5.0e-14);
    }

    #[test]
    fn cuda_double_double_nested_fft_then_direct_n886_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64 {
            return;
        }
        let length = 2usize * 443;
        let tuning = crate::PlannerTuning::for_device(profile, Precision::DoubleDouble)
            .with_recursive_fft_rader(true);
        let Some(expected_threads) = crate::scheduler::plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
            length,
            &[(443, 1)],
            1,
            tuning,
            profile,
        )
        .unwrap() else {
            return;
        };
        if profile.max_threads_per_block >= 1024 && profile.max_workgroup_size[0] >= 1024 {
            assert_eq!(expected_threads, 74);
        }
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N886 should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N886 should keep smooth-2 x p443 Cooley root");
            };
            assert_eq!(
                root.pack_right
                    .axis_batch_block
                    .unwrap()
                    .threads_per_transform,
                expected_threads
            );
            let outer = match (&root.left, &root.right) {
                (
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                        rader,
                    ),
                    _,
                ) if rader.prime == 443 => rader,
                (
                    _,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                        rader,
                    ),
                ) if rader.prime == 443 => rader,
                _ => panic!("CUDA DD N886 should contain p443 FFT-Rader child"),
            };
            let crate::double_double_ir::DoubleDoubleBluesteinConvolutionIr::Recursive(conv) =
                &outer.forward_fft
            else {
                panic!("CUDA DD p443 convolution should remain recursive 442=2x13x17");
            };
            fn contains_fft_17(
                node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            ) -> bool {
                match node {
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.prime == 17,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => contains_fft_17(&cooley.left) || contains_fft_17(&cooley.right),
                    _ => false,
                }
            }
            fn contains_direct_13(
                node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            ) -> bool {
                match node {
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == 13,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => contains_direct_13(&cooley.left) || contains_direct_13(&cooley.right),
                    _ => false,
                }
            }
            assert!(contains_fft_17(&conv.root));
            assert!(contains_direct_13(&conv.root));
        }
        let impulse = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_parts(1.25, 3.0e-31),
            crate::DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let impulse_index = 3usize;
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = spectrum
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = impulse
                    * crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(forward_error <= 5.0e-13);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(round_trip_error <= 2.0e-12);
    }

    #[test]
    fn cuda_double_double_nested_custom_tuning_n566_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64 {
            return;
        }
        let length = 2usize * 283;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        tuning.max_rader_direct_prime = 47;
        tuning.validate().unwrap();
        let Some(expected_threads) = crate::scheduler::plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
            length,
            &[(283, 1)],
            batch_count,
            tuning,
            profile,
        )
        .unwrap() else {
            return;
        };
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                profile,
            )
            .unwrap()
        };
        let forward = build(Precision::DoubleDouble, Direction::Forward);
        let inverse = build(Precision::DoubleDouble, Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD custom N566 should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD custom N566 should keep smooth-2 x p283 Cooley root");
            };
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, expected_threads);
            let outer = match (&root.left, &root.right) {
                (
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                        rader,
                    ),
                    _,
                ) if rader.prime == 283 => rader,
                (
                    _,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(
                        rader,
                    ),
                ) if rader.prime == 283 => rader,
                _ => panic!("CUDA DD custom N566 should contain p283 FFT-Rader child"),
            };
            let crate::double_double_ir::DoubleDoubleBluesteinConvolutionIr::Recursive(conv) =
                &outer.forward_fft
            else {
                panic!("CUDA DD custom p283 convolution should remain recursive");
            };
            fn contains_fft_prime(
                node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
                prime: usize,
            ) -> bool {
                match node {
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.prime == prime,
                    crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => contains_fft_prime(&cooley.left, prime) || contains_fft_prime(&cooley.right, prime),
                    _ => false,
                }
            }
            assert!(contains_fft_prime(&conv.root, 47));
        }
        let mut input = vec![crate::ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(1.0),
                crate::DoubleDouble::default(),
            );
        }
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(forward_error <= 5.0e-17);
        let expected_restored = inverse.execute_double_double_reference(&expected).unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(inverse_error <= 5.0e-17);

        let f64_forward = build(Precision::DoubleDoubleF64Storage, Direction::Forward);
        let f64_inverse = build(Precision::DoubleDoubleF64Storage, Direction::Inverse);
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(f64_forward_error <= 3.0e-11 * length as f64);
        let f64_expected_restored = f64_inverse
            .execute_complex_reference(&f64_expected)
            .unwrap();
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_inverse_error = f64_restored
            .iter()
            .zip(&f64_expected_restored)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(f64_inverse_error <= 3.0e-11 * length as f64);
    }

    #[test]
    fn cuda_double_double_nested_sub_rader_composite_n214_parent_floor_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 2usize * 107;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Precision::DoubleDouble, Direction::Forward);
        let inverse = build(Precision::DoubleDouble, Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
                transform
            else {
                panic!("CUDA DD N214 nested sub-Rader should remain recursive");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &ir.root
            else {
                panic!("CUDA DD N214 should keep smooth-2 x p107 Cooley root");
            };
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 18);
            assert_eq!([block.local_size_x, block.local_size_y], [18, 1]);
        }
        let mut input = vec![crate::ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] = crate::ComplexDoubleDouble::new(
                crate::DoubleDouble::from_f64(1.0),
                crate::DoubleDouble::default(),
            );
        }
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(forward_error <= 5.0e-18);
        let expected_restored = inverse.execute_double_double_reference(&expected).unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(inverse_error <= 5.0e-18);

        let f64_forward = build(Precision::DoubleDoubleF64Storage, Direction::Forward);
        let f64_inverse = build(Precision::DoubleDoubleF64Storage, Direction::Inverse);
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(f64_forward_error <= 3.0e-11 * length as f64);
        let f64_expected_restored = f64_inverse
            .execute_complex_reference(&f64_expected)
            .unwrap();
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_inverse_error = f64_restored
            .iter()
            .zip(&f64_expected_restored)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(f64_inverse_error <= 3.0e-11 * length as f64);
    }

    #[test]
    fn cuda_double_double_nested_fft_rader_p107_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 107usize;
        let batch_count = 3usize;
        let grouped_batch = 2usize;
        let left = 17usize;
        let right = 29usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let config = |precision, inverse_normalization| {
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_inverse_normalization(inverse_normalization)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, left, right)
                .unwrap()
        };
        let forward = TransformIr::build(
            config(Precision::DoubleDouble, false),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            config(Precision::DoubleDouble, true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(rader)) =
                transform
            else {
                panic!("CUDA p107 recursive-Rader opt-in should keep an FFT-Rader caller");
            };
            let crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) = &rader.forward_fft
            else {
                panic!("CUDA p107 should use a recursive 106-point DD convolution child");
            };
            assert_eq!(child.logical_len, 106);
            assert_eq!(child.grouped_batch, grouped_batch);
            assert!(child.zero_pad_pass.is_none());
            assert!(matches!(
                rader.inverse_fft,
                crate::DoubleDoubleBluesteinConvolutionIr::Recursive(_)
            ));
        }

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.021 * x).sin() + 0.00011 * x,
                        (index + 1) as f64 * 4.0e-32,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.017 * x).cos() - 0.00007 * x,
                        -(index as f64 + 1.0) * 3.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-20,
            "CUDA nested DD p107 FFT-Rader forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let expected_restored = inverse.execute_double_double_reference(&expected).unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 2.0e-18,
            "CUDA nested DD p107 FFT-Rader inverse mismatch on {}: {inverse_error:e}",
            context.device_name()
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + left..base + right]
                    .iter()
                    .all(|value| *value == crate::ComplexDoubleDouble::default())
            );
        }

        let f64_forward = TransformIr::build(
            config(Precision::DoubleDoubleF64Storage, false),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_inverse = TransformIr::build(
            config(Precision::DoubleDoubleF64Storage, true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 3.0e-11 * length as f64,
            "CUDA nested DD/F64 p107 forward mismatch on {}: {f64_forward_error:e}",
            context.device_name()
        );
        let f64_expected_restored = f64_inverse
            .execute_complex_reference(&f64_expected)
            .unwrap();
        let f64_restored = context
            .execute_transform_double_double_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_inverse_error = f64_restored
            .iter()
            .zip(&f64_expected_restored)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_inverse_error <= 3.0e-11 * length as f64,
            "CUDA nested DD/F64 p107 inverse mismatch on {}: {f64_inverse_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_device_scored_fft_rader_p67_nested_bluestein_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 67usize;
        let batch_count = 3usize;
        let grouped_batch = 2usize;
        let left = 9usize;
        let right = 17usize;
        let config = |inverse_normalization| {
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(inverse_normalization)
                .with_tuning(crate::PlannerTuning::portable())
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, left, right)
                .unwrap()
        };
        let forward =
            TransformIr::build(config(false), Direction::Forward, context.device_profile())
                .unwrap();
        let inverse =
            TransformIr::build(config(true), Direction::Inverse, context.device_profile()).unwrap();
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(rader)) =
                transform
            else {
                panic!("CUDA p67 should retain the outer DD FFT-Rader caller");
            };
            let crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) = &rader.forward_fft
            else {
                panic!("CUDA p67 should device-score its 66-point child to nested Bluestein");
            };
            assert_eq!(child.logical_len, 66);
            assert!(child.zero_padding.is_none());
        }

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.023 * x).sin() + 0.00017 * x,
                        (index + 1) as f64 * 4.0e-32,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.015 * x).cos() - 0.00009 * x,
                        -(index as f64 + 1.0) * 3.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-20,
            "CUDA device-scored DD p67 forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let expected_restored = inverse.execute_double_double_reference(&expected).unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 2.0e-18,
            "CUDA device-scored DD p67 inverse mismatch on {}: {inverse_error:e}",
            context.device_name()
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + left..base + right]
                    .iter()
                    .all(|value| *value == crate::ComplexDoubleDouble::default())
            );
        }
    }

    #[test]
    fn double_double_fft_rader_p4159_crosses_old_4096_gate_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 4159usize;
        let batch_count = 2usize;
        let grouped_batch = 2usize;
        let left = 1024usize;
        let right = 1536usize;
        let config = |precision| {
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_tuning(crate::PlannerTuning::portable())
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, left, right)
                .unwrap()
        };

        let forward = TransformIr::Complex1dDoubleDouble(
            crate::DoubleDoubleOneDimIr::build(
                &crate::FftPlan::build(config(Precision::DoubleDouble)).unwrap(),
                Direction::Forward,
            )
            .unwrap(),
        );
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(rader)) =
            &forward
        else {
            panic!("CUDA p4159 should remain standalone DD FFT-Rader");
        };
        assert_eq!(rader.convolution_len, 4158);
        assert!(rader.forward_fft.sequence_len() > 4096);
        assert!(rader.zero_pad_pass.is_some());
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.0031 * x).sin() + 0.00001 * x,
                        (index + 1) as f64 * 6.0e-32,
                    ),
                    DoubleDouble::from_parts(
                        (0.0023 * x).cos() - 0.000007 * x,
                        -(index as f64 + 1.0) * 4.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: ComplexDoubleDouble, expected: ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 2.0e-15,
            "CUDA p4159 padded DD FFT-Rader mismatch: {error:e}"
        );

        let f64_ir = TransformIr::Complex1dDoubleDouble(
            crate::DoubleDoubleOneDimIr::build(
                &crate::FftPlan::build(config(Precision::DoubleDoubleF64Storage)).unwrap(),
                Direction::Forward,
            )
            .unwrap(),
        );
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(f64_rader)) =
            &f64_ir
        else {
            panic!("CUDA p4159 DD/F64 should remain standalone FFT-Rader");
        };
        assert_eq!(f64_rader.convolution_len, 4158);
        let f64_input = input
            .iter()
            .copied()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-9,
            "CUDA p4159 padded DD/F64 FFT-Rader mismatch: {f64_error:e}"
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_direct_rader_p47_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 47usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let expected_block = crate::scheduler::plan_gpu_axis0_direct_rader_batch_block(
            length,
            batch_count,
            32,
            false,
            Some(grouped_batch),
            context.device_profile(),
        )
        .unwrap();
        let Some(expected_block) =
            expected_block.filter(|block| block.grouped_batch == grouped_batch)
        else {
            return;
        };
        let config = |precision, inverse_normalization| {
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_inverse_normalization(inverse_normalization)
                .with_tuning(crate::PlannerTuning::portable())
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
        };
        let forward = TransformIr::build(
            config(Precision::DoubleDouble, false),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            config(Precision::DoubleDouble, true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::DirectRader(ir)) =
                transform
            else {
                panic!("grouped CUDA DD p47 should remain direct Rader");
            };
            assert_eq!(ir.axis_batch_block, Some(expected_block));
            assert_eq!(ir.batch_group_count(), 3);
        }
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.031 * x).sin(),
                        (index + 1) as f64 * 1.0e-30,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.017 * x).cos(),
                        -(index as f64 + 1.0) * 5.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error <= 2.0e-24,
            "CUDA DD direct-Rader forward mismatch on {}: {error:e}",
            context.device_name()
        );
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-23,
            "CUDA DD direct-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            config(Precision::DoubleDoubleF64Storage, false),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::DirectRader(
            f64_direct,
        )) = &f64_ir
        else {
            panic!("grouped CUDA DD/F64 p47 should remain direct Rader");
        };
        assert_eq!(f64_direct.axis_batch_block, Some(expected_block));
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 2.0e-12 * length as f64,
            "CUDA DD/F64 direct-Rader mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_real_gpu_double_double_stockham_preserves_residual_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }

        let length = 4usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let base = crate::DoubleDouble::from_f64(1.0e16);
        let mut input =
            vec![crate::ComplexDoubleDouble::new(base, crate::DoubleDouble::ZERO); length];
        input[0].re += crate::DoubleDouble::ONE;
        let spectrum = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dc_offset = spectrum[0].re - crate::DoubleDouble::from_f64(4.0e16);
        assert!((dc_offset - crate::DoubleDouble::ONE).abs().to_f64().abs() <= 1.0e-28);
        let restored = context
            .execute_transform_double_double(&inverse, &spectrum)
            .unwrap();
        for (actual, expected) in restored.iter().zip(&input) {
            assert!((actual.re - expected.re).abs().to_f64().abs() <= 1.0e-28);
            assert!((actual.im - expected.im).abs().to_f64().abs() <= 1.0e-28);
        }

        let length = 15usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.071 * x).sin() + x * 0.0002, (0.029 * x).cos())
            })
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double_f64_storage(&ir, &input)
            .unwrap();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            error <= 3.0e-13 * length as f64,
            "CUDA DD/F64 error {error:e}"
        );
    }

    #[test]
    fn cuda_double_double_grouped_quad_xy_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for (length, expected_y) in [(16usize, 4usize), (15, 5), (72, 12)] {
            let ir = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(7)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::Complex1dDoubleDouble(one_dim) = &ir else {
                panic!("grouped CUDA DD N={length} should build a 1D DD transform");
            };
            let crate::DoubleDoubleOneDimIr::Stockham(stockham) = one_dim else {
                panic!("grouped CUDA DD N={length} should remain Stockham");
            };
            let block = stockham.axis_batch_block.unwrap();
            assert_eq!(block.local_size_x, 3);
            assert_eq!(block.local_size_y, expected_y);
            let input = (0..length * 7)
                .map(|index| {
                    let x = index as f64;
                    crate::ComplexDoubleDouble::from_complex64(Complex64::new(
                        (0.071 * x).sin() + 0.0002 * x,
                        (0.029 * x).cos() - 0.0001 * x,
                    ))
                })
                .collect::<Vec<_>>();
            let expected = crate::execute_double_double_one_dim_ir(one_dim, &input).unwrap();
            let actual = context
                .execute_transform_double_double(&ir, &input)
                .unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual.re - expected.re).abs().to_f64().abs() <= 1.0e-24);
                assert!((actual.im - expected.im).abs().to_f64().abs() <= 1.0e-24);
            }
        }
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_n1922_three_upload_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.vendor != crate::GpuVendor::Nvidia {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 1024;
        device.shared_memory_pow2_bytes = 1024;
        device.max_threads_per_block = device.max_threads_per_block.min(32);
        let length = 2usize * 31 * 31;
        let dimensions = vec![length, 2usize];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone()).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [1922,2] must use ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleOneDimIr::Recursive(recursive) = &outer.transform else {
            panic!("CUDA DD [1922,2] higher axis must use forced-Rader recursive IR");
        };
        assert_eq!(
            recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA DD [1922,2] forced-Rader schedule")
                .axis_split,
            vec![2, 31, 31]
        );
        let components = recursive
            .forced_rader_three_upload_mapped_components()
            .unwrap()
            .expect("CUDA DD [1922,2] should expose mapped three-upload components");
        assert!(components.iter().all(|component| match component {
            crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham { ir, .. } =>
                ir.axis_batch_block.is_some_and(|block| block.transforms_on_x),
            crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive { ir, .. } => match ir {
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(ir) =>
                    ir.axis_batch_block.is_some_and(|block| block.transforms_on_x),
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(ir) =>
                    ir.axis_batch_block.is_some_and(|block| block.transforms_on_x),
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::FftRader(ir) =>
                    ir.caller_axis_batch_block.is_some_and(|block| block.transforms_on_x),
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Bluestein(ir) =>
                    ir.wrapper_axis_batch_block().is_some_and(|block| block.transforms_on_x),
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) =>
                    ir.pack_right.axis_batch_block.is_some_and(|block| block.transforms_on_x),
            },
        }));

        let mut input = vec![crate::ComplexDoubleDouble::default(); length * 2];
        input[2] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_f64(1.0),
            crate::DoubleDouble::default(),
        );
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let mut forward_error = 0.0f64;
        for k0 in 0..length {
            let expected = crate::double_double_unit_root(k0, length, Direction::Forward).unwrap();
            for k1 in 0..2 {
                let value = actual[k0 * 2 + k1];
                let re = (value.re - expected.re).abs();
                let im = (value.im - expected.im).abs();
                forward_error =
                    forward_error.max(re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs());
            }
        }
        assert!(
            forward_error <= 5.0e-10,
            "CUDA DD [1922,2] higher-axis three-upload mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-12,
            "CUDA DD [1922,2] higher-axis three-upload round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_p2053_recursive_bluestein_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 || actual_device.shared_memory_bytes < 48 * 1024 {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let dimensions = vec![2_053usize, 2];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [2053,2] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
            panic!("CUDA DD [2053,2] higher axis should use Bluestein");
        };
        assert_eq!(bluestein.convolution_len, 4_368);
        assert!(
            bluestein
                .wrapper_axis_batch_block()
                .is_some_and(|block| block.transforms_on_x)
        );
        let crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) = &bluestein.forward_fft
        else {
            panic!("CUDA DD [2053,2] M4368 convolution should be recursive");
        };
        let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) =
            &child.root
        else {
            panic!("CUDA DD [2053,2] M4368 root should remain Cooley-Tukey");
        };
        assert!(
            root.pack_right
                .axis_batch_block
                .is_some_and(|block| block.transforms_on_x)
        );

        let input = (0..2_053 * 2)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00004 * x,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.023 * x).cos() - 0.00003 * x,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = crate::execute_double_double_nd_ir(nd, &input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-15,
            "CUDA DD [2053,2] recursive Bluestein mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 8.0e-14,
            "CUDA DD [2053,2] recursive Bluestein round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_direct_rader_auto_group_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![47usize, 8usize];
        let config = FftConfig::new(dimensions.clone())
            .with_precision(Precision::DoubleDouble)
            .with_tuning(crate::PlannerTuning::portable());
        let forward =
            TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [47,8] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.grouped_batch, 1);
        assert_eq!(outer.grouped_batch_override, None);
        let crate::DoubleDoubleOneDimIr::DirectRader(rader) = &outer.transform else {
            panic!("CUDA DD [47,8] higher axis should use direct Rader");
        };
        let block = rader
            .axis_batch_block
            .expect("CUDA default higher-axis p47 should auto-group");
        assert!(block.grouped_batch > 1);
        assert!(block.transforms_on_x);

        let input = (0..47 * 8)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.037 * x).sin() + 0.0002 * x,
                        (index + 1) as f64 * 3.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.019 * x).cos() - 0.0001 * x,
                        -(index as f64 + 1.0) * 2.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD [47,8] direct-Rader auto-group mismatch: {forward_error:e}"
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(crate::PlannerTuning::portable())
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-16,
            "CUDA DD [47,8] direct-Rader auto-group round trip mismatch: {round_trip_error:e}"
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_fft_rader_auto_group_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![257usize, 8usize];
        let config = FftConfig::new(dimensions.clone())
            .with_precision(Precision::DoubleDouble)
            .with_tuning(crate::PlannerTuning::portable());
        let forward =
            TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [257,8] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.grouped_batch, 1);
        assert_eq!(outer.grouped_batch_override, None);
        let crate::DoubleDoubleOneDimIr::FftRader(rader) = &outer.transform else {
            panic!("CUDA DD [257,8] higher axis should use FFT Rader");
        };
        let block = rader
            .caller_axis_batch_block
            .expect("CUDA default higher-axis p257 should auto-group its caller");
        assert_eq!(block.threads_per_transform, 17);
        assert!(block.grouped_batch > 1);
        assert!(block.transforms_on_x);
        assert_eq!(rader.forward_fft.grouped_batch(), 1);
        assert_eq!(rader.inverse_fft.grouped_batch(), 1);

        let input = (0..257 * 8)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00003 * x,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.021 * x).cos() - 0.00002 * x,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-17,
            "CUDA DD [257,8] FFT-Rader auto-group mismatch: {forward_error:e}"
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(crate::PlannerTuning::portable())
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-15,
            "CUDA DD [257,8] FFT-Rader auto-group round trip mismatch: {round_trip_error:e}"
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_bluestein_auto_group_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let tuning = crate::PlannerTuning {
            min_rader_direct_prime: 17,
            max_rader_direct_prime: 17,
            min_rader_fft_prime: 17,
            max_rader_fft_prime: 1024,
            allow_recursive_fft_rader: false,
        };
        let dimensions = vec![11usize, 8usize];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA forced-Bluestein DD [11,8] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
            panic!("CUDA forced DD higher-axis p11 should use Bluestein");
        };
        assert_eq!(outer.grouped_batch, 1);
        assert_eq!(outer.grouped_batch_override, None);
        let wrapper = bluestein
            .wrapper_axis_batch_block()
            .expect("CUDA default higher-axis Bluestein wrapper should auto-group");
        assert_eq!(wrapper.grouped_batch, 4);
        assert!(wrapper.transforms_on_x);
        assert!(
            bluestein
                .stockham_convolution_axis_batch_block()
                .is_some_and(|block| block.grouped_batch > 1 && block.transforms_on_x)
        );

        let input = (0..11 * 8)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.041 * x).sin(),
                        (index + 1) as f64 * 3.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.029 * x).cos(),
                        -(index as f64 + 1.0) * 2.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD [11,8] Bluestein auto-group mismatch: {forward_error:e}"
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-16,
            "CUDA DD [11,8] Bluestein auto-group round trip mismatch: {round_trip_error:e}"
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_n5100_forced_rader_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = device.max_threads_per_block.min(128);
        let dimensions = vec![5_100usize, 2];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone()).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [5100,2] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleOneDimIr::Recursive(child) = &outer.transform else {
            panic!("CUDA DD [5100,2] higher axis should be recursive");
        };
        assert_eq!(
            child
                .rader_forced_upload_schedule
                .as_ref()
                .expect("CUDA higher-axis N5100 forced-Rader schedule")
                .axis_split,
            vec![68, 75]
        );
        let (mapped_high, _) = child
            .forced_rader_two_upload_mapped_high_stockham()
            .unwrap()
            .expect("CUDA higher-axis N5100 mapped high Stockham");
        assert!(
            mapped_high
                .axis_batch_block
                .is_some_and(|block| block.transforms_on_x)
        );
        let mapped_low = child
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("CUDA higher-axis N5100 mapped low component");
        let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
            mapped_low,
        ) = mapped_low
        else {
            panic!("CUDA higher-axis N5100 mapped low should remain Cooley-Tukey");
        };
        assert!(
            mapped_low
                .pack_right
                .axis_batch_block
                .is_some_and(|block| block.transforms_on_x)
        );

        let input = (0..5_100 * 2)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.011 * x).sin() + 0.00003 * x,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.017 * x).cos() - 0.00002 * x,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = crate::execute_double_double_nd_ir(nd, &input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-18,
            "CUDA DD [5100,2] forced-Rader mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-17,
            "CUDA DD [5100,2] forced-Rader round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_rader_auto_group_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for length in [47usize, 94usize, 257usize] {
            let forward = TransformIr::build(
                FftConfig::new(vec![length, 8usize])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(crate::PlannerTuning::portable()),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
                panic!("CUDA DD [{length},8] should build ND C2C IR");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.grouped_batch, 1);
            assert_eq!(outer.grouped_batch_override, None);
            let block = match &outer.transform {
                crate::DoubleDoubleOneDimIr::DirectRader(rader) => rader.axis_batch_block,
                crate::DoubleDoubleOneDimIr::FftRader(rader) => rader.caller_axis_batch_block,
                crate::DoubleDoubleOneDimIr::Recursive(recursive) if length == 94 => {
                    let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) =
                        &recursive.root
                    else {
                        panic!("CUDA DD N94 higher-axis composite should keep a Cooley root");
                    };
                    assert!(matches!(
                        root.right,
                        crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
                    ));
                    let block = root.pack_right.axis_batch_block;
                    if let Some(block) = block {
                        assert_eq!(block.threads_per_transform, 48);
                        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
                        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
                    }
                    block
                }
                other => panic!("unexpected CUDA DD higher-axis Rader child: {other:?}"),
            }
            .expect("CUDA default DD higher-axis Rader should auto-group");
            assert!(block.grouped_batch > 1);
            assert!(block.transforms_on_x);

            let input = (0..length * 8)
                .map(|index| {
                    let x = index as f64;
                    crate::ComplexDoubleDouble::new(
                        crate::DoubleDouble::from_parts(
                            (0.027 * x).sin() + 0.00005 * x,
                            (index + 1) as f64 * 2.0e-31,
                        ),
                        crate::DoubleDouble::from_parts(
                            (0.019 * x).cos() - 0.00004 * x,
                            -(index as f64 + 1.0) * 1.0e-31,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let expected = crate::execute_double_double_nd_ir(nd, &input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let re = (actual.re - expected.re).abs();
                    let im = (actual.im - expected.im).abs();
                    re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                error <= 5.0e-18,
                "CUDA DD [{length},8] auto-group Rader mismatch: {error:e}"
            );
        }
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_stockham_auto_group_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![64usize, 8usize];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone()).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [64,8] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.grouped_batch, 1);
        assert_eq!(outer.grouped_batch_override, None);
        let crate::DoubleDoubleOneDimIr::Stockham(stockham) = &outer.transform else {
            panic!("CUDA DD [64,8] higher axis should use Stockham");
        };
        let block = stockham
            .axis_batch_block
            .expect("CUDA default DD higher-axis Stockham should auto-group");
        assert!(block.grouped_batch > 1);
        assert!(block.transforms_on_x);

        let input = (0..64 * 8)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.031 * x).sin(),
                        (index + 1) as f64 * 3.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.023 * x).cos(),
                        -(index as f64 + 1.0) * 2.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: crate::ComplexDoubleDouble,
                        expected: crate::ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-18,
            "CUDA DD [64,8] auto-group mismatch: {forward_error:e}"
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-16,
            "CUDA DD [64,8] auto-group round trip mismatch: {round_trip_error:e}"
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_n4116_device_schedule_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        let dimensions = vec![4_116usize, 2];
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone()).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &forward else {
            panic!("CUDA DD [4116,2] should build ND C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleOneDimIr::Recursive(child) = &outer.transform else {
            panic!("CUDA DD [4116,2] higher axis should be recursive");
        };
        assert_eq!(
            child
                .stockham_upload_schedule
                .as_ref()
                .expect("CUDA higher-axis N4116 upload schedule")
                .axis_split,
            vec![84, 49]
        );
        assert!(child.two_upload_four_step_plan.is_some());

        let input = (0..4_116 * 2)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.013 * x).sin(),
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.021 * x).cos(),
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = crate::execute_double_double_nd_ir(nd, &input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-19,
            "CUDA DD [4116,2] mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-18,
            "CUDA DD [4116,2] round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        let f64_forward = TransformIr::build(
            FftConfig::new(dimensions).with_precision(Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device,
        )
        .unwrap();
        let TransformIr::ComplexNdDoubleDouble(f64_nd) = &f64_forward else {
            panic!("CUDA DD/F64 [4116,2] should build ND C2C IR");
        };
        let f64_input = input
            .iter()
            .copied()
            .map(crate::ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected =
            crate::execute_double_double_nd_ir_f64_storage(f64_nd, &f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (actual.re - expected.re).hypot(actual.im - expected.im))
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 5.0e-11,
            "CUDA DD/F64 [4116,2] mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_double_double_nd_higher_axis_n16_quad_xy_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let config = FftConfig::new(vec![16, 8])
            .with_batch_count(5)
            .with_precision(Precision::DoubleDouble)
            .with_grouped_batch(0, 3)
            .unwrap()
            .with_grouped_batch(1, 3)
            .unwrap();
        let ir = TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let TransformIr::ComplexNdDoubleDouble(nd) = &ir else {
            panic!("grouped CUDA DD [16,8] should build multidimensional DD C2C IR");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleOneDimIr::Stockham(stockham) = &outer.transform else {
            panic!("grouped CUDA DD [16,8] outer axis should remain Stockham");
        };
        let block = stockham.axis_batch_block.unwrap();
        assert_eq!([block.local_size_x, block.local_size_y], [3, 4]);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!(stockham.batch_group_count(), 14);
        let program = crate::ProgramIr::double_double_nd(nd).unwrap();
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 2));
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 14));

        let input = (0..16 * 8 * 5)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::from_complex64(Complex64::new(
                    (0.071 * x).sin() + 0.0002 * x,
                    (0.029 * x).cos() - 0.0001 * x,
                ))
            })
            .collect::<Vec<_>>();
        let expected = crate::execute_double_double_nd_ir(nd, &input).unwrap();
        let actual = context
            .execute_transform_double_double(&ir, &input)
            .unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual.re - expected.re).abs().to_f64().abs() <= 1.0e-24);
            assert!((actual.im - expected.im).abs().to_f64().abs() <= 1.0e-24);
        }
    }

    #[test]
    fn cuda_double_double_nd_real_r2r_device_schedule_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64 {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;

        let real_dimensions = vec![4_116usize, 2];
        let r2c = TransformIr::build(
            FftConfig::new(real_dimensions.clone())
                .with_transform(crate::TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let c2r = TransformIr::build(
            FftConfig::new(real_dimensions.clone())
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let TransformIr::RealNdDoubleDouble(r2c_ir) = &r2c else {
            panic!("CUDA DD [4116,2] R2C should build ND-real IR");
        };
        let TransformIr::RealNdDoubleDouble(c2r_ir) = &c2r else {
            panic!("CUDA DD [4116,2] C2R should build ND-real IR");
        };
        let outer = r2c_ir
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let crate::DoubleDoubleOneDimIr::Recursive(child) = &outer.transform else {
            panic!("CUDA DD [4116,2] outer real axis should be recursive N4116");
        };
        assert_eq!(
            child
                .stockham_upload_schedule
                .as_ref()
                .expect("CUDA ND-real N4116 device upload schedule")
                .axis_split,
            vec![84, 49]
        );
        let four_step = child.two_upload_four_step_plan.unwrap();
        assert!(four_step.left_axis_block.transforms_on_x);
        assert!(four_step.right_axis_block.transforms_on_x);
        assert!(
            four_step.left_axis_block.grouped_batch > 1
                || four_step.right_axis_block.grouped_batch > 1
        );
        let real_input = (0..4_116 * 2)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.017 * x).sin() + 0.13 * (0.029 * x).cos() + 0.0002 * x,
                    (index + 1) as f64 * 4.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let real_expected = crate::execute_double_double_nd_r2c_ir(r2c_ir, &real_input).unwrap();
        let real_actual = context
            .execute_double_double_nd_r2c(r2c_ir, &real_input)
            .unwrap();
        let real_forward_error = real_actual
            .iter()
            .copied()
            .zip(real_expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            real_forward_error <= 5.0e-17,
            "CUDA DD [4116,2] device-aware R2C mismatch on {}: {real_forward_error:e}",
            context.device_name()
        );
        let real_restored = context
            .execute_double_double_nd_c2r(c2r_ir, &real_actual)
            .unwrap();
        let real_round_trip_error = real_restored
            .iter()
            .copied()
            .zip(real_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            real_round_trip_error <= 5.0e-16,
            "CUDA DD [4116,2] device-aware real round trip mismatch on {}: {real_round_trip_error:e}",
            context.device_name()
        );

        let r2r_dimensions = vec![2_058usize, 2];
        let transform = crate::TransformKind::Dct(crate::DctType::II);
        let forward = TransformIr::build(
            FftConfig::new(r2r_dimensions.clone())
                .with_transform(transform)
                .with_precision(Precision::DoubleDouble),
            Direction::Forward,
            device,
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(r2r_dimensions.clone())
                .with_transform(transform)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device,
        )
        .unwrap();
        let TransformIr::RealToRealNdDoubleDouble(forward_ir) = &forward else {
            panic!("CUDA DD DCT-II [2058,2] should build ND-R2R IR");
        };
        let outer = forward_ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &outer.transform.algorithm
        else {
            panic!("CUDA DD DCT-II N2058 should use FFT reduction");
        };
        let crate::DoubleDoubleOneDimIr::Recursive(child) = fft.as_ref() else {
            panic!("CUDA DD DCT-II N2058 child should remain recursive N2058");
        };
        assert_eq!(
            child
                .stockham_upload_schedule
                .as_ref()
                .expect("CUDA ND-R2R N2058 device upload schedule")
                .axis_split,
            vec![42, 49]
        );
        let four_step = child.two_upload_four_step_plan.unwrap();
        assert!(four_step.left_axis_block.transforms_on_x);
        assert!(four_step.right_axis_block.transforms_on_x);
        assert!(
            four_step.left_axis_block.grouped_batch > 1
                || four_step.right_axis_block.grouped_batch > 1
        );
        let r2r_program = crate::ProgramIr::double_double_r2r(&outer.transform).unwrap();
        assert_eq!(r2r_program.passes.len(), 2);
        assert!(r2r_program.resources.iter().all(|resource| {
            !resource.name.contains("double_double_r2r_fft_input")
                && !resource.name.contains("double_double_r2r_fft_output")
        }));
        let r2r_input = (0..2_058 * 2)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.023 * x).sin() + 0.11 * (0.037 * x).cos() + 0.0003 * x,
                    (index + 1) as f64 * 3.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let r2r_expected = forward
            .execute_double_double_r2r_reference(&r2r_input)
            .unwrap();
        let r2r_actual = context
            .execute_transform_double_double_r2r(&forward, &r2r_input)
            .unwrap();
        let r2r_forward_error = r2r_actual
            .iter()
            .copied()
            .zip(r2r_expected.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            r2r_forward_error <= 5.0e-17,
            "CUDA DD DCT-II [2058,2] device-aware mismatch on {}: {r2r_forward_error:e}",
            context.device_name()
        );
        let r2r_restored = context
            .execute_transform_double_double_r2r(&inverse, &r2r_actual)
            .unwrap();
        let r2r_round_trip_error = r2r_restored
            .iter()
            .copied()
            .zip(r2r_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            r2r_round_trip_error <= 5.0e-16,
            "CUDA DD DCT-II [2058,2] device-aware round trip mismatch on {}: {r2r_round_trip_error:e}",
            context.device_name()
        );

        let f64_build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![2_058usize])
                    .with_transform(transform)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let f64_forward = f64_build(Direction::Forward);
        let f64_inverse = f64_build(Direction::Inverse);
        for transform in [&f64_forward, &f64_inverse] {
            let TransformIr::RealToRealDoubleDouble(ir) = transform else {
                panic!("CUDA DD/F64 DCT-II N2058 should build 1D R2R IR");
            };
            assert_eq!(
                crate::ProgramIr::double_double_r2r(ir)
                    .unwrap()
                    .passes
                    .len(),
                2
            );
        }
        let f64_input = r2r_input[..2_058]
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 2.0e-10,
            "CUDA DD/F64 DCT-II N2058 two-upload error {f64_forward_error:e}"
        );
        let f64_restored = context
            .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 2.0e-11,
            "CUDA DD/F64 DCT-II/III N2058 two-upload round-trip error {f64_round_trip_error:e}"
        );

        let dst_build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![2_058usize])
                    .with_transform(crate::TransformKind::Dst(crate::DstType::II))
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let dst_forward = dst_build(Direction::Forward);
        let dst_inverse = dst_build(Direction::Inverse);
        for transform in [&dst_forward, &dst_inverse] {
            let TransformIr::RealToRealDoubleDouble(ir) = transform else {
                panic!("CUDA DD DST-II N2058 should build 1D R2R IR");
            };
            assert_eq!(
                crate::ProgramIr::double_double_r2r(ir)
                    .unwrap()
                    .passes
                    .len(),
                2
            );
        }
        let dst_input = r2r_input[..2_058].to_vec();
        let dst_expected = dst_forward
            .execute_double_double_r2r_reference(&dst_input)
            .unwrap();
        let dst_actual = context
            .execute_transform_double_double_r2r(&dst_forward, &dst_input)
            .unwrap();
        let dst_forward_error = dst_actual
            .iter()
            .copied()
            .zip(dst_expected.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            dst_forward_error <= 5.0e-17,
            "CUDA DD DST-II N2058 two-upload error {dst_forward_error:e}"
        );
        let dst_restored = context
            .execute_transform_double_double_r2r(&dst_inverse, &dst_actual)
            .unwrap();
        let dst_round_trip_error = dst_restored
            .iter()
            .copied()
            .zip(dst_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            dst_round_trip_error <= 5.0e-16,
            "CUDA DD DST-II/III N2058 two-upload round-trip error {dst_round_trip_error:e}"
        );
    }

    #[test]
    fn cuda_double_double_nd_real_outer_n16_quad_xy_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let config = FftConfig::new(vec![16, 8])
            .with_batch_count(5)
            .with_precision(Precision::DoubleDouble)
            .with_transform(crate::TransformKind::RealToComplex)
            .with_grouped_batch(0, 3)
            .unwrap()
            .with_grouped_batch(1, 3)
            .unwrap();
        let ir = TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let TransformIr::RealNdDoubleDouble(nd) = &ir else {
            panic!("grouped CUDA DD [16,8] R2C should build multidimensional DD real IR");
        };
        let crate::DoubleDoubleOneDimIr::Stockham(real_stockham) = &nd.real_axis.transform else {
            panic!("grouped CUDA DD [16,8] real fastest half-size child should remain Stockham");
        };
        let real_block = real_stockham.axis_batch_block.unwrap();
        assert_eq!([real_block.local_size_x, real_block.local_size_y], [3, 1]);
        assert!(real_block.transforms_on_x);
        assert!(real_block.axis_swapped);
        assert_eq!(real_stockham.batch_group_count(), 27);
        let crate::DoubleDoubleOneDimIr::Stockham(stockham) = &nd.complex_axes[0].transform else {
            panic!("grouped CUDA DD [16,8] real outer axis should remain Stockham");
        };
        let block = stockham.axis_batch_block.unwrap();
        assert_eq!([block.local_size_x, block.local_size_y], [3, 4]);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!(stockham.batch_group_count(), 9);
        let program = crate::ProgramIr::double_double_nd_real(nd).unwrap();
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 2));
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 9));
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 27));

        let input = (0..16 * 8 * 5)
            .map(|index| {
                let x = index as f64;
                crate::DoubleDouble::from_parts(
                    (0.061 * x).sin() + 0.17 * (0.023 * x).cos() + 0.0003 * x,
                    (index as f64 + 1.0) * 1.0e-24,
                )
            })
            .collect::<Vec<_>>();
        let expected = crate::execute_double_double_nd_r2c_ir(nd, &input).unwrap();
        let actual = context.execute_double_double_nd_r2c(nd, &input).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual.re - expected.re).abs().to_f64().abs() <= 1.0e-24);
            assert!((actual.im - expected.im).abs().to_f64().abs() <= 1.0e-24);
        }
    }

    #[test]
    fn cuda_double_double_true_real_quad_xy_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        for (length, child_len, expected_y) in [(15usize, 15usize, 5usize), (32, 32, 4)] {
            let r2c = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(7)
                    .with_transform(crate::TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let c2r = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(7)
                    .with_transform(crate::TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::RealDoubleDouble(r2c_ir) = &r2c else {
                panic!("grouped CUDA DD R2C N={length} should build real DD IR");
            };
            let TransformIr::RealDoubleDouble(c2r_ir) = &c2r else {
                panic!("grouped CUDA DD C2R N={length} should build real DD IR");
            };
            assert_eq!(r2c_ir.transform.sequence_len(), child_len);
            assert_eq!(c2r_ir.transform.sequence_len(), child_len);
            for real in [r2c_ir, c2r_ir] {
                let block = real.stockham_axis_batch_block().unwrap();
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, expected_y]);
                assert!(block.transforms_on_x);
                assert!(block.axis_swapped);
            }

            let input = (0..length * 7)
                .map(|index| {
                    let x = index as f64;
                    crate::DoubleDouble::from_parts(
                        (0.071 * x).sin() + 0.19 * (0.027 * x).cos() + 0.0004 * x,
                        (index as f64 + 1.0) * 1.0e-25,
                    )
                })
                .collect::<Vec<_>>();
            let expected = crate::execute_double_double_r2c_ir(r2c_ir, &input).unwrap();
            let actual = context.execute_double_double_r2c(r2c_ir, &input).unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let re = (actual.re - expected.re).abs();
                    let im = (actual.im - expected.im).abs();
                    re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 2.0e-17,
                "CUDA DD true-real N={length} R2C mismatch: {forward_error:e}"
            );
            let restored = context.execute_double_double_c2r(c2r_ir, &actual).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 2.0e-16,
                "CUDA DD true-real N={length} roundtrip mismatch: {round_trip_error:e}"
            );
        }
    }

    #[test]
    fn cuda_probe_is_fail_soft() {
        let availability = CudaExecutionContext::probe();
        assert_eq!(availability.backend, Backend::Cuda);
        if availability.available() {
            assert!(availability.device_count > 0);
            let context = new_test_context(0).unwrap();
            let versions = context.nvrtc_versions();
            assert!(!versions.is_empty());
            assert!(versions.windows(2).all(|pair| pair[0] >= pair[1]));
            assert!(versions.iter().copied().any(|version| {
                nvrtc_version_is_driver_compatible(version, context.driver_cuda_version())
            }));
        }
    }

    #[test]
    fn cuda_driver_version_and_nvrtc_compatibility_cover_legacy_pascal_toolchains() {
        assert_eq!(decode_cuda_driver_version(8000).unwrap(), (8, 0));
        assert_eq!(decode_cuda_driver_version(9010).unwrap(), (9, 1));
        assert_eq!(decode_cuda_driver_version(12080).unwrap(), (12, 8));
        assert!(decode_cuda_driver_version(0).is_err());
        assert!(NVRTC_LIBRARY_CANDIDATES.contains(&"libnvrtc.so.8.0"));
        assert!(NVRTC_LIBRARY_CANDIDATES.contains(&"libnvrtc.so.13"));
        assert!(nvrtc_version_is_driver_compatible((8, 0), (9, 1)));
        assert!(nvrtc_version_is_driver_compatible((9, 1), (9, 1)));
        assert!(!nvrtc_version_is_driver_compatible((13, 0), (9, 1)));
        let diagnostic = nvrtc_compile_failure_message(
            (8, 0),
            "libnvrtc.so.8.0",
            7,
            "nvrtc: error: failed to load builtins",
        );
        assert!(diagnostic.contains("same CUDA toolkit"));
        assert!(diagnostic.contains("LD_LIBRARY_PATH"));
    }

    #[test]
    fn cuda_axis0_swapped_batched_stockham_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 64usize;
        let batch_count = 32usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_batch_count(batch_count),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("64-point batched CUDA transform should be a recursive Stockham IR");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("64-point batched CUDA transform should be one Stockham root");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 16);
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [16, 8]);
        assert_eq!(kernel.dispatch.x, 2);

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.071 * x).sin() + 0.0003 * x,
                    (0.037 * x).cos() - 0.0002 * x,
                )
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 4.0e-4 * length as f64);
    }

    #[test]
    fn cuda_axis0_swapped_smooth_batched_stockham_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 18usize;
        let batch_count = 32usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_batch_count(batch_count),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("18-point batched CUDA transform should be a recursive Stockham IR");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("18-point batched CUDA transform should be one Stockham root");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 32);
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [32, 3]);
        assert_eq!(kernel.dispatch.x, 1);
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.091 * x).sin() + 0.0002 * x, (0.053 * x).cos())
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 5.0e-4 * length as f64);
    }

    #[test]
    fn cuda_zero_padded_axis0_grouping_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 64usize;
        let batch_count = 32usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_zero_padding(0, 16, 32)
            .unwrap();
        let ir = TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("zero-padded batched CUDA transform should use recursive Stockham IR");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("zero-padded batched CUDA transform should remain one Stockham root");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 16);
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::ThreadsXTransformsY
        );
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [8, 16]);
        assert_eq!(kernel.dispatch.x, 2);

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.067 * x).sin() + 0.0003 * x, (0.041 * x).cos())
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 5.0e-4 * length as f64);
    }

    #[test]
    fn cuda_zero_padded_smooth_rader_axis_grouping_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 19usize;
        let batch_count = 32usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_zero_padding(0, 5, 12)
            .unwrap();
        let ir = TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("zero-padded p19 CUDA transform should use recursive Rader IR");
        };
        let crate::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
            panic!("zero-padded p19 CUDA transform should keep an FFT-Rader root");
        };
        let block = rader
            .axis_batch_block
            .expect("zero-padded p19 should keep batching");
        assert_eq!(block.grouped_batch, 32);
        assert_eq!(block.threads_per_transform, 4);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [4, 32]);

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.071 * x).sin() + 0.0002 * x, (0.047 * x).cos())
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 8.0e-4 * length as f64);
    }

    #[test]
    fn cuda_grouped_four_step_multi_upload_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 49_152usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length]),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("49,152-point CUDA transform should use recursive Stockham IR");
        };
        let four_step = recursive
            .four_step_plan
            .as_ref()
            .expect("49,152-point CUDA transform should use multi-upload Four-step");
        assert!(four_step.uploads.len() >= 2);
        assert!(
            four_step
                .uploads
                .iter()
                .all(|upload| upload.axis_block.is_some())
        );
        let kernels = recursive
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        assert!(
            kernels
                .iter()
                .all(|kernel| kernel.workgroup_grouping.transforms_per_workgroup > 1)
        );
        assert!(kernels.iter().all(|kernel| kernel.workgroup_size.y > 1));

        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.013 * x).sin() + 0.00001 * x,
                    (0.021 * x).cos() - 0.00002 * x,
                )
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 2.0e-3 * length as f64);
    }

    #[test]
    fn cuda_forced_rader_four_step_matches_sparse_oracle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia
            || profile.max_threads_per_block < 64
            || profile.max_workgroup_size[0] < 64
        {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.max_threads_per_block = 64;
        planning_profile.max_workgroup_size[0] = 64;
        let length = 17usize * 8192;
        let impulse_index = 12_345usize;
        let impulse = Complex32::new(0.375, -0.25);
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[0] = Complex32::new(1.0, 0.0);
        input[impulse_index] = impulse;

        let forward = TransformIr::build(
            FftConfig::new(vec![length]),
            Direction::Forward,
            planning_profile,
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA forced-Rader Four-step probe should remain recursive C2C");
        };
        let schedule = recursive
            .rader_forced_upload_schedule
            .as_ref()
            .expect("CUDA p17-container axis should preserve forced-Rader upload geometry");
        assert_eq!(schedule.axis_split, vec![512, 272]);
        let uploads = recursive
            .four_step_rader_upload_nodes()
            .unwrap()
            .expect("CUDA forced-Rader path should materialize Four-step upload nodes");
        assert_eq!(uploads.len(), 2);
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(high) = &uploads[0] else {
            panic!("CUDA 272-point upload1 should remain 16 x p17 Cooley");
        };
        let block = high.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 16);
        assert_eq!(block.grouped_batch, 4);
        assert!(block.transforms_on_x);
        assert_eq!([block.local_size_x, block.local_size_y], [4, 16]);

        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let impulse64 = Complex64::new(impulse.re as f64, impulse.im as f64);
        for bin in [0usize, 1, 2, 17, 271, 272, 511, 512, length / 2, length - 1] {
            let angle = -core::f64::consts::TAU * (bin * impulse_index) as f64 / length as f64;
            let expected = Complex64::new(1.0, 0.0) + impulse64 * Complex64::exp_i(angle);
            let actual = spectrum[bin];
            assert!(
                (actual.re as f64 - expected.re).abs() <= 1.2e-2
                    && (actual.im as f64 - expected.im).abs() <= 1.2e-2,
                "CUDA forced-Rader Four-step mismatch on {} at bin {bin}: actual={actual:?}, expected={expected:?}",
                context.device_name()
            );
        }

        let inverse = TransformIr::build(
            FftConfig::new(vec![length]).with_inverse_normalization(true),
            Direction::Inverse,
            planning_profile,
        )
        .unwrap();
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| {
                let re = actual.re - expected.re;
                let im = actual.im - expected.im;
                (re * re + im * im).sqrt()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 3.0e-2,
            "CUDA forced-Rader Four-step round-trip mismatch on {}: max_error={max_error:e}",
            context.device_name()
        );

        let pad_left = 12_000usize;
        let pad_right = 12_700usize;
        let kept_index = 54_321usize;
        let kept = Complex32::new(-0.1875, 0.3125);
        let mut mixed_input = vec![Complex32::new(0.0, 0.0); length];
        mixed_input[0] = Complex32::new(1.0, 0.0);
        mixed_input[impulse_index] = impulse;
        mixed_input[kept_index] = kept;
        let mixed_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::F16StorageF32Compute)
                .with_zero_padding(0, pad_left, pad_right)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(mixed_recursive)) = &mixed_forward
        else {
            panic!("mixed forced-Rader CUDA probe should remain recursive");
        };
        assert_eq!(
            mixed_recursive
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![512, 272]
        );
        assert!(
            mixed_recursive
                .four_step_rader_upload_nodes()
                .unwrap()
                .is_some()
        );
        let mixed_spectrum = context
            .execute_transform_complex32(&mixed_forward, &mixed_input)
            .unwrap();
        let kept64 = Complex64::new(kept.re as f64, kept.im as f64);
        for bin in [0usize, 1, 17, 272, 512, length / 2, length - 1] {
            let angle = -core::f64::consts::TAU * (bin * kept_index) as f64 / length as f64;
            let expected = Complex64::new(1.0, 0.0) + kept64 * Complex64::exp_i(angle);
            let expected_re = crate::Binary16::from_f32(expected.re as f32).to_f32();
            let expected_im = crate::Binary16::from_f32(expected.im as f32).to_f32();
            let actual = mixed_spectrum[bin];
            assert!(
                (actual.re - expected_re).abs() <= 5.0e-3
                    && (actual.im - expected_im).abs() <= 5.0e-3,
                "CUDA mixed forced-Rader Four-step mismatch at bin {bin}: actual={actual:?}, expected=({expected_re},{expected_im})"
            );
        }

        let mixed_inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(true)
                .with_zero_padding(0, pad_left, pad_right)
                .unwrap(),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let mut dc = vec![Complex32::new(0.0, 0.0); length];
        dc[0] = Complex32::new(1.0, 0.0);
        let mixed_restored = context
            .execute_transform_complex32(&mixed_inverse, &dc)
            .unwrap();
        let expected_dc = crate::Binary16::from_f32(1.0 / length as f32).to_f32();
        for index in [0usize, 1, pad_left - 1, pad_right, kept_index, length - 1] {
            let actual = mixed_restored[index];
            assert!(
                (actual.re - expected_dc).abs() <= 2.0e-5 && actual.im.abs() <= 2.0e-5,
                "CUDA mixed forced-Rader inverse mismatch at {index}: {actual:?}"
            );
        }
        assert!(
            mixed_restored[pad_left..pad_right]
                .iter()
                .all(|value| value.re == 0.0 && value.im == 0.0)
        );
    }

    #[test]
    fn cuda_direct_rader_forced_upload_matches_sparse_oracle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.shared_memory_bytes = 4 * 1024;
        planning_profile.shared_memory_pow2_bytes = 4 * 1024;
        let length = 17usize * 47 * 128;
        let impulse_index = 31_337usize;
        let dc = Complex32::new(0.875, -0.125);
        let impulse = Complex32::new(-0.3125, 0.21875);
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[0] = dc;
        input[impulse_index] = impulse;

        let forward = TransformIr::build(
            FftConfig::new(vec![length]),
            Direction::Forward,
            planning_profile,
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA direct-Rader forced-upload probe should remain recursive C2C");
        };
        assert_eq!(
            recursive
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![64, 47, 34]
        );
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        assert!(matches!(
            uploads[1],
            crate::recursive_ir::RecursiveFftNodeIr::DirectRader(ref direct)
                if direct.prime == 47
        ));
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
        let impulse64 = Complex64::new(impulse.re as f64, impulse.im as f64);
        for bin in [
            0usize,
            1,
            17,
            33,
            34,
            46,
            47,
            63,
            64,
            1024,
            length / 2,
            length - 1,
        ] {
            let angle = -core::f64::consts::TAU * (bin * impulse_index) as f64 / length as f64;
            let expected = dc64 + impulse64 * Complex64::exp_i(angle);
            let actual = spectrum[bin];
            assert!(
                (actual.re as f64 - expected.re).abs() <= 2.0e-2
                    && (actual.im as f64 - expected.im).abs() <= 2.0e-2,
                "CUDA direct-Rader forced-upload mismatch on {} at bin {bin}: actual={actual:?}, expected={expected:?}",
                context.device_name()
            );
        }

        let inverse = TransformIr::build(
            FftConfig::new(vec![length]).with_inverse_normalization(true),
            Direction::Inverse,
            planning_profile,
        )
        .unwrap();
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 6.0e-2,
            "CUDA direct-Rader forced-upload round-trip mismatch on {}: {max_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn cuda_forced_rader_default_direct_leaf_n8789_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia
            || profile.max_threads_per_block < 384
            || profile.max_workgroup_size[0] < 16
            || profile.max_workgroup_size[1] < 24
        {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.shared_memory_bytes = 8 * 1024;
        planning_profile.shared_memory_pow2_bytes = 8 * 1024;
        let length = 11usize * 17 * 47;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                planning_profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N8789 default forced-Rader route should remain recursive");
            };
            assert_eq!(
                ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
                vec![187, 47]
            );
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(high) = &uploads[0] else {
                panic!("CUDA N8789 high upload should remain p47 Direct Rader");
            };
            let block = high.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 24);
            assert_eq!(block.grouped_batch, 16);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!([block.local_size_x, block.local_size_y], [16, 24]);
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for k in [0usize, 1, 17, 47, 187, length / 2, length - 1] {
            let angle = -std::f32::consts::TAU * k as f32 / length as f32;
            let actual = spectrum[k];
            assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 4.0e-3);
        }
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(error <= 5.0e-3);
    }

    #[test]
    fn cuda_direct_rader_f16_three_upload_mixed_grouped_padding_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.shared_memory_bytes = 8 * 1024;
        planning_profile.shared_memory_pow2_bytes = 8 * 1024;
        let length = 11usize * 17 * 47;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let pad_left = 1000usize;
        let pad_right = 1200usize;
        let kept_index = 3456usize;
        let dropped_index = 1100usize;
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        let mut probes = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let dc = Complex32::new(0.75 + 0.03125 * batch as f32, -0.125);
            let kept = Complex32::new(-0.3125, 0.1875 - 0.015625 * batch as f32);
            let dropped = Complex32::new(0.21875, -0.15625);
            let base = batch * length;
            input[base] = dc;
            input[base + kept_index] = kept;
            input[base + dropped_index] = dropped;
            probes.push((dc, kept, dropped));
        }
        let build = |direction: Direction, padded: bool| {
            let mut config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse);
            if padded {
                config = config.with_zero_padding(0, pad_left, pad_right).unwrap();
            }
            TransformIr::build(config, direction, planning_profile).unwrap()
        };

        let unpadded = build(Direction::Forward, false);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &unpadded else {
            panic!("CUDA mixed direct-Rader probe should remain recursive");
        };
        let schedule = recursive.rader_forced_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.axis_split, vec![47, 17, 11]);
        assert_eq!(
            schedule.reason,
            crate::scheduler::RaderUploadReason::CapacityOrBandwidth
        );
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        assert_eq!(uploads.len(), 3);
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(high) = &uploads[0] else {
            panic!("CUDA 11-point high upload should remain Stockham");
        };
        assert_eq!(high.bindings[0].scalar, crate::ScalarType::F16);
        let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) = &uploads[2] else {
            panic!("CUDA 47-point low upload should be direct Rader");
        };
        assert_eq!(direct.prime, 47);
        assert_eq!(direct.input_storage_scalar, crate::ScalarType::F32);
        assert_eq!(direct.output_storage_scalar, crate::ScalarType::F16);
        assert!(direct.axis_batch_block.is_some());
        let unpadded_spectrum = context
            .execute_transform_complex32(&unpadded, &input)
            .unwrap();
        for (batch, &(dc, kept, dropped)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let kept64 = Complex64::new(kept.re as f64, kept.im as f64);
            let dropped64 = Complex64::new(dropped.re as f64, dropped.im as f64);
            for bin in [0usize, 1, 17, 47, 187, length / 2, length - 1] {
                let a = -core::f64::consts::TAU * (bin * kept_index) as f64 / length as f64;
                let b = -core::f64::consts::TAU * (bin * dropped_index) as f64 / length as f64;
                let expected =
                    dc64 + kept64 * Complex64::exp_i(a) + dropped64 * Complex64::exp_i(b);
                let actual = unpadded_spectrum[batch * length + bin];
                assert!(
                    (actual.re as f64 - expected.re).abs() <= 1.5e-2
                        && (actual.im as f64 - expected.im).abs() <= 1.5e-2,
                    "CUDA mixed direct-Rader unpadded mismatch batch={batch} bin={bin}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let padded = build(Direction::Forward, true);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &padded else {
            panic!("CUDA padded direct-Rader probe should remain recursive");
        };
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(high) = &uploads[0] else {
            panic!("CUDA padded 11-point high upload should remain Stockham");
        };
        assert_eq!(high.bindings[0].scalar, crate::ScalarType::F32);
        let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) = &uploads[2] else {
            panic!("CUDA padded 47-point low upload should be direct Rader");
        };
        assert_eq!(direct.prime, 47);
        assert_eq!(direct.input_storage_scalar, crate::ScalarType::F32);
        assert_eq!(direct.output_storage_scalar, crate::ScalarType::F16);
        let padded_spectrum = context
            .execute_transform_complex32(&padded, &input)
            .unwrap();
        for (batch, &(dc, kept, _)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let kept64 = Complex64::new(kept.re as f64, kept.im as f64);
            for bin in [0usize, 1, 17, 47, 187, length / 2, length - 1] {
                let angle = -core::f64::consts::TAU * (bin * kept_index) as f64 / length as f64;
                let expected = dc64 + kept64 * Complex64::exp_i(angle);
                let actual = padded_spectrum[batch * length + bin];
                assert!(
                    (actual.re as f64 - expected.re).abs() <= 1.5e-2
                        && (actual.im as f64 - expected.im).abs() <= 1.5e-2,
                    "CUDA mixed direct-Rader padded mismatch batch={batch} bin={bin}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let inverse = build(Direction::Inverse, true);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &inverse else {
            panic!("CUDA inverse direct-Rader probe should remain recursive");
        };
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(high) = &uploads[0] else {
            panic!("CUDA inverse 11-point high upload should remain Stockham");
        };
        assert_eq!(high.bindings[0].scalar, crate::ScalarType::F16);
        let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) = &uploads[2] else {
            panic!("CUDA inverse 47-point low upload should be direct Rader");
        };
        assert_eq!(direct.prime, 47);
        assert_eq!(direct.output_storage_scalar, crate::ScalarType::F32);
        let restored = context
            .execute_transform_complex32(&inverse, &padded_spectrum)
            .unwrap();
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + pad_left..base + pad_right]
                    .iter()
                    .all(|value| value.re == 0.0 && value.im == 0.0)
            );
            for index in [0usize, kept_index, length - 1] {
                let expected = input[base + index];
                let actual = restored[base + index];
                assert!(
                    (actual.re - expected.re).abs() <= 2.5e-2
                        && (actual.im - expected.im).abs() <= 2.5e-2,
                    "CUDA mixed direct-Rader round-trip mismatch batch={batch} index={index}: actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn cuda_forced_rader_n1922_dual_p31_fused_generator_loads_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if context.device_profile().vendor != GpuVendor::Nvidia
            || context.device_profile().max_threads_per_block < 32
            || context.device_profile().max_workgroup_size[0] < 32
        {
            return;
        }
        let mut planning_profile = context.device_profile();
        planning_profile.shared_memory_bytes = 1024;
        planning_profile.shared_memory_pow2_bytes = 1024;
        planning_profile.max_threads_per_block = 32;
        planning_profile.max_workgroup_size[0] = 32;
        let length = 2usize * 31 * 31;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                planning_profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N1922 fusion probe should remain recursive");
            };
            assert_eq!(
                ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
                vec![2, 31, 31]
            );
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            for (index, upload_id) in [(0usize, 2usize), (1usize, 1usize)] {
                let crate::RecursiveFftNodeIr::FftRader(rader) = &uploads[index] else {
                    panic!("CUDA N1922 upload{upload_id} must remain p31 FFT Rader");
                };
                assert_eq!(
                    rader.input_strategy,
                    crate::RaderFftInputStrategy::GeneratorOrderStockham
                );
                let forward = rader.forward_recursive().unwrap();
                let crate::RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
                    panic!(
                        "CUDA N1922 upload{upload_id} should fuse generator loads into Stockham"
                    );
                };
                assert!(matches!(
                    kernel.io_mapping,
                    crate::kernel_ir::StockhamIoMapping::RaderGeneratorFourStep(_)
                ));
            }
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for k in [0usize, 1, 31, 257, length / 2, length - 1] {
            let angle = -std::f32::consts::TAU * k as f32 / length as f32;
            let actual = spectrum[k];
            assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 3.0e-3);
        }
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(a, b)| (*a - *b).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(error <= 3.0e-3);
    }

    #[test]
    fn cuda_forced_rader_pass_local_direct_parent_n33728_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if context.device_profile().vendor != GpuVendor::Nvidia
            || context.device_profile().max_threads_per_block < 64
            || context.device_profile().max_workgroup_size[0] < 64
        {
            return;
        }
        let mut planning_profile = context.device_profile();
        planning_profile.shared_memory_bytes = 2 * 1024;
        planning_profile.shared_memory_pow2_bytes = 2 * 1024;
        planning_profile.max_threads_per_block = 64;
        planning_profile.max_workgroup_size[0] = 64;
        let length = 17usize * 31 * 64;
        let build = |direction| {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                planning_profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(ir)) = transform else {
                panic!("CUDA N33728 pass-local Rader probe should remain recursive");
            };
            assert_eq!(
                ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
                vec![32, 34, 31]
            );
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            let crate::recursive_ir::RecursiveFftNodeIr::FftRader(high) = &uploads[0] else {
                panic!("CUDA upload2 must remain p31 FFT Rader");
            };
            let high_block = high
                .caller_axis_batch_block
                .expect("CUDA p31 upload2 pass-local caller block");
            assert_eq!(high_block.threads_per_transform, 7);
            assert_eq!(high_block.grouped_batch, 8);
            assert!(high_block.transforms_on_x);
            assert!(!high_block.axis_swapped);
            assert_eq!([high_block.local_size_x, high_block.local_size_y], [8, 7]);
            let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(middle) = &uploads[1] else {
                panic!("CUDA upload1 must remain 2 x p17 Cooley");
            };
            let block = middle.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, 9);
            assert_eq!(block.grouped_batch, 4);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!([block.local_size_x, block.local_size_y], [4, 9]);
            assert_eq!(middle.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(middle.scatter_output.axis_batch_block, Some(block));
        }
        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for k in [0usize, 1, 17, 31, 257, length / 2, length - 1] {
            let angle = -std::f32::consts::TAU * k as f32 / length as f32;
            let actual = spectrum[k];
            assert!((actual.re - angle.cos()).hypot(actual.im - angle.sin()) <= 3.0e-3);
        }
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(round_trip_error <= 3.0e-3);
    }

    #[test]
    fn cuda_fft_rader_forced_upload_mixed_grouped_padding_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.shared_memory_bytes = 4 * 1024;
        planning_profile.shared_memory_pow2_bytes = 4 * 1024;
        let length = 17usize * 31 * 64;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let pad_left = 1000usize;
        let pad_right = 1200usize;
        let kept_index = 3456usize;
        let dropped_index = 1100usize;
        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        let mut probes = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let dc = Complex32::new(0.75 + 0.03125 * batch as f32, -0.125);
            let kept = Complex32::new(-0.3125, 0.1875 - 0.015625 * batch as f32);
            let dropped = Complex32::new(0.21875, -0.15625);
            let base = batch * length;
            input[base] = dc;
            input[base + kept_index] = kept;
            input[base + dropped_index] = dropped;
            probes.push((dc, kept, dropped));
        }
        let build = |direction: Direction, padded: bool| {
            let mut config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse);
            if padded {
                config = config.with_zero_padding(0, pad_left, pad_right).unwrap();
            }
            TransformIr::build(config, direction, planning_profile).unwrap()
        };

        let unpadded = build(Direction::Forward, false);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &unpadded else {
            panic!("CUDA mixed FFT-Rader probe should remain recursive");
        };
        assert_eq!(
            recursive
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![32, 34, 31]
        );
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        let crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) = &uploads[0] else {
            panic!("CUDA 31-point highest upload should be direct Rader");
        };
        assert_eq!(rader.input_storage_scalar, crate::ScalarType::F16);
        assert!(rader.caller_axis_batch_block.is_some());
        assert_eq!(
            rader.input_strategy,
            crate::RaderFftInputStrategy::GeneratorOrderStockham
        );
        let forward = rader.forward_recursive().unwrap();
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
            panic!("CUDA mapped p31 should fuse generator loads into forward Stockham");
        };
        assert!(matches!(
            kernel.io_mapping,
            crate::kernel_ir::StockhamIoMapping::RaderGeneratorFourStep(_)
        ));
        assert_eq!(kernel.bindings[0].scalar, crate::ScalarType::F16);
        let unpadded_spectrum = context
            .execute_transform_complex32(&unpadded, &input)
            .unwrap();
        for (batch, &(dc, kept, dropped)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let kept64 = Complex64::new(kept.re as f64, kept.im as f64);
            let dropped64 = Complex64::new(dropped.re as f64, dropped.im as f64);
            for bin in [0usize, 1, 17, 47, 187, length / 2, length - 1] {
                let a = -core::f64::consts::TAU * (bin * kept_index) as f64 / length as f64;
                let b = -core::f64::consts::TAU * (bin * dropped_index) as f64 / length as f64;
                let expected =
                    dc64 + kept64 * Complex64::exp_i(a) + dropped64 * Complex64::exp_i(b);
                let actual = unpadded_spectrum[batch * length + bin];
                assert!(
                    (actual.re as f64 - expected.re).abs() <= 1.5e-2
                        && (actual.im as f64 - expected.im).abs() <= 1.5e-2,
                    "CUDA mixed FFT-Rader unpadded mismatch batch={batch} bin={bin}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let padded = build(Direction::Forward, true);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &padded else {
            panic!("CUDA padded FFT-Rader probe should remain recursive");
        };
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        let crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) = &uploads[0] else {
            panic!("CUDA padded 31-point highest upload should be direct Rader");
        };
        assert_eq!(rader.input_storage_scalar, crate::ScalarType::F32);
        let padded_spectrum = context
            .execute_transform_complex32(&padded, &input)
            .unwrap();
        for (batch, &(dc, kept, _)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let kept64 = Complex64::new(kept.re as f64, kept.im as f64);
            for bin in [0usize, 1, 17, 47, 187, length / 2, length - 1] {
                let angle = -core::f64::consts::TAU * (bin * kept_index) as f64 / length as f64;
                let expected = dc64 + kept64 * Complex64::exp_i(angle);
                let actual = padded_spectrum[batch * length + bin];
                assert!(
                    (actual.re as f64 - expected.re).abs() <= 1.5e-2
                        && (actual.im as f64 - expected.im).abs() <= 1.5e-2,
                    "CUDA mixed FFT-Rader padded mismatch batch={batch} bin={bin}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let inverse = build(Direction::Inverse, true);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &inverse else {
            panic!("CUDA inverse FFT-Rader probe should remain recursive");
        };
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        let crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) = &uploads[0] else {
            panic!("CUDA inverse 31-point highest upload should be direct Rader");
        };
        assert_eq!(rader.input_storage_scalar, crate::ScalarType::F16);
        let restored = context
            .execute_transform_complex32(&inverse, &padded_spectrum)
            .unwrap();
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + pad_left..base + pad_right]
                    .iter()
                    .all(|value| value.re == 0.0 && value.im == 0.0)
            );
            for index in [0usize, kept_index, length - 1] {
                let expected = input[base + index];
                let actual = restored[base + index];
                assert!(
                    (actual.re - expected.re).abs() <= 2.5e-2
                        && (actual.im - expected.im).abs() <= 2.5e-2,
                    "CUDA mixed FFT-Rader round-trip mismatch batch={batch} index={index}: actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn cuda_composite_rader_grouped_mixed_padding_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 34usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.071 * x).sin() + 0.0002 * x,
                    (0.029 * x).cos() - 0.0001 * x,
                )
            })
            .collect::<Vec<_>>();
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_zero_padding(0, 7, 12)
                    .unwrap(),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let quantize = |values: &[Complex64]| {
            values
                .iter()
                .map(|value| {
                    Complex32::new(
                        crate::Binary16::from_f32(value.re as f32).to_f32(),
                        crate::Binary16::from_f32(value.im as f32).to_f32(),
                    )
                })
                .collect::<Vec<_>>()
        };

        let forward = build(Direction::Forward);
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA N=34 grouped composite Rader should remain recursive C2C");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
            panic!("CUDA N=34 grouped composite Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.grouped_batch, grouped_batch);
        assert_eq!(root.pack_right.dispatch.x, 2);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let quantized_input = input
            .iter()
            .map(|value| {
                Complex64::new(
                    crate::Binary16::from_f32(value.re).to_f32() as f64,
                    crate::Binary16::from_f32(value.im).to_f32() as f64,
                )
            })
            .collect::<Vec<_>>();
        let expected_forward =
            quantize(&forward.execute_complex_reference(&quantized_input).unwrap());
        let actual_forward = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = actual_forward
            .iter()
            .zip(&expected_forward)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            forward_error <= 1.0e-3 * length as f32,
            "CUDA composite grouped F16 forward mismatch on {}: {forward_error:e}",
            context.device_name()
        );

        let inverse = build(Direction::Inverse);
        let spectrum64 = actual_forward
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected_inverse = quantize(&inverse.execute_complex_reference(&spectrum64).unwrap());
        let actual_inverse = context
            .execute_transform_complex32(&inverse, &actual_forward)
            .unwrap();
        let inverse_error = actual_inverse
            .iter()
            .zip(&expected_inverse)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            inverse_error <= 1.0e-3 * length as f32,
            "CUDA composite grouped F16 inverse mismatch on {}: {inverse_error:e}",
            context.device_name()
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                actual_inverse[base + 7..base + 12]
                    .iter()
                    .all(|value| value.re == 0.0 && value.im == 0.0)
            );
        }
    }

    #[test]
    fn cuda_grouped_forced_rader_four_step_tail_matches_sparse_oracle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia
            || profile.max_threads_per_block < 816
            || profile.max_workgroup_size[1] < 272
        {
            return;
        }
        let length = 17usize * 8192;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA grouped forced-Rader probe should remain recursive");
        };
        let four_step = recursive.four_step_plan.as_ref().unwrap();
        assert_eq!(
            four_step.uploads[1].axis_block.unwrap().grouped_batch,
            grouped_batch
        );
        assert!(
            !four_step.uploads[1]
                .transform_count
                .is_multiple_of(grouped_batch)
        );

        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        let mut probes = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let dc = Complex32::new(1.0 + 0.125 * batch as f32, -0.03125 * batch as f32);
            let impulse_index = 12_345usize + 997 * batch;
            let impulse =
                Complex32::new(0.25 - 0.03125 * batch as f32, -0.1875 + 0.02 * batch as f32);
            let base = batch * length;
            input[base] = dc;
            input[base + impulse_index] = impulse;
            probes.push((dc, impulse_index, impulse));
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for (batch, &(dc, impulse_index, impulse)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let impulse64 = Complex64::new(impulse.re as f64, impulse.im as f64);
            for bin in [0usize, 1, 17, 271, 272, 511, 512, length / 2, length - 1] {
                let angle = -core::f64::consts::TAU * (bin * impulse_index) as f64 / length as f64;
                let expected = dc64 + impulse64 * Complex64::exp_i(angle);
                let actual = spectrum[batch * length + bin];
                assert!(
                    (actual.re as f64 - expected.re).abs() <= 1.5e-2
                        && (actual.im as f64 - expected.im).abs() <= 1.5e-2,
                    "CUDA grouped forced-Rader mismatch batch={batch} bin={bin}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_inverse_normalization(true),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 4.0e-2,
            "CUDA grouped forced-Rader round-trip mismatch: {max_error:e}"
        );
    }

    #[test]
    fn cuda_grouped_forced_rader_mixed_padding_tail_matches_oracle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia
            || profile.max_threads_per_block < 816
            || profile.max_workgroup_size[1] < 272
        {
            return;
        }
        let length = 17usize * 8192;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let pad_left = 12_000usize;
        let pad_right = 12_700usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_zero_padding(0, pad_left, pad_right)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA grouped mixed-padding forced-Rader probe should remain recursive");
        };
        assert_eq!(
            recursive
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![512, 272]
        );
        assert_eq!(
            recursive.four_step_plan.as_ref().unwrap().uploads[1]
                .axis_block
                .unwrap()
                .grouped_batch,
            grouped_batch
        );

        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        let mut probes = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let dc = Complex32::new(1.0 + 0.125 * batch as f32, -0.0625 * batch as f32);
            let padded_index = pad_left + 17 * batch;
            let padded = Complex32::new(0.375, -0.25);
            let kept_index = 54_321usize + 997 * batch;
            let kept = Complex32::new(
                -0.1875 + 0.03125 * batch as f32,
                0.3125 - 0.015625 * batch as f32,
            );
            let base = batch * length;
            input[base] = dc;
            input[base + padded_index] = padded;
            input[base + kept_index] = kept;
            probes.push((dc, kept_index, kept));
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for (batch, &(dc, kept_index, kept)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let kept64 = Complex64::new(kept.re as f64, kept.im as f64);
            for bin in [0usize, 1, 17, 271, 272, 511, 512, length / 2, length - 1] {
                let angle = -core::f64::consts::TAU * (bin * kept_index) as f64 / length as f64;
                let expected = dc64 + kept64 * Complex64::exp_i(angle);
                let expected_re = crate::Binary16::from_f32(expected.re as f32).to_f32();
                let expected_im = crate::Binary16::from_f32(expected.im as f32).to_f32();
                let actual = spectrum[batch * length + bin];
                assert!(
                    (actual.re - expected_re).abs() <= 8.0e-3
                        && (actual.im - expected_im).abs() <= 8.0e-3,
                    "CUDA grouped mixed-padding forced-Rader mismatch batch={batch} bin={bin}: actual={actual:?}, expected=({expected_re},{expected_im})"
                );
            }
        }

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(true)
                .with_zero_padding(0, pad_left, pad_right)
                .unwrap(),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let mut dc_spectrum = vec![Complex32::new(0.0, 0.0); length * batch_count];
        let mut expected_dc = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let dc = Complex32::new(1.0 + 0.125 * batch as f32, -0.0625 * batch as f32);
            dc_spectrum[batch * length] = dc;
            expected_dc.push(Complex32::new(
                crate::Binary16::from_f32(dc.re / length as f32).to_f32(),
                crate::Binary16::from_f32(dc.im / length as f32).to_f32(),
            ));
        }
        let restored = context
            .execute_transform_complex32(&inverse, &dc_spectrum)
            .unwrap();
        for (batch, expected) in expected_dc.iter().copied().enumerate() {
            let base = batch * length;
            for index in [0usize, 1, pad_left - 1, pad_right, 54_321, length - 1] {
                let actual = restored[base + index];
                assert!(
                    (actual.re - expected.re).abs() <= 2.0e-5
                        && (actual.im - expected.im).abs() <= 2.0e-5,
                    "CUDA grouped mixed-padding inverse mismatch batch={batch} index={index}: actual={actual:?}, expected={expected:?}"
                );
            }
            assert!(
                restored[base + pad_left..base + pad_right]
                    .iter()
                    .all(|value| value.re == 0.0 && value.im == 0.0)
            );
        }
    }

    #[test]
    fn cuda_grouped_forced_rader_three_upload_tail_matches_sparse_oracle_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Nvidia
            || profile.max_threads_per_block < 544
            || profile.max_workgroup_size[1] < 68
        {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.shared_memory_bytes = 32 * 1024;
        planning_profile.shared_memory_pow2_bytes = 32 * 1024;
        let length = 17usize * 65_536;
        let batch_count = 2usize;
        let grouped_batch = 3usize;
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap(),
            Direction::Forward,
            planning_profile,
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward else {
            panic!("CUDA grouped three-upload forced-Rader probe should remain recursive");
        };
        assert_eq!(
            recursive
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![128, 68, 128]
        );
        let four_step = recursive.four_step_plan.as_ref().unwrap();
        assert_eq!(
            four_step.uploads[2].axis_block.unwrap().grouped_batch,
            grouped_batch
        );
        assert!(
            !four_step.uploads[2]
                .transform_count
                .is_multiple_of(grouped_batch)
        );

        let mut input = vec![Complex32::new(0.0, 0.0); length * batch_count];
        let mut probes = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let dc = Complex32::new(0.875 + 0.125 * batch as f32, -0.0625 * batch as f32);
            let impulse_index = 98_765usize + 7_919 * batch;
            let impulse = Complex32::new(-0.3125 + 0.03125 * batch as f32, 0.1875);
            let base = batch * length;
            input[base] = dc;
            input[base + impulse_index] = impulse;
            probes.push((dc, impulse_index, impulse));
        }
        let spectrum = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        for (batch, &(dc, impulse_index, impulse)) in probes.iter().enumerate() {
            let dc64 = Complex64::new(dc.re as f64, dc.im as f64);
            let impulse64 = Complex64::new(impulse.re as f64, impulse.im as f64);
            for bin in [
                0usize,
                1,
                17,
                67,
                68,
                127,
                128,
                4095,
                length / 2,
                length - 1,
            ] {
                let angle = -core::f64::consts::TAU * (bin * impulse_index) as f64 / length as f64;
                let expected = dc64 + impulse64 * Complex64::exp_i(angle);
                let actual = spectrum[batch * length + bin];
                assert!(
                    (actual.re as f64 - expected.re).abs() <= 2.5e-2
                        && (actual.im as f64 - expected.im).abs() <= 2.5e-2,
                    "CUDA grouped three-upload forced-Rader mismatch batch={batch} bin={bin}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_inverse_normalization(true),
            Direction::Inverse,
            planning_profile,
        )
        .unwrap();
        let restored = context
            .execute_transform_complex32(&inverse, &spectrum)
            .unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 7.5e-2,
            "CUDA grouped three-upload forced-Rader round-trip mismatch: {max_error:e}"
        );
    }

    #[test]
    fn cuda_dd_r2r_forced_three_upload_n1114112_fused_boundaries_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if !profile.supports_f64
            || profile.vendor != GpuVendor::Nvidia
            || profile.max_threads_per_block < 544
            || profile.max_workgroup_size[1] < 68
        {
            return;
        }
        let mut planning_profile = profile;
        planning_profile.shared_memory_bytes = 32 * 1024;
        planning_profile.shared_memory_pow2_bytes = 32 * 1024;
        let length = 17usize * 65_536;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                planning_profile,
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                panic!("CUDA DD N1114112 DCT-II/III must build 1D R2R IR");
            };
            let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                panic!("CUDA DD N1114112 DCT-II/III must use FFT reduction");
            };
            let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("CUDA DD N1114112 DCT-II/III must retain recursive child");
            };
            assert_eq!(
                recursive
                    .rader_forced_upload_schedule
                    .as_ref()
                    .expect("CUDA DD N1114112 forced-three schedule")
                    .axis_split,
                vec![128, 68, 128]
            );
            let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
            let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
            assert_eq!(program.passes.len(), child_program.passes.len());
            assert!(program.resources.iter().any(|resource| {
                resource.name == "double_double_r2r_forced_three_upload_exchange_0"
                    && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
            }));
            assert!(program.resources.iter().all(|resource| {
                !resource.name.contains("double_double_r2r_fft_input")
                    && !resource.name.contains("double_double_r2r_fft_output")
            }));
        }

        let impulse_index = 98_765usize;
        let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
        let mut input = vec![DoubleDouble::ZERO; length];
        input[impulse_index] = impulse;
        let spectrum = context
            .execute_transform_double_double_r2r(&forward, &input)
            .unwrap();
        let two = DoubleDouble::from_f64(2.0);
        let half = DoubleDouble::from_f64(0.5);
        for k in [
            0usize,
            1,
            17,
            67,
            68,
            127,
            128,
            4095,
            length / 2,
            length - 1,
        ] {
            let angle = DoubleDouble::PI
                * (DoubleDouble::from_f64(impulse_index as f64) + half)
                * DoubleDouble::from_f64(k as f64)
                / DoubleDouble::from_f64(length as f64);
            let (_, cosine) = angle.sin_cos();
            let expected = impulse * two * cosine;
            let delta = (spectrum[k] - expected).abs();
            let error = delta.hi.abs() + delta.lo.abs();
            assert!(
                error <= 5.0e-10,
                "CUDA DD N1114112 fused forced-three DCT-II bin {k} error {error:e}"
            );
        }
        let restored = context
            .execute_transform_double_double_r2r(&inverse, &spectrum)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-11,
            "CUDA DD N1114112 fused forced-three DCT-II/III round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn cuda_dd_r2r_forced_three_upload_n102272_recursive_u2_fused_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_device = context.device_profile();
        if !actual_device.supports_f64
            || actual_device.vendor != crate::GpuVendor::Nvidia
            || actual_device.shared_memory_bytes < 8 * 1024
            || actual_device.shared_memory_pow2_bytes < 8 * 1024
            || actual_device.max_threads_per_block < 1024
        {
            return;
        }
        let mut device = actual_device;
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        let length = 17usize * 47 * 128;
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(crate::PlannerTuning::portable())
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                device,
            )
            .unwrap()
        };
        let bins = [
            0usize,
            1,
            17,
            33,
            34,
            46,
            47,
            63,
            64,
            4095,
            length / 2,
            length - 1,
        ];
        let impulse_index = 9_876usize;

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let forward = build(precision, Direction::Forward);
            let inverse = build(precision, Direction::Inverse);
            for transform in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("CUDA DD N102272 DCT-II/III must build 1D R2R IR");
                };
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm
                else {
                    panic!("CUDA DD N102272 DCT-II/III must use FFT reduction");
                };
                let crate::DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("CUDA DD N102272 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("CUDA DD N102272 forced-three schedule")
                        .axis_split,
                    vec![64, 47, 34]
                );
                let components = recursive
                    .forced_rader_three_upload_mapped_components()
                    .unwrap()
                    .expect("CUDA DD N102272 mapped forced-three components");
                let Some(
                    crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                        upload_id: 2,
                        ir: crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(high),
                    },
                ) = components.first()
                else {
                    panic!("CUDA DD N102272 upload2 must remain recursive Cooley-Tukey");
                };
                assert!(matches!(
                    high.pack_right.input_modifier,
                    crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepThreeUpload2(_)
                ));
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_r2r_forced_three_upload_exchange_0"
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
            }

            match precision {
                Precision::DoubleDouble => {
                    let impulse = DoubleDouble::from_parts(0.875, 3.0e-31);
                    let mut input = vec![DoubleDouble::ZERO; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r(&forward, &input)
                        .unwrap();
                    let two = DoubleDouble::from_f64(2.0);
                    let half = DoubleDouble::from_f64(0.5);
                    for k in bins {
                        let angle = DoubleDouble::PI
                            * (DoubleDouble::from_f64(impulse_index as f64) + half)
                            * DoubleDouble::from_f64(k as f64)
                            / DoubleDouble::from_f64(length as f64);
                        let (_, cosine) = angle.sin_cos();
                        let expected = impulse * two * cosine;
                        let delta = (spectrum[k] - expected).abs();
                        let error = delta.hi.abs() + delta.lo.abs();
                        assert!(
                            error <= 5.0e-10,
                            "CUDA DD N102272 recursive-u2 DCT-II bin {k} error {error:e}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| {
                            let delta = (actual - expected).abs();
                            delta.hi.abs() + delta.lo.abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        round_trip_error <= 5.0e-11,
                        "CUDA DD N102272 recursive-u2 round-trip error {round_trip_error:e}"
                    );
                }
                Precision::DoubleDoubleF64Storage => {
                    let impulse = 0.875f64;
                    let mut input = vec![0.0f64; length];
                    input[impulse_index] = impulse;
                    let spectrum = context
                        .execute_transform_double_double_r2r_f64_storage(&forward, &input)
                        .unwrap();
                    for k in bins {
                        let angle = std::f64::consts::PI * (impulse_index as f64 + 0.5) * k as f64
                            / length as f64;
                        let expected = 2.0 * impulse * angle.cos();
                        let error = (spectrum[k] - expected).abs();
                        assert!(
                            error <= 2.0e-9,
                            "CUDA DD/F64 N102272 recursive-u2 DCT-II bin {k} error {error:e}"
                        );
                    }
                    let restored = context
                        .execute_transform_double_double_r2r_f64_storage(&inverse, &spectrum)
                        .unwrap();
                    let round_trip_error = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        round_trip_error <= 2.0e-10,
                        "CUDA DD/F64 N102272 recursive-u2 round-trip error {round_trip_error:e}"
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn cuda_user_grouped_batch_four_step_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 49_152usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length])
                .with_grouped_batch(0, 5)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("49,152-point groupedBatch CUDA transform should use recursive Stockham IR");
        };
        assert_eq!(recursive.axis0_grouped_batch_override, Some(5));
        assert_eq!(
            recursive
                .four_step_plan
                .as_ref()
                .and_then(|plan| plan.uploads.last())
                .and_then(|upload| upload.axis_block)
                .map(|block| block.grouped_batch),
            Some(5)
        );
        let kernels = recursive
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        assert!(kernels.iter().any(|kernel| {
            !kernel
                .batch_count
                .is_multiple_of(kernel.workgroup_grouping.transforms_per_workgroup)
        }));

        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.017 * x).sin() + 0.00001 * x,
                    (0.023 * x).cos() - 0.00002 * x,
                )
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 2.0e-3 * length as f64);
    }

    #[test]
    fn cuda_user_grouped_batch_rader_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 257usize;
        let batch_count = 32usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, 5)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
            panic!("groupedBatch p257 CUDA transform should use recursive Rader IR");
        };
        let crate::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
            panic!("groupedBatch p257 CUDA transform should keep an FFT-Rader root");
        };
        let block = rader
            .axis_batch_block
            .expect("CUDA p257 should consume groupedBatch=5");
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 5);
        let crate::RecursiveFftNodeIr::Stockham(kernel) = &rader.forward_recursive().unwrap().root
        else {
            panic!("groupedBatch p257 CUDA convolution should remain Stockham");
        };
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [17, 5]);
        assert_eq!(kernel.dispatch.x, 7);

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.037 * x).sin() + x * 0.00001, (0.019 * x).cos())
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 4.0e-3 * length as f64);

        let direct_length = 47usize;
        let direct_ir = TransformIr::build(
            FftConfig::new(vec![direct_length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, 3)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(direct_recursive)) = &direct_ir
        else {
            panic!("groupedBatch p47 CUDA transform should use recursive direct Rader IR");
        };
        let crate::RecursiveFftNodeIr::DirectRader(direct) = &direct_recursive.root else {
            panic!("groupedBatch p47 CUDA transform should keep a direct-Rader root");
        };
        let direct_block = direct
            .axis_batch_block
            .expect("CUDA p47 should consume groupedBatch=3 user branch");
        assert_eq!(direct_block.threads_per_transform, 24);
        assert_eq!(direct_block.grouped_batch, 1);
        assert_eq!([direct.workgroup_size.x, direct.workgroup_size.y], [24, 1]);
        assert_eq!(direct.dispatch.x, 32);
        let direct_input = (0..direct_length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.041 * x).sin() + x * 0.00001, (0.017 * x).cos())
            })
            .collect::<Vec<_>>();
        let direct_expected_input = direct_input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let direct_expected = direct_ir
            .execute_complex_reference(&direct_expected_input)
            .unwrap();
        let direct_actual = context
            .execute_transform_complex32(&direct_ir, &direct_input)
            .unwrap();
        assert!(max_error32(&direct_actual, &direct_expected) <= 4.0e-3 * direct_length as f64);
    }

    #[test]
    fn cuda_real_mixed_storage_even_and_odd_match_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for length in [15usize, 16] {
                let input = (0..length * batch_count)
                    .map(|index| {
                        let x = index as f32;
                        (0.107 * x).sin() + 0.21 * (0.041 * x).cos() + x * 0.0002
                    })
                    .collect::<Vec<_>>();
                let forward = TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_precision(precision)
                        .with_transform(crate::TransformKind::RealToComplex)
                        .with_zero_padding(0, 3, 7)
                        .unwrap(),
                    Direction::Forward,
                    context.device_profile(),
                )
                .unwrap();
                let spectrum = match context
                    .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
                    .unwrap()
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => {
                        panic!("mixed CUDA R2C returned real output")
                    }
                };
                let oracle_input = input
                    .iter()
                    .map(|value| {
                        if precision == Precision::F16StorageF32Compute {
                            crate::Binary16::from_f32(*value).to_f32() as f64
                        } else {
                            *value as f64
                        }
                    })
                    .collect::<Vec<_>>();
                let mut expected = forward.execute_r2c_reference(&oracle_input).unwrap();
                for value in &mut expected {
                    if precision == Precision::F16StorageF32Compute {
                        value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                        value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                    } else {
                        value.re = value.re as f32 as f64;
                        value.im = value.im as f32 as f64;
                    }
                }
                let tolerance = if precision == Precision::F16StorageF32Compute {
                    2.5e-3 * length as f64
                } else {
                    3.0e-5 * length as f64
                };
                assert!(max_error32(&spectrum, &expected) <= tolerance);

                let inverse = TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_precision(precision)
                        .with_transform(crate::TransformKind::ComplexToReal)
                        .with_inverse_normalization(true)
                        .with_zero_padding(0, 3, 7)
                        .unwrap(),
                    Direction::Inverse,
                    context.device_profile(),
                )
                .unwrap();
                let restored = match context
                    .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
                    .unwrap()
                {
                    NativeTransformOutput32::Real(values) => values,
                    NativeTransformOutput32::Complex(_) => {
                        panic!("mixed CUDA C2R returned complex output")
                    }
                };
                let spectrum64 = spectrum
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let mut expected_real = inverse.execute_c2r_reference(&spectrum64).unwrap();
                for value in &mut expected_real {
                    if precision == Precision::F16StorageF32Compute {
                        *value = crate::Binary16::from_f32(*value as f32).to_f32() as f64;
                    } else {
                        *value = *value as f32 as f64;
                    }
                }
                let error = restored
                    .iter()
                    .zip(&expected_real)
                    .map(|(actual, expected)| (*actual as f64 - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    error <= tolerance,
                    "CUDA mixed C2R N={length} error {error}"
                );
            }
        }
    }

    #[test]
    fn cuda_grouped_even_real_stockham_fusion_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 64usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                (0.071 * x).sin() + 0.25 * (0.019 * x).cos() + x * 0.0002
            })
            .collect::<Vec<_>>();

        let forward_plan = crate::FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(crate::TransformKind::RealToComplex),
        )
        .unwrap();
        let forward = TransformIr::Real(
            crate::RealFftIr::build(&forward_plan, context.device_profile()).unwrap(),
        );
        let TransformIr::Real(real) = &forward else {
            panic!("CUDA grouped N64 R2C did not build RealFftIr");
        };
        let fused = real
            .fused_even_input_stockham_kernel()
            .unwrap()
            .expect("CUDA grouped N64 R2C should fuse around N32 Stockham");
        assert_eq!(
            fused.workgroup_grouping.transforms_per_workgroup,
            grouped_batch
        );
        assert_eq!(fused.dispatch.x, 3);
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("CUDA grouped R2C returned real output"),
        };
        let expected = forward
            .execute_r2c_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        let tolerance = 3.0e-5 * length as f64;
        assert!(max_error32(&spectrum, &expected) <= tolerance);

        let inverse_plan = crate::FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse = TransformIr::Real(
            crate::RealFftIr::build(&inverse_plan, context.device_profile()).unwrap(),
        );
        let TransformIr::Real(real) = &inverse else {
            panic!("CUDA grouped N64 C2R did not build RealFftIr");
        };
        assert!(real.fused_even_input_stockham_kernel().unwrap().is_some());
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA grouped C2R returned complex output")
            }
        };
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected_real = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&expected_real)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= tolerance,
            "CUDA grouped fused C2R error {inverse_error}"
        );
    }

    #[test]
    fn cuda_grouped_even_real_recursive_fusion_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 68usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                (0.047 * x).sin() + 0.19 * (0.023 * x).cos() + x * 0.0002
            })
            .collect::<Vec<_>>();
        let forward_plan = crate::FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(crate::TransformKind::RealToComplex),
        )
        .unwrap();
        let forward_real =
            crate::RealFftIr::build(&forward_plan, context.device_profile()).unwrap();
        let fused = forward_real
            .fused_even_recursive_ir()
            .unwrap()
            .expect("CUDA grouped N68 R2C should fuse into N34 recursive child");
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &fused.root else {
            panic!("CUDA grouped N68 half child should keep Cooley root");
        };
        assert_eq!(
            root.pack_right.axis_batch_block.unwrap().grouped_batch,
            grouped_batch
        );
        let forward = TransformIr::Real(forward_real);
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => {
                panic!("CUDA grouped recursive R2C returned real output")
            }
        };
        let expected = forward
            .execute_r2c_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        let tolerance = 4.0e-5 * length as f64;
        assert!(max_error32(&spectrum, &expected) <= tolerance);
        let inverse_plan = crate::FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse_real =
            crate::RealFftIr::build(&inverse_plan, context.device_profile()).unwrap();
        assert!(inverse_real.fused_even_recursive_ir().unwrap().is_some());
        let inverse = TransformIr::Real(inverse_real);
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA grouped recursive C2R returned complex output")
            }
        };
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected_real = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&expected_real)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= tolerance,
            "CUDA grouped recursive C2R error {inverse_error}"
        );
    }

    #[test]
    fn cuda_grouped_even_real_bluestein_fusion_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let profile = context.device_profile();
        if profile.max_threads_per_block < 384
            || profile.max_workgroup_size[0] < 128
            || profile.max_workgroup_size[1] < 3
        {
            return;
        }
        let length = 206usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                (0.029 * x).sin() + 0.11 * (0.017 * x).cos() + x * 0.00015
            })
            .collect::<Vec<_>>();
        let forward_plan = crate::FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(crate::TransformKind::RealToComplex)
                .with_tuning(tuning),
        )
        .unwrap();
        let forward_real = crate::RealFftIr::build(&forward_plan, profile).unwrap();
        let fused = forward_real
            .fused_even_bluestein_ir()
            .unwrap()
            .expect("CUDA grouped N206 R2C should fuse around p103 Bluestein");
        let block = fused.preprocess.axis_batch_block.unwrap();
        assert_eq!([block.local_size_x, block.local_size_y], [128, 3]);
        let forward = TransformIr::Real(forward_real);
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => {
                panic!("CUDA grouped Bluestein R2C returned real output")
            }
        };
        let expected = forward
            .execute_r2c_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        let tolerance = 6.0e-5 * length as f64;
        assert!(max_error32(&spectrum, &expected) <= tolerance);
        let inverse_plan = crate::FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
        )
        .unwrap();
        let inverse_real = crate::RealFftIr::build(&inverse_plan, profile).unwrap();
        assert!(inverse_real.fused_even_bluestein_ir().unwrap().is_some());
        let inverse = TransformIr::Real(inverse_real);
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA grouped Bluestein C2R returned complex output")
            }
        };
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected_real = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&expected_real)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= tolerance,
            "CUDA grouped Bluestein C2R error {inverse_error}"
        );
    }

    #[test]
    fn cuda_real_gpu_stockham_and_multpass_bluestein_match_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for length in [64usize, 256] {
            let input = (0..length)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new((0.13 * x).sin() + 0.001 * x, (0.07 * x).cos() - 0.002 * x)
                })
                .collect::<Vec<_>>();
            let ir = TransformIr::build(
                FftConfig::new(vec![length]),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let expected_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected = ir.execute_complex_reference(&expected_input).unwrap();
            assert!(max_error32(&actual, &expected) <= 4.0e-4 * length as f64);
        }

        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let length = 103usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.11 * x).sin(), (0.05 * x).cos())
            })
            .collect::<Vec<_>>();
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::Cuda)
            .lower_transform(&ir)
            .unwrap();
        assert!(source.shaders.len() >= 4);
        let actual = context.execute_program_complex32(&source, &input).unwrap();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        assert!(max_error32(&actual, &expected) <= 2.0e-3 * length as f64);
    }

    #[test]
    fn cuda_device_resident_program_chain_round_trips_and_rejects_mismatch_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 4096usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.017 * x).sin() + x * 1.0e-6, (0.029 * x).cos())
            })
            .collect::<Vec<_>>();
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_batch_count(batch_count),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let lowering = NativeSourceBackend::new(Backend::Cuda);
        let forward_source = lowering.lower_transform(&forward).unwrap();
        let inverse_source = lowering.lower_transform(&inverse).unwrap();
        let restored = context
            .execute_program_chain_complex32(&[&forward_source, &inverse_source], &input)
            .unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re as f64;
                let di = actual.im as f64 - expected.im as f64;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-4,
            "resident CUDA round-trip error {max_error}"
        );

        let one_source = context
            .execute_program_chain_complex32(&[&forward_source], &input)
            .unwrap();
        let ordinary = context
            .execute_program_complex32(&forward_source, &input)
            .unwrap();
        assert_eq!(one_source, ordinary);

        let incompatible = TransformIr::build(
            FftConfig::new(vec![length / 2]).with_batch_count(batch_count),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let incompatible_source = lowering.lower_transform(&incompatible).unwrap();
        assert!(matches!(
            context
                .execute_program_chain_complex32(&[&forward_source, &incompatible_source], &input),
            Err(VkFftError::InvalidKernelIr(_))
        ));
        assert!(matches!(
            context.execute_program_chain_complex32(&[], &input),
            Err(VkFftError::InvalidKernelIr(_))
        ));
    }

    #[test]
    fn cuda_device_resident_program_chain_into_reuses_caller_output_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 4096usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.013 * x).sin() + x * 2.0e-6, (0.023 * x).cos())
            })
            .collect::<Vec<_>>();
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_batch_count(batch_count),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let lowering = NativeSourceBackend::new(Backend::Cuda);
        let forward_source = lowering.lower_transform(&forward).unwrap();
        let inverse_source = lowering.lower_transform(&inverse).unwrap();

        let mut output = vec![Complex32::new(123.0, -456.0); input.len()];
        let output_ptr = output.as_ptr();
        for _ in 0..2 {
            context
                .execute_program_chain_complex32_into(
                    &[&forward_source, &inverse_source],
                    &input,
                    &mut output,
                )
                .unwrap();
            assert_eq!(output.as_ptr(), output_ptr);
            let max_error = output
                .iter()
                .zip(&input)
                .map(|(actual, expected)| {
                    let dr = actual.re as f64 - expected.re as f64;
                    let di = actual.im as f64 - expected.im as f64;
                    (dr * dr + di * di).sqrt()
                })
                .fold(0.0, f64::max);
            assert!(
                max_error <= 2.0e-4,
                "direct resident CUDA round-trip error {max_error}"
            );
            output.fill(Complex32::new(-9.0, 7.0));
        }

        let mut short_output = vec![Complex32::default(); input.len() - 1];
        assert!(matches!(
            context.execute_program_chain_complex32_into(
                &[&forward_source, &inverse_source],
                &input,
                &mut short_output,
            ),
            Err(VkFftError::InvalidKernelIr(_))
        ));

        let f16 = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::F16StorageF32Compute),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f16_source = lowering.lower_transform(&f16).unwrap();
        assert!(matches!(
            context.execute_program_chain_complex32_into(&[&f16_source], &input, &mut output),
            Err(VkFftError::InvalidKernelIr(_))
        ));
    }

    #[test]
    fn cuda_two_high_level_tickets_can_be_in_flight_before_wait_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };

        let length = 3840usize;
        let complex_input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.019 * x).sin() + x * 0.00002, (0.031 * x).cos())
            })
            .collect::<Vec<_>>();
        let complex_ir = TransformIr::build(
            FftConfig::new(vec![length]),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let complex_ticket = context
            .submit_transform_complex32(&complex_ir, &complex_input)
            .unwrap();

        let real_length = 30usize;
        let real_input = (0..real_length)
            .map(|index| {
                let x = index as f32;
                (0.071 * x).sin() + 0.31 * (0.043 * x).cos()
            })
            .collect::<Vec<_>>();
        let real_ir = TransformIr::build(
            FftConfig::new(vec![real_length]).with_transform(crate::TransformKind::RealToComplex),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let real_ticket = context
            .submit_transform_f32(&real_ir, NativeTransformInput32::Real(&real_input))
            .unwrap();

        let real_output = match real_ticket.wait().unwrap() {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("async CUDA R2C returned real output"),
        };
        let real_expected = real_ir
            .execute_r2c_reference(
                &real_input
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(max_error32(&real_output, &real_expected) <= 2.0e-3 * real_length as f64);

        let complex_output = complex_ticket.wait().unwrap();
        let complex_expected_input = complex_input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let complex_expected =
            crate::reference::fft(&complex_expected_input, Direction::Forward, false).unwrap();
        assert!(max_error32(&complex_output, &complex_expected) <= 8.0e-4 * length as f64);
    }

    #[test]
    fn cuda_ptx_cache_archive_round_trips_and_rejects_incompatible_headers() {
        let archive = CudaPtxCacheArchive {
            compute_capability: (8, 0),
            driver_cuda_version: (12, 8),
            entries: vec![
                ("source-b".to_owned(), vec![4, 5, 6]),
                ("source-a".to_owned(), vec![1, 2, 3]),
            ],
        };
        let encoded = archive.encode();
        let decoded = CudaPtxCacheArchive::decode(&encoded).unwrap();
        assert_eq!(decoded.compute_capability(), (8, 0));
        assert_eq!(decoded.driver_cuda_version(), (12, 8));
        assert_eq!(decoded.entry_count(), 2);
        assert_eq!(decoded.encode(), encoded);

        let mut bad_version = encoded.clone();
        bad_version[8..12].copy_from_slice(&(CUDA_PTX_CACHE_ARCHIVE_VERSION + 1).to_le_bytes());
        assert!(CudaPtxCacheArchive::decode(&bad_version).is_err());

        let mut bad_commit = encoded.clone();
        bad_commit[28] ^= 0x01;
        assert!(CudaPtxCacheArchive::decode(&bad_commit).is_err());

        let mut trailing = encoded;
        trailing.push(0);
        assert!(CudaPtxCacheArchive::decode(&trailing).is_err());
    }

    #[test]
    fn cuda_runtime_caches_modules_luts_transients_and_persists_ptx_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        assert_eq!(context.cached_ptx_count().unwrap(), 0);
        assert_eq!(context.cached_module_count().unwrap(), 0);
        assert_eq!(context.cached_lut_count().unwrap(), 0);
        assert_eq!(context.cached_transient_buffer_count().unwrap(), 0);
        assert!(context.host_output_pool.lock().unwrap().is_empty());

        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let length = 103usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.071 * x).sin(), (0.043 * x).cos())
            })
            .collect::<Vec<_>>();
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_tuning(tuning),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::Cuda)
            .lower_transform(&ir)
            .unwrap();

        let first = context.execute_program_complex32(&source, &input).unwrap();
        let first_counts = (
            context.cached_ptx_count().unwrap(),
            context.cached_module_count().unwrap(),
            context.cached_lut_count().unwrap(),
            context.cached_transient_buffer_count().unwrap(),
        );
        assert!(first_counts.0 > 0);
        assert!(first_counts.1 > 0);
        assert!(first_counts.2 > 0);
        assert!(first_counts.3 > 0);
        let output_bytes = length * std::mem::size_of::<Complex32>();
        let first_host_output_ptr = {
            let pool = context.host_output_pool.lock().unwrap();
            assert_eq!(pool.len(), 1);
            let buffer = pool.get(&output_bytes).expect("cached CUDA output buffer");
            assert_eq!(buffer.len(), output_bytes);
            buffer.as_ptr()
        };

        let second = context.execute_program_complex32(&source, &input).unwrap();
        let second_counts = (
            context.cached_ptx_count().unwrap(),
            context.cached_module_count().unwrap(),
            context.cached_lut_count().unwrap(),
            context.cached_transient_buffer_count().unwrap(),
        );
        assert_eq!(first, second);
        assert_eq!(first_counts, second_counts);
        let second_host_output_ptr = {
            let pool = context.host_output_pool.lock().unwrap();
            assert_eq!(pool.len(), 1);
            pool.get(&output_bytes)
                .expect("recycled CUDA output buffer")
                .as_ptr()
        };
        assert_eq!(first_host_output_ptr, second_host_output_ptr);

        let archive_data = context.ptx_cache_archive_data().unwrap();
        let archive = CudaPtxCacheArchive::decode(&archive_data).unwrap();
        assert_eq!(archive.compute_capability(), context.compute_capability());
        assert_eq!(archive.entry_count(), first_counts.0);
        let mut wrong_capability = archive.clone();
        wrong_capability.compute_capability.1 += 1;
        let wrong_archive_data = wrong_capability.encode();

        context.clear_runtime_caches().unwrap();
        assert_eq!(context.cached_ptx_count().unwrap(), 0);
        assert_eq!(context.cached_module_count().unwrap(), 0);
        assert_eq!(context.cached_lut_count().unwrap(), 0);
        assert_eq!(context.cached_transient_buffer_count().unwrap(), 0);
        assert!(context.host_output_pool.lock().unwrap().is_empty());
        drop(context);

        assert!(matches!(
            new_test_context_with_ptx_cache_archive(0, &wrong_archive_data),
            Err(VkFftError::NativeRuntime { .. })
        ));
        let restored = new_test_context_with_ptx_cache_archive(0, &archive_data).unwrap();
        assert_eq!(restored.cached_ptx_count().unwrap(), first_counts.0);
        assert_eq!(restored.cached_module_count().unwrap(), 0);
        assert_eq!(restored.cached_lut_count().unwrap(), 0);
        assert_eq!(restored.cached_transient_buffer_count().unwrap(), 0);
        assert!(restored.host_output_pool.lock().unwrap().is_empty());
        let restored_output = restored.execute_program_complex32(&source, &input).unwrap();
        assert_eq!(restored_output, first);
        assert_eq!(restored.cached_ptx_count().unwrap(), first_counts.0);
        assert!(restored.cached_module_count().unwrap() > 0);
        assert!(restored.cached_lut_count().unwrap() > 0);
        assert!(restored.cached_transient_buffer_count().unwrap() > 0);
        assert_eq!(restored.host_output_pool.lock().unwrap().len(), 1);
    }

    #[test]
    fn cuda_bluestein_even_real_boundaries_match_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        crate::backend::native_runtime::assert_native_bluestein_even_real(&*context);
    }

    #[test]
    fn cuda_high_level_full_complex_even_real_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 578usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                (0.083 * x).sin() + 0.19 * (0.031 * x).cos() + x * 0.0002
            })
            .collect::<Vec<_>>();
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_transform(crate::TransformKind::RealToComplex),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Real(real) = &forward else {
            panic!("expected real transform");
        };
        assert_eq!(real.algorithm, crate::RealFftAlgorithm::FullComplex);
        assert_eq!(real.transform_len(), length);
        assert!(matches!(real.transform, crate::OneDimFftIr::Recursive(_)));
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("CUDA recursive R2C returned real output"),
        };
        let expected = forward
            .execute_r2c_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        assert!(max_error32(&spectrum, &expected) <= 3.0e-3 * length as f64);

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Real(real) = &inverse else {
            panic!("expected real transform");
        };
        assert_eq!(real.algorithm, crate::RealFftAlgorithm::FullComplex);
        assert_eq!(real.transform_len(), length);
        assert!(matches!(real.transform, crate::OneDimFftIr::Recursive(_)));
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA recursive C2R returned complex output")
            }
        };
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let inverse_expected = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let oracle_error = restored
            .iter()
            .zip(&inverse_expected)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 3.5e-3 * length as f32 && oracle_error <= 3.5e-3 * length as f64,
            "CUDA recursive C2R errors: round_trip={round_trip_error}, oracle={oracle_error}"
        );
    }

    #[test]
    fn cuda_real_spatial_zero_padding_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 34usize;
        let left = 17usize;
        let mut input = (0..length)
            .map(|index| {
                let x = index as f32;
                (0.071 * x).sin() + 0.23 * (0.041 * x).cos()
            })
            .collect::<Vec<_>>();
        for (offset, value) in input[left..].iter_mut().enumerate() {
            *value = 10_000.0 + offset as f32;
        }
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::RealToComplex)
                .with_zero_padding(0, left, length)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("CUDA zero-padded R2C returned real output"),
        };
        let expected = forward
            .execute_r2c_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        assert!(max_error32(&spectrum, &expected) <= 2.5e-3 * length as f64);

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_zero_padding(0, left, length)
                .unwrap(),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA zero-padded C2R returned complex output")
            }
        };
        assert!(restored[left..].iter().all(|value| *value == 0.0));
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let inverse_expected = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let cuda_oracle_error = restored[..left]
            .iter()
            .zip(&inverse_expected[..left])
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        let oracle_round_trip_error = inverse_expected[..left]
            .iter()
            .zip(&input[..left])
            .map(|(actual, expected)| (actual - *expected as f64).abs())
            .fold(0.0, f64::max);
        assert!(
            cuda_oracle_error <= 3.0e-3 * length as f64
                && oracle_round_trip_error <= 3.0e-3 * length as f64,
            "CUDA zero-padded C2R errors: cuda_oracle={cuda_oracle_error}, oracle_round_trip={oracle_round_trip_error}"
        );
    }

    #[test]
    fn cuda_high_level_real_and_r2r_paths_match_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 30usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                (0.12 * x).sin() + 0.21 * (0.035 * x).cos() + x * 0.0007
            })
            .collect::<Vec<_>>();
        let forward = TransformIr::build(
            FftConfig::new(vec![length]).with_transform(crate::TransformKind::RealToComplex),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("CUDA R2C returned real output"),
        };
        let expected = forward
            .execute_r2c_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        assert!(max_error32(&spectrum, &expected) <= 2.0e-3 * length as f64);

        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => panic!("CUDA C2R returned complex output"),
        };
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected_inverse = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&expected_inverse)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 3.0e-3 * length as f64,
            "CUDA C2R mismatch against inverse oracle: {inverse_error}"
        );
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        assert!(
            round_trip_error <= 3.0e-3 * length as f32,
            "CUDA R2C/C2R round-trip error: {round_trip_error}"
        );

        let dct = TransformIr::build(
            FftConfig::new(vec![12]).with_transform(crate::TransformKind::Dct(crate::DctType::II)),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let dct_input = input[..12].to_vec();
        let actual = match context
            .execute_transform_f32(&dct, NativeTransformInput32::Real(&dct_input))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => panic!("CUDA DCT returned complex output"),
        };
        let expected = dct
            .execute_r2r_reference(
                &dct_input
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(actual.iter().zip(&expected).all(|(actual, expected)| {
            (*actual as f64 - expected).abs() <= 3.0e-3 * dct_input.len() as f64
        }));
    }

    #[test]
    fn cuda_prime_rader_and_nd_programs_match_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for (length, batch_count) in [
            (17usize, 1usize),
            (17, 32),
            (19, 32),
            (29, 32),
            (47, 32),
            (257, 1),
            (257, 32),
        ] {
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new((0.073 * x).sin() + x * 0.0002, (0.031 * x).cos())
                })
                .collect::<Vec<_>>();
            let ir = TransformIr::build(
                FftConfig::new(vec![length]).with_batch_count(batch_count),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            if matches!(length, 19 | 29 | 257) {
                let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                    panic!("CUDA batched FFT-Rader should use recursive Rader IR");
                };
                let crate::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
                    panic!("CUDA batched FFT-Rader should keep an FFT-Rader root");
                };
                let block = rader
                    .axis_batch_block
                    .expect("CUDA batched FFT-Rader should use axis batching");
                let (expected_group, expected_threads, expected_local) = match (length, batch_count)
                {
                    (19, _) => (32, 4, [32, 4]),
                    (29, _) => (32, 5, [32, 5]),
                    (257, 1) => (1, 17, [17, 1]),
                    (257, _) => (7, 17, [17, 7]),
                    _ => unreachable!(),
                };
                assert_eq!(block.grouped_batch, expected_group);
                assert_eq!(block.threads_per_transform, expected_threads);
                assert_eq!([block.local_size_x, block.local_size_y], expected_local);
                assert_eq!(
                    rader
                        .internal_register_schedule
                        .as_ref()
                        .map(|schedule| schedule.container_fft_num),
                    Some(1)
                );
            }
            if length == 47 {
                let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                    panic!("CUDA p47/batch32 should use recursive direct Rader IR");
                };
                let crate::RecursiveFftNodeIr::DirectRader(rader) = &recursive.root else {
                    panic!("CUDA p47/batch32 should keep a direct-Rader root");
                };
                let block = rader
                    .axis_batch_block
                    .expect("CUDA p47/batch32 should use direct-Rader batching");
                assert_eq!(block.threads_per_transform, 24);
                assert_eq!(block.grouped_batch, 5);
                assert_eq!([rader.workgroup_size.x, rader.workgroup_size.y], [24, 5]);
                assert_eq!(rader.dispatch.x, 7);
            }
            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let expected_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected = ir.execute_complex_reference(&expected_input).unwrap();
            assert!(
                max_error32(&actual, &expected) <= 3.5e-3 * length as f64,
                "CUDA prime-length {length} mismatch on {}",
                context.device_name()
            );
        }

        let dimensions = vec![3usize, 8];
        let len = dimensions.iter().product::<usize>();
        let input = (0..len)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.055 * x).sin(), (0.021 * x).cos() - x * 0.0004)
            })
            .collect::<Vec<_>>();
        let ir = TransformIr::build(
            FftConfig::new(dimensions),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::ComplexNd(nd) = &ir else {
            panic!("CUDA [3,8] should use ND C2C IR");
        };
        let outer = nd
            .axes
            .iter()
            .find(|axis| axis.axis == 0)
            .expect("CUDA ND outer axis");
        let crate::OneDimFftIr::Recursive(outer_recursive) = &outer.transform else {
            panic!("CUDA ND outer Stockham axis unexpectedly selected Bluestein");
        };
        let crate::RecursiveFftNodeIr::Stockham(outer_kernel) = &outer_recursive.root else {
            panic!("CUDA ND outer Stockham axis unexpectedly selected Rader");
        };
        assert_eq!(
            [outer_kernel.workgroup_size.x, outer_kernel.workgroup_size.y],
            [8, 1]
        );
        assert_eq!(outer_kernel.dispatch.x, 1);
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        assert!(max_error32(&actual, &expected) <= 3.0e-3 * len as f64);
    }

    #[test]
    fn cuda_multidimensional_real_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let grouped_batch = 3usize;
        let batch_count = 5usize;
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for (dimensions, padded) in [
                (vec![3usize, 7], false),
                (vec![3usize, 8], false),
                (vec![3usize, 8], true),
            ] {
                let full_len = dimensions.iter().product::<usize>();
                let elements = full_len * batch_count;
                let input = (0..elements)
                    .map(|index| {
                        let x = index as f32;
                        (0.093 * x).sin() + 0.16 * (0.029 * x).cos() + x * 0.0002
                    })
                    .collect::<Vec<_>>();
                let mut forward_config = FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap()
                    .with_transform(crate::TransformKind::RealToComplex);
                if padded {
                    forward_config = forward_config
                        .with_zero_padding(
                            dimensions.len() - 1,
                            dimensions.last().unwrap() - 2,
                            *dimensions.last().unwrap(),
                        )
                        .unwrap();
                }
                let forward = TransformIr::build(
                    forward_config,
                    Direction::Forward,
                    context.device_profile(),
                )
                .unwrap();
                let TransformIr::RealNd(nd_forward) = &forward else {
                    panic!("grouped CUDA ND R2C should build ordinary ND real IR");
                };
                assert_eq!(nd_forward.real_grouped_batch, Some(grouped_batch));
                assert_eq!(nd_forward.real_axis.grouped_batch, grouped_batch);
                assert!(nd_forward.complex_axes.iter().all(|axis| {
                    axis.pack.grouped_batch == Some(grouped_batch)
                        && axis.scatter.grouped_batch == Some(grouped_batch)
                        && axis.pack.dispatch.x == 2
                        && axis.scatter.dispatch.x == 2
                        && axis.transform.grouped_batch() == grouped_batch
                }));
                let spectrum = match context
                    .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
                    .unwrap()
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => {
                        panic!("CUDA ND mixed R2C returned real output")
                    }
                };
                let oracle_input = input
                    .iter()
                    .map(|value| {
                        if precision == Precision::F16StorageF32Compute {
                            crate::Binary16::from_f32(*value).to_f32() as f64
                        } else {
                            *value as f64
                        }
                    })
                    .collect::<Vec<_>>();
                let mut expected = forward.execute_r2c_reference(&oracle_input).unwrap();
                for value in &mut expected {
                    if precision == Precision::F16StorageF32Compute {
                        value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                        value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                    } else {
                        value.re = value.re as f32 as f64;
                        value.im = value.im as f32 as f64;
                    }
                }
                let tolerance = if precision == Precision::F16StorageF32Compute {
                    5.0e-3 * full_len as f64
                } else {
                    5.0e-5 * full_len as f64
                };
                assert!(max_error32(&spectrum, &expected) <= tolerance);

                let mut inverse_config = FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap()
                    .with_transform(crate::TransformKind::ComplexToReal)
                    .with_inverse_normalization(true);
                if padded {
                    inverse_config = inverse_config
                        .with_zero_padding(
                            dimensions.len() - 1,
                            dimensions.last().unwrap() - 2,
                            *dimensions.last().unwrap(),
                        )
                        .unwrap();
                }
                let inverse = TransformIr::build(
                    inverse_config,
                    Direction::Inverse,
                    context.device_profile(),
                )
                .unwrap();
                let restored = match context
                    .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
                    .unwrap()
                {
                    NativeTransformOutput32::Real(values) => values,
                    NativeTransformOutput32::Complex(_) => {
                        panic!("CUDA ND mixed C2R returned complex output")
                    }
                };
                let spectrum64 = spectrum
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let mut expected_real = inverse.execute_c2r_reference(&spectrum64).unwrap();
                for value in &mut expected_real {
                    if precision == Precision::F16StorageF32Compute {
                        *value = crate::Binary16::from_f32(*value as f32).to_f32() as f64;
                    } else {
                        *value = *value as f32 as f64;
                    }
                }
                let inverse_error = restored
                    .iter()
                    .zip(&expected_real)
                    .map(|(actual, expected)| (*actual as f64 - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    inverse_error <= tolerance,
                    "CUDA ND mixed C2R {:?} padded={padded} error {inverse_error}",
                    dimensions
                );
            }
        }
    }

    #[test]
    fn cuda_medium_recursive_register_and_nd_real_r2r_match_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for length in [289usize, 323, 3840] {
            let input = (0..length)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new((0.017 * x).sin() + x * 0.00003, (0.023 * x).cos())
                })
                .collect::<Vec<_>>();
            let ir = TransformIr::build(
                FftConfig::new(vec![length]),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let expected_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected =
                crate::reference::fft(&expected_input, Direction::Forward, false).unwrap();
            assert!(
                max_error32(&actual, &expected) <= 8.0e-4 * length as f64,
                "CUDA medium path mismatch for N={length}"
            );
        }

        let dimensions = vec![3usize, 8];
        let full_len = dimensions.iter().product::<usize>();
        let real_input = (0..full_len)
            .map(|index| {
                let x = index as f32;
                (0.13 * x).sin() + 0.27 * (0.047 * x).cos() + x * 0.0004
            })
            .collect::<Vec<_>>();
        let forward = TransformIr::build(
            FftConfig::new(dimensions.clone()).with_transform(crate::TransformKind::RealToComplex),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let spectrum = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&real_input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("CUDA ND R2C returned real output"),
        };
        let expected = forward
            .execute_r2c_reference(
                &real_input
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(max_error32(&spectrum, &expected) <= 2.0e-3 * full_len as f64);

        let inverse = TransformIr::build(
            FftConfig::new(dimensions)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => panic!("CUDA ND C2R returned complex output"),
        };
        let spectrum64 = spectrum
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let inverse_expected = inverse.execute_c2r_reference(&spectrum64).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&inverse_expected)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        let round_trip_error = restored
            .iter()
            .zip(&real_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        assert!(
            inverse_error <= 3.0e-3 * full_len as f64,
            "CUDA ND C2R inverse mismatch: {inverse_error}"
        );
        assert!(
            round_trip_error <= 3.0e-3 * full_len as f32,
            "CUDA ND real round-trip mismatch: {round_trip_error}"
        );

        let dimensions = vec![3usize, 4];
        let tensor_len = dimensions.iter().product::<usize>();
        let dct_input = real_input[..tensor_len].to_vec();
        let dct = TransformIr::build(
            FftConfig::new(dimensions)
                .with_transform(crate::TransformKind::Dct(crate::DctType::II)),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let actual = match context
            .execute_transform_f32(&dct, NativeTransformInput32::Real(&dct_input))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => panic!("CUDA ND DCT returned complex output"),
        };
        let expected = dct
            .execute_r2r_reference(
                &dct_input
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual as f64 - expected).abs()
                    <= 3.0e-3 * tensor_len as f64)
        );
    }

    #[test]
    fn ordinary_nd_r2r_padding_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dimensions = vec![3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 5usize;
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f32;
                (0.13 * x).sin() + 0.23 * (0.041 * x).cos() + 0.0009 * x
            })
            .collect::<Vec<_>>();
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap()
                    .with_zero_padding(0, 1, 3)
                    .unwrap()
                    .with_zero_padding(1, 2, 4)
                    .unwrap(),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let expected = forward
            .execute_r2r_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        let actual = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA padded ND DCT returned complex output")
            }
        };
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 4.0e-3 * tensor_len as f64,
            "CUDA padded ND DCT-II forward mismatch: {forward_error}"
        );

        let inverse = build(Direction::Inverse);
        let actual64 = actual.iter().map(|value| *value as f64).collect::<Vec<_>>();
        let inverse_expected = inverse.execute_r2r_reference(&actual64).unwrap();
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Real(&actual))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA padded inverse ND DCT returned complex output")
            }
        };
        let inverse_error = restored
            .iter()
            .zip(&inverse_expected)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 4.0e-3 * tensor_len as f64,
            "CUDA padded inverse ND DCT-II mismatch: {inverse_error}"
        );
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for n0 in 0..dimensions[0] {
                for n1 in 0..dimensions[1] {
                    if (1..3).contains(&n0) || (2..4).contains(&n1) {
                        assert_eq!(restored[base + n0 * dimensions[1] + n1], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn cuda_r2r_mixed_storage_direct_fft_and_nd_match_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let grouped_batch = 3usize;
        let grouped_batch_count = 5usize;
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for (dimensions, transform) in [
                (vec![9usize], crate::TransformKind::Dct(crate::DctType::I)),
                (vec![9usize], crate::TransformKind::Dst(crate::DstType::I)),
                (vec![9usize], crate::TransformKind::Dct(crate::DctType::II)),
                (vec![9usize], crate::TransformKind::Dst(crate::DstType::III)),
                (vec![8usize], crate::TransformKind::Dct(crate::DctType::IV)),
                (vec![8usize], crate::TransformKind::Dst(crate::DstType::IV)),
                (vec![9usize], crate::TransformKind::Dct(crate::DctType::IV)),
                (vec![9usize], crate::TransformKind::Dst(crate::DstType::IV)),
                (
                    vec![3usize, 4],
                    crate::TransformKind::Dct(crate::DctType::I),
                ),
                (
                    vec![3usize, 4],
                    crate::TransformKind::Dst(crate::DstType::I),
                ),
                (
                    vec![3usize, 4],
                    crate::TransformKind::Dct(crate::DctType::II),
                ),
                (
                    vec![3usize, 4],
                    crate::TransformKind::Dst(crate::DstType::III),
                ),
                (
                    vec![4usize, 8],
                    crate::TransformKind::Dct(crate::DctType::IV),
                ),
                (
                    vec![4usize, 8],
                    crate::TransformKind::Dst(crate::DstType::IV),
                ),
            ] {
                let shape_elements = dimensions.iter().product::<usize>();
                let batch_count = grouped_batch_count;
                let elements = shape_elements * batch_count;
                let input = (0..elements)
                    .map(|index| {
                        let x = index as f32;
                        (0.079 * x).sin() + 0.17 * (0.035 * x).cos() + x * 0.0003
                    })
                    .collect::<Vec<_>>();
                for direction in [Direction::Forward, Direction::Inverse] {
                    let mut config = FftConfig::new(dimensions.clone())
                        .with_batch_count(batch_count)
                        .with_precision(precision)
                        .with_transform(transform)
                        .with_inverse_normalization(direction == Direction::Inverse);
                    for axis in 0..dimensions.len() {
                        config = config.with_grouped_batch(axis, grouped_batch).unwrap();
                    }
                    let ir =
                        TransformIr::build(config, direction, context.device_profile()).unwrap();
                    if dimensions.len() == 1
                        && matches!(
                            transform,
                            crate::TransformKind::Dct(
                                crate::DctType::I | crate::DctType::II | crate::DctType::IV,
                            ) | crate::TransformKind::Dst(
                                crate::DstType::I | crate::DstType::III | crate::DstType::IV,
                            )
                        )
                    {
                        let TransformIr::RealToReal(r2r) = &ir else {
                            panic!("1D mixed CUDA R2R should build ordinary R2R IR");
                        };
                        assert_eq!(r2r.grouped_batch, grouped_batch);
                        assert_eq!(r2r.dispatch.x, 2);
                        let reduction = r2r.fft_reduction.as_deref().unwrap();
                        assert_eq!(reduction.grouped_batch, grouped_batch);
                        assert_eq!(reduction.preprocess.dispatch.x, 2);
                        assert_eq!(reduction.postprocess.dispatch.x, 2);
                        let expected_fft_len = match transform {
                            crate::TransformKind::Dct(crate::DctType::I) => 16,
                            crate::TransformKind::Dst(crate::DstType::I) => 20,
                            crate::TransformKind::Dct(crate::DctType::II)
                            | crate::TransformKind::Dst(crate::DstType::III) => 9,
                            crate::TransformKind::Dct(crate::DctType::IV)
                            | crate::TransformKind::Dst(crate::DstType::IV) => {
                                if dimensions[0].is_multiple_of(2) {
                                    dimensions[0] / 2
                                } else {
                                    2 * dimensions[0]
                                }
                            }
                            _ => unreachable!(),
                        };
                        assert_eq!(reduction.fft_len, expected_fft_len);
                    }
                    if dimensions.len() > 1 {
                        let TransformIr::RealToRealNd(r2r) = &ir else {
                            panic!("grouped CUDA ND R2R should build ordinary ND R2R IR");
                        };
                        assert!(r2r.axes.iter().all(|axis| {
                            axis.pack.grouped_batch == Some(grouped_batch)
                                && axis.scatter.grouped_batch == Some(grouped_batch)
                                && axis.pack.dispatch.x == 2
                                && axis.scatter.dispatch.x == 2
                                && axis.transform.grouped_batch == grouped_batch
                        }));
                    }
                    let actual = match context
                        .execute_transform_f32(&ir, NativeTransformInput32::Real(&input))
                        .unwrap()
                    {
                        NativeTransformOutput32::Real(values) => values,
                        NativeTransformOutput32::Complex(_) => {
                            panic!("mixed CUDA R2R returned complex output")
                        }
                    };
                    let oracle_input = input
                        .iter()
                        .map(|value| {
                            if precision == Precision::F16StorageF32Compute {
                                crate::Binary16::from_f32(*value).to_f32() as f64
                            } else {
                                *value as f64
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut expected = ir.execute_r2r_reference(&oracle_input).unwrap();
                    for value in &mut expected {
                        if precision == Precision::F16StorageF32Compute {
                            *value = crate::Binary16::from_f32(*value as f32).to_f32() as f64;
                        } else {
                            *value = *value as f32 as f64;
                        }
                    }
                    let error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(actual, expected)| (*actual as f64 - expected).abs())
                        .fold(0.0, f64::max);
                    let tolerance = if precision == Precision::F16StorageF32Compute {
                        8.0e-3 * elements as f64
                    } else {
                        1.0e-4 * elements as f64
                    };
                    assert!(
                        error <= tolerance,
                        "CUDA mixed R2R {:?} {transform:?} {direction:?} error {error}",
                        dimensions
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_spatial_zero_padding_matches_manual_zeroing_across_algorithms_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for (length, force_bluestein) in [(64usize, false), (103, true), (257, false)] {
            let left = length / 2;
            let tuning = if force_bluestein {
                let mut tuning = crate::PlannerTuning::portable();
                tuning.max_rader_fft_prime = 100;
                tuning
            } else {
                crate::PlannerTuning::default()
            };
            let config = FftConfig::new(vec![length])
                .with_tuning(tuning)
                .with_zero_padding(0, left, length)
                .unwrap();
            let ir =
                TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
            let mut input = (0..length)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new((0.031 * x).sin(), (0.019 * x).cos())
                })
                .collect::<Vec<_>>();
            for (offset, value) in input[left..].iter_mut().enumerate() {
                *value = Complex32::new(10_000.0 + offset as f32, -20_000.0 - offset as f32);
            }
            let mut expected_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            expected_input[left..].fill(Complex64::default());
            let expected =
                crate::reference::fft(&expected_input, Direction::Forward, false).unwrap();
            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            assert!(
                max_error32(&actual, &expected) <= 1.0e-3 * length as f64,
                "CUDA zero-padding mismatch for N={length}"
            );
        }

        let length = 64usize;
        let left = 32usize;
        let config = FftConfig::new(vec![length])
            .with_inverse_normalization(true)
            .with_zero_padding(0, left, length)
            .unwrap();
        let ir = TransformIr::build(config, Direction::Inverse, context.device_profile()).unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.021 * x).sin(), (0.017 * x).cos())
            })
            .collect::<Vec<_>>();
        let input64 = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let mut expected = crate::reference::fft(&input64, Direction::Inverse, true).unwrap();
        expected[left..].fill(Complex64::default());
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 1.0e-3 * length as f64);
        assert!(
            actual[left..]
                .iter()
                .all(|value| value.re == 0.0 && value.im == 0.0)
        );
    }

    #[test]
    fn cuda_multidimensional_spatial_zero_padding_matches_cpu_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dimensions = vec![3usize, 4];
        let config = FftConfig::new(dimensions.clone())
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 3, 4)
            .unwrap();
        let ir = TransformIr::build(config, Direction::Forward, context.device_profile()).unwrap();
        let mut input = (0..12)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.17 * x).sin(), (0.11 * x).cos())
            })
            .collect::<Vec<_>>();
        for row in 0..3 {
            for col in 0..4 {
                if row == 2 || col == 3 {
                    input[row * 4 + col] =
                        Complex32::new(10_000.0 + row as f32, -20_000.0 - col as f32);
                }
            }
        }
        let expected_input = input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let row = index / 4;
                let col = index % 4;
                if row == 2 || col == 3 {
                    Complex64::default()
                } else {
                    Complex64::new(value.re as f64, value.im as f64)
                }
            })
            .collect::<Vec<_>>();
        let reference = TransformIr::build(
            FftConfig::new(dimensions),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let expected = reference
            .execute_complex_reference(&expected_input)
            .unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        assert!(max_error32(&actual, &expected) <= 2.0e-3 * 12.0);
    }

    #[test]
    fn cuda_r2c_even_half_p17_stays_finite_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut failures = Vec::new();
        for dimensions in [
            vec![34usize],
            vec![32],
            vec![33],
            vec![18],
            vec![26],
            vec![17, 8],
            vec![3, 34],
            vec![17, 34],
            vec![17, 32],
            vec![17, 33],
        ] {
            let len = dimensions.iter().product::<usize>();
            let input = (0..len)
                .map(|index| {
                    let x = index as f32;
                    (0.057 * x).sin() + 0.19 * (0.017 * x).cos() - 0.00031 * x
                })
                .collect::<Vec<_>>();
            let ir = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_transform(crate::TransformKind::RealToComplex),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let output = match context
                .execute_transform_f32(&ir, NativeTransformInput32::Real(&input))
                .unwrap()
            {
                NativeTransformOutput32::Complex(values) => values,
                NativeTransformOutput32::Real(_) => panic!("CUDA R2C returned real output"),
            };
            let non_finite = output
                .iter()
                .filter(|value| !value.re.is_finite() || !value.im.is_finite())
                .count();
            if non_finite != 0 {
                failures.push((dimensions, non_finite, output.len()));
            }
        }
        assert!(
            failures.is_empty(),
            "CUDA R2C non-finite cases: {failures:?}"
        );
    }

    #[test]
    fn cuda_precision_matrix_extends_to_nd_real_and_r2r_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        crate::backend::native_runtime::assert_native_precision_matrix(&*context);
    }

    #[test]
    fn cuda_precision_sweep_tracks_upstream_four_error_metrics_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }

        for (length, force_bluestein) in [(64usize, false), (103, true), (257, false), (384, false)]
        {
            let tuning = if force_bluestein {
                let mut tuning = crate::PlannerTuning::portable();
                tuning.max_rader_fft_prime = 100;
                tuning
            } else {
                crate::PlannerTuning::default()
            };
            let input32 = (0..length)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new(
                        (0.071 * x).sin() + x * 0.00017,
                        (0.043 * x).cos() - x * 0.00009,
                    )
                })
                .collect::<Vec<_>>();
            let input64 = input32
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected = crate::reference::fft(&input64, Direction::Forward, false).unwrap();

            let ir32 = TransformIr::build(
                FftConfig::new(vec![length]).with_tuning(tuning),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let ir64 = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_precision(Precision::F64),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let actual32 = context
                .execute_transform_complex32(&ir32, &input32)
                .unwrap();
            let actual64 = context
                .execute_transform_complex64(&ir64, &input64)
                .unwrap();
            let actual32_promoted = actual32
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let f32_metrics =
                crate::complex_precision_metrics(&actual32_promoted, &expected).unwrap();
            let f64_metrics = crate::complex_precision_metrics(&actual64, &expected).unwrap();
            for value in [
                f32_metrics.avg_difference,
                f32_metrics.max_difference,
                f32_metrics.avg_eps,
                f32_metrics.max_eps,
                f64_metrics.avg_difference,
                f64_metrics.max_difference,
                f64_metrics.avg_eps,
                f64_metrics.max_eps,
            ] {
                assert!(
                    value.is_finite(),
                    "non-finite CUDA precision metric for N={length}"
                );
            }
            assert!(
                f32_metrics.max_difference <= 5.0e-3 * length as f64,
                "CUDA F32 max difference too large for N={length}: {f32_metrics:?}"
            );
            assert!(
                f32_metrics.avg_eps <= 5.0e-4,
                "CUDA F32 average relative epsilon too large for N={length}: {f32_metrics:?}"
            );
            assert!(
                f64_metrics.max_difference <= 1.0e-8 * length as f64,
                "CUDA F64 max difference too large for N={length}: {f64_metrics:?}"
            );
            assert!(
                f64_metrics.avg_eps <= 1.0e-9,
                "CUDA F64 average relative epsilon too large for N={length}: {f64_metrics:?}"
            );
            assert!(
                f64_metrics.avg_difference <= f32_metrics.avg_difference * 0.1,
                "CUDA F64 should materially improve average error for N={length}: f32={f32_metrics:?}, f64={f64_metrics:?}"
            );
        }
    }

    #[test]
    fn cuda_f16_storage_f32_compute_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 64usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.071 * x).sin() + x * 0.0002, (0.029 * x).cos())
            })
            .collect::<Vec<_>>();
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::F16StorageF32Compute),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        let quantized_input = input
            .iter()
            .map(|value| {
                Complex64::new(
                    crate::Binary16::from_f32(value.re).to_f32() as f64,
                    crate::Binary16::from_f32(value.im).to_f32() as f64,
                )
            })
            .collect::<Vec<_>>();
        let expected = ir
            .execute_complex_reference(&quantized_input)
            .unwrap()
            .into_iter()
            .map(|value| {
                Complex32::new(
                    crate::Binary16::from_f32(value.re as f32).to_f32(),
                    crate::Binary16::from_f32(value.im as f32).to_f32(),
                )
            })
            .collect::<Vec<_>>();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            error <= 1.0e-3 * length as f32,
            "CUDA F16-storage error {error}"
        );
    }

    #[test]
    fn cuda_f64_compute_f32_storage_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 64usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.071 * x).sin() + x * 0.0002, (0.029 * x).cos())
            })
            .collect::<Vec<_>>();
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::F64ComputeF32Storage),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&expected_input).unwrap();
        let error = max_error32(&actual, &expected);
        assert!(
            error <= 2.0e-5 * length as f64,
            "CUDA mixed-storage error {error}"
        );
    }

    #[test]
    fn cuda_direct_rader_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 47usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.057 * x).sin() + x * 0.00021, (0.027 * x).cos())
            })
            .collect::<Vec<_>>();
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            let ir = TransformIr::build(
                FftConfig::new(vec![length]).with_precision(precision),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("CUDA mixed-storage p47 did not build recursive C2C");
            };
            let crate::RecursiveFftNodeIr::DirectRader(rader) = &recursive.root else {
                panic!("CUDA mixed-storage p47 did not keep direct Rader root");
            };
            let expected_storage = if precision == Precision::F16StorageF32Compute {
                crate::ScalarType::F16
            } else {
                crate::ScalarType::F32
            };
            assert_eq!(rader.input_storage_scalar, expected_storage);
            assert_eq!(rader.output_storage_scalar, expected_storage);

            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| {
                    if precision == Precision::F16StorageF32Compute {
                        Complex64::new(
                            crate::Binary16::from_f32(value.re).to_f32() as f64,
                            crate::Binary16::from_f32(value.im).to_f32() as f64,
                        )
                    } else {
                        Complex64::new(value.re as f64, value.im as f64)
                    }
                })
                .collect::<Vec<_>>();
            let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
            if precision == Precision::F16StorageF32Compute {
                for value in &mut expected {
                    value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                    value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                }
            }
            let error = max_error32(&actual, &expected);
            let tolerance = if precision == Precision::F16StorageF32Compute {
                1.0e-3 * length as f64
            } else {
                3.0e-5 * length as f64
            };
            assert!(
                error <= tolerance,
                "CUDA direct-Rader mixed-storage error {error}"
            );
        }
    }

    #[test]
    fn cuda_fft_rader_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 19usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.061 * x).sin() + x * 0.00027, (0.031 * x).cos())
            })
            .collect::<Vec<_>>();
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            let ir = TransformIr::build(
                FftConfig::new(vec![length]).with_precision(precision),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("CUDA mixed-storage p19 did not build recursive C2C");
            };
            let crate::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
                panic!("CUDA mixed-storage p19 did not keep FFT-Rader root");
            };
            let expected_storage = if precision == Precision::F16StorageF32Compute {
                crate::ScalarType::F16
            } else {
                crate::ScalarType::F32
            };
            assert_eq!(rader.input_storage_scalar, expected_storage);
            assert_eq!(rader.output_storage_scalar, expected_storage);

            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| {
                    if precision == Precision::F16StorageF32Compute {
                        Complex64::new(
                            crate::Binary16::from_f32(value.re).to_f32() as f64,
                            crate::Binary16::from_f32(value.im).to_f32() as f64,
                        )
                    } else {
                        Complex64::new(value.re as f64, value.im as f64)
                    }
                })
                .collect::<Vec<_>>();
            let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
            if precision == Precision::F16StorageF32Compute {
                for value in &mut expected {
                    value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                    value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                }
            }
            let error = max_error32(&actual, &expected);
            let tolerance = if precision == Precision::F16StorageF32Compute {
                1.0e-3 * length as f64
            } else {
                3.0e-5 * length as f64
            };
            assert!(
                error <= tolerance,
                "CUDA FFT-Rader mixed-storage error {error}"
            );
        }
    }

    #[test]
    fn cuda_f16_p47_thread_capped_rader_falls_back_to_bluestein_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut planning_profile = context.device_profile();
        if planning_profile.max_threads_per_block < 128
            || planning_profile.max_workgroup_size[0] < 128
        {
            return;
        }
        planning_profile.max_threads_per_block = 128;
        planning_profile.max_workgroup_size[0] = 128;
        planning_profile.max_workgroup_size[1] = planning_profile.max_workgroup_size[1].min(128);
        let length = 47usize;

        let f32 = TransformIr::build(
            FftConfig::new(vec![length]),
            Direction::Forward,
            planning_profile,
        )
        .unwrap();
        let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &f32 else {
            panic!("CUDA F32 p47 should remain Direct-Rader under the 63-prime cap");
        };
        assert!(matches!(
            &recursive.root,
            crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
        ));

        let build_f16 = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                planning_profile,
            )
            .unwrap()
        };
        let forward = build_f16(Direction::Forward);
        let inverse = build_f16(Direction::Inverse);
        let TransformIr::Complex1d(crate::OneDimFftIr::Bluestein(pipeline)) = &forward else {
            panic!("CUDA F16 p47 should fall back to Bluestein under the 31-prime cap");
        };
        assert_eq!(pipeline.external_storage_scalar(), crate::ScalarType::F16);

        let mut impulse = vec![Complex32::new(0.0, 0.0); length];
        impulse[1] = Complex32::new(1.0, 0.0);
        let spectrum = context
            .execute_transform_complex32(&forward, &impulse)
            .unwrap();
        for (k, actual) in spectrum.iter().enumerate() {
            let angle = -core::f32::consts::TAU * k as f32 / length as f32;
            let expected_re = crate::Binary16::from_f32(angle.cos()).to_f32();
            let expected_im = crate::Binary16::from_f32(angle.sin()).to_f32();
            assert!(
                (actual.re - expected_re).abs() <= 8.0e-3
                    && (actual.im - expected_im).abs() <= 8.0e-3,
                "CUDA F16 p47 Bluestein mismatch at bin {k}: actual={actual:?}, expected=({expected_re},{expected_im})"
            );
        }

        let mut dc = vec![Complex32::new(0.0, 0.0); length];
        dc[0] = Complex32::new(1.0, 0.0);
        let restored = context.execute_transform_complex32(&inverse, &dc).unwrap();
        let expected_dc = crate::Binary16::from_f32(1.0 / length as f32).to_f32();
        assert!(
            restored.iter().all(|value| {
                (value.re - expected_dc).abs() <= 2.0e-3 && value.im.abs() <= 2.0e-3
            })
        );
    }

    #[test]
    fn cuda_bluestein_mixed_storage_and_zero_padding_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 103usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    0.065 * (0.033 * x).sin() + 0.004 * (0.097 * x).cos(),
                    0.047 * (0.027 * x).cos() - 0.003 * (0.071 * x).sin(),
                )
            })
            .collect::<Vec<_>>();
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for direction in [Direction::Forward, Direction::Inverse] {
                for zero_padded in [false, true] {
                    let mut config = FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_tuning(tuning);
                    if zero_padded {
                        config = config.with_zero_padding(0, length / 2, length).unwrap();
                    }
                    let ir =
                        TransformIr::build(config, direction, context.device_profile()).unwrap();
                    let TransformIr::Complex1d(crate::OneDimFftIr::Bluestein(pipeline)) = &ir
                    else {
                        panic!("forced CUDA p103 mixed-storage plan did not build Bluestein");
                    };
                    let expected_storage = if precision == Precision::F16StorageF32Compute {
                        crate::ScalarType::F16
                    } else {
                        crate::ScalarType::F32
                    };
                    assert_eq!(pipeline.external_storage_scalar(), expected_storage);
                    assert_eq!(pipeline.grouped_batch, grouped_batch);
                    assert_eq!(pipeline.preprocess.dispatch.x, 2);
                    assert_eq!(pipeline.postprocess.dispatch.x, 2);
                    assert_eq!(
                        pipeline.forward_fft.axis0_grouped_batch_override,
                        Some(grouped_batch)
                    );
                    assert_eq!(
                        pipeline.inverse_fft.axis0_grouped_batch_override,
                        Some(grouped_batch)
                    );
                    if let Some(zero) = &pipeline.zero_pad_pass {
                        assert_eq!(zero.grouped_batch, grouped_batch);
                        assert_eq!(zero.dispatch.x, 2);
                    }

                    let actual = context.execute_transform_complex32(&ir, &input).unwrap();
                    let oracle_input = input
                        .iter()
                        .map(|value| {
                            if precision == Precision::F16StorageF32Compute {
                                Complex64::new(
                                    crate::Binary16::from_f32(value.re).to_f32() as f64,
                                    crate::Binary16::from_f32(value.im).to_f32() as f64,
                                )
                            } else {
                                Complex64::new(value.re as f64, value.im as f64)
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
                    for value in &mut expected {
                        if precision == Precision::F16StorageF32Compute {
                            value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                            value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                        } else {
                            value.re = value.re as f32 as f64;
                            value.im = value.im as f32 as f64;
                        }
                    }
                    let error = max_error32(&actual, &expected);
                    let tolerance = if precision == Precision::F16StorageF32Compute {
                        1.0e-3 * length as f64
                    } else {
                        3.0e-5 * length as f64
                    };
                    assert!(
                        error <= tolerance,
                        "CUDA Bluestein mixed-storage {precision:?} {direction:?} padding={zero_padded} error {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_explicit_fft_rader_mixed_storage_constrained_profile_executes_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 12_289usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    0.012 * (0.0017 * x).sin() + 0.004 * (0.0071 * x).cos(),
                    0.009 * (0.0023 * x).cos() - 0.003 * (0.0053 * x).sin(),
                )
            })
            .collect::<Vec<_>>();

        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            let mut planner_profile = context.device_profile();
            planner_profile.shared_memory_bytes = 8 * 1024;
            planner_profile.shared_memory_pow2_bytes = 8 * 1024;
            let ir = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
                Direction::Forward,
                planner_profile,
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("CUDA p12289 mixed-storage probe did not build recursive C2C");
            };
            let crate::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
                panic!("CUDA p12289 mixed-storage probe did not keep FFT-Rader root");
            };
            assert_eq!(
                rader.input_strategy,
                crate::RaderFftInputStrategy::GatherReversePass
            );
            assert!(rader.forward_recursive().unwrap().four_step_plan.is_some());
            assert!(rader.inverse_recursive().unwrap().four_step_plan.is_some());
            let expected_storage = if precision == Precision::F16StorageF32Compute {
                crate::ScalarType::F16
            } else {
                crate::ScalarType::F32
            };
            let expected_compute = if precision == Precision::F16StorageF32Compute {
                crate::ScalarType::F32
            } else {
                crate::ScalarType::F64
            };
            assert_eq!(rader.gather.input_storage_scalar, expected_storage);
            assert_eq!(rader.gather.output_storage_scalar, expected_compute);
            assert_eq!(rader.scatter.input_storage_scalar, expected_compute);
            assert_eq!(rader.scatter.output_storage_scalar, expected_storage);
            assert_eq!(rader.scatter.auxiliary_storage_scalar, expected_storage);

            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| {
                    if precision == Precision::F16StorageF32Compute {
                        Complex64::new(
                            crate::Binary16::from_f32(value.re).to_f32() as f64,
                            crate::Binary16::from_f32(value.im).to_f32() as f64,
                        )
                    } else {
                        Complex64::new(value.re as f64, value.im as f64)
                    }
                })
                .collect::<Vec<_>>();
            for k in [0usize, 1, 2, 7, 31, 257, 1024, 4096, 8192, 12_287, 12_288] {
                let mut expected = Complex64::new(0.0, 0.0);
                for (n, value) in oracle_input.iter().enumerate() {
                    let angle = -std::f64::consts::TAU * (k as f64) * (n as f64) / length as f64;
                    expected += *value * Complex64::new(angle.cos(), angle.sin());
                }
                if precision == Precision::F16StorageF32Compute {
                    expected.re = crate::Binary16::from_f32(expected.re as f32).to_f32() as f64;
                    expected.im = crate::Binary16::from_f32(expected.im as f32).to_f32() as f64;
                } else {
                    expected.re = expected.re as f32 as f64;
                    expected.im = expected.im as f32 as f64;
                }
                let dr = actual[k].re as f64 - expected.re;
                let di = actual[k].im as f64 - expected.im;
                let error = (dr * dr + di * di).sqrt();
                let tolerance = if precision == Precision::F16StorageF32Compute {
                    1.0e-3 * length as f64
                } else {
                    3.0e-5 * length as f64
                };
                assert!(
                    error <= tolerance,
                    "CUDA p12289 explicit mixed Rader {precision:?} bin {k} error {error}"
                );
            }
        }
    }

    #[test]
    fn cuda_recursive_p4001_zero_padding_f16_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 4001usize;
        let left = length / 2;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_zero_padding(0, left, length)
                .unwrap();
            let ir = TransformIr::build(config, direction, context.device_profile()).unwrap();
            let input = (0..length)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new(0.09 * (0.017 * x).sin(), 0.07 * (0.011 * x).cos())
                })
                .collect::<Vec<_>>();
            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| {
                    Complex64::new(
                        crate::Binary16::from_f32(value.re).to_f32() as f64,
                        crate::Binary16::from_f32(value.im).to_f32() as f64,
                    )
                })
                .collect::<Vec<_>>();
            let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
            for value in &mut expected {
                value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
            }
            let error = max_error32(&actual, &expected);
            assert!(
                error <= 1.0e-3 * length as f64,
                "CUDA p4001 F16-storage zero-pad error {direction:?}: {error}"
            );
            if direction == Direction::Inverse {
                assert!(
                    actual[left..]
                        .iter()
                        .all(|value| value.re == 0.0 && value.im == 0.0)
                );
            }
        }
    }

    #[test]
    fn cuda_rader_zero_padding_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for direction in [Direction::Forward, Direction::Inverse] {
                for length in [47usize, 19usize] {
                    let left = length / 2;
                    let config = FftConfig::new(vec![length])
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_zero_padding(0, left, length)
                        .unwrap();
                    let ir =
                        TransformIr::build(config, direction, context.device_profile()).unwrap();
                    let input = (0..length)
                        .map(|index| {
                            let x = index as f32;
                            Complex32::new(0.17 * (0.071 * x).sin(), 0.11 * (0.037 * x).cos())
                        })
                        .collect::<Vec<_>>();
                    let actual = context.execute_transform_complex32(&ir, &input).unwrap();
                    let oracle_input = input
                        .iter()
                        .map(|value| {
                            if precision == Precision::F16StorageF32Compute {
                                Complex64::new(
                                    crate::Binary16::from_f32(value.re).to_f32() as f64,
                                    crate::Binary16::from_f32(value.im).to_f32() as f64,
                                )
                            } else {
                                Complex64::new(value.re as f64, value.im as f64)
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
                    for value in &mut expected {
                        if precision == Precision::F16StorageF32Compute {
                            value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                            value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                        } else {
                            value.re = value.re as f32 as f64;
                            value.im = value.im as f32 as f64;
                        }
                    }
                    let error = max_error32(&actual, &expected);
                    let tolerance = if precision == Precision::F16StorageF32Compute {
                        1.0e-3 * length as f64
                    } else {
                        3.0e-5 * length as f64
                    };
                    assert!(
                        error <= tolerance,
                        "CUDA Rader zero-pad mixed-storage error N={length} {direction:?}: {error}"
                    );
                    if direction == Direction::Inverse {
                        assert!(
                            actual[left..]
                                .iter()
                                .all(|value| value.re == 0.0 && value.im == 0.0)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cuda_multidimensional_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dimensions = vec![3usize, 4usize];
        let batch_count = 2usize;
        let length = dimensions.iter().product::<usize>() * batch_count;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.083 * x).sin() + x * 0.0007, (0.047 * x).cos())
            })
            .collect::<Vec<_>>();
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for direction in [Direction::Forward, Direction::Inverse] {
                let ir = TransformIr::build(
                    FftConfig::new(dimensions.clone())
                        .with_batch_count(batch_count)
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    context.device_profile(),
                )
                .unwrap();
                let TransformIr::ComplexNd(nd) = &ir else {
                    panic!("CUDA 2D mixed-storage plan did not build ComplexNd");
                };
                let storage = if precision == Precision::F16StorageF32Compute {
                    crate::ScalarType::F16
                } else {
                    crate::ScalarType::F32
                };
                assert_eq!(nd.external_scalar, storage);

                let actual = context.execute_transform_complex32(&ir, &input).unwrap();
                let oracle_input = input
                    .iter()
                    .map(|value| {
                        if precision == Precision::F16StorageF32Compute {
                            Complex64::new(
                                crate::Binary16::from_f32(value.re).to_f32() as f64,
                                crate::Binary16::from_f32(value.im).to_f32() as f64,
                            )
                        } else {
                            Complex64::new(value.re as f64, value.im as f64)
                        }
                    })
                    .collect::<Vec<_>>();
                let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
                for value in &mut expected {
                    if precision == Precision::F16StorageF32Compute {
                        value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                        value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                    } else {
                        value.re = value.re as f32 as f64;
                        value.im = value.im as f32 as f64;
                    }
                }
                let error = max_error32(&actual, &expected);
                let tolerance = if precision == Precision::F16StorageF32Compute {
                    1.0e-3 * dimensions.iter().product::<usize>() as f64
                } else {
                    3.0e-5 * dimensions.iter().product::<usize>() as f64
                };
                assert!(
                    error <= tolerance,
                    "CUDA 2D mixed-storage error {direction:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn cuda_grouped_multidimensional_c2c_partial_tail_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dimensions = vec![3usize, 4usize];
        let batch_count = 5usize;
        let length = dimensions.iter().product::<usize>() * batch_count;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.089 * x).sin() + x * 0.0004, (0.039 * x).cos())
            })
            .collect::<Vec<_>>();

        for direction in [Direction::Forward, Direction::Inverse] {
            let ir = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap()
                    .with_inverse_normalization(direction == Direction::Inverse),
                direction,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::ComplexNd(nd) = &ir else {
                panic!("CUDA grouped 2D C2C plan did not build ComplexNd");
            };
            assert!(nd.axes.iter().all(|axis| {
                axis.pack.grouped_batch == Some(3)
                    && axis.scatter.grouped_batch == Some(3)
                    && axis.pack.dispatch.x == 2
                    && axis.scatter.dispatch.x == 2
                    && axis.transform.grouped_batch() == 3
            }));

            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected = ir.execute_complex_reference(&oracle_input).unwrap();
            let error = max_error32(&actual, &expected);
            assert!(
                error <= 3.0e-5 * dimensions.iter().product::<usize>() as f64,
                "CUDA grouped 2D C2C error {direction:?}: {error}"
            );
        }
    }

    #[test]
    fn cuda_grouped_higher_axis_two_upload_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut planning_profile = context.device_profile();
        if planning_profile.shared_memory_bytes < 32 * 1024 {
            return;
        }
        planning_profile.shared_memory_bytes = 32 * 1024;
        planning_profile.shared_memory_pow2_bytes = 32 * 1024;
        let dimensions = vec![6_144usize, 8usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let mut input = vec![Complex32::new(0.0, 0.0); tensor_len * batch_count];
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            input[base] = Complex32::new(1.0 + 0.125 * batch as f32, -0.0625);
            input[base + 1_234 * 8 + 3] = Complex32::new(-0.375, 0.1875 + 0.03125 * batch as f32);
        }
        let ir = TransformIr::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap(),
            Direction::Forward,
            planning_profile,
        )
        .unwrap();
        let TransformIr::ComplexNd(nd) = &ir else {
            panic!("CUDA grouped N6144x8 probe did not build ComplexNd");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let crate::OneDimFftIr::Recursive(recursive) = &outer.transform else {
            panic!("CUDA grouped N6144 higher axis should be recursive");
        };
        assert_eq!(
            recursive
                .stockham_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![96, 64]
        );
        assert!(
            recursive
                .four_step_plan
                .as_ref()
                .unwrap()
                .uploads
                .iter()
                .all(|upload| {
                    upload.axis_block.is_some_and(|block| {
                        block.grouped_batch == 3 && block.transforms_on_x && !block.axis_swapped
                    })
                })
        );

        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        let oracle_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&oracle_input).unwrap();
        let error = max_error32(&actual, &expected);
        assert!(
            error <= 5.0e-3,
            "CUDA grouped higher-axis two-upload error: {error}"
        );
    }

    #[test]
    fn cuda_grouped_higher_axis_rader_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for prime in [34usize, 47, 94, 257] {
            let dimensions = vec![prime, 8usize];
            let tensor_len = dimensions.iter().product::<usize>();
            let batch_count = 2usize;
            let mut input = vec![Complex32::new(0.0, 0.0); tensor_len * batch_count];
            for batch in 0..batch_count {
                let base = batch * tensor_len;
                input[base] = Complex32::new(1.0 + 0.125 * batch as f32, -0.0625);
                input[base + (prime / 2) * 8 + 3] =
                    Complex32::new(-0.375, 0.1875 + 0.03125 * batch as f32);
            }
            for direction in [Direction::Forward, Direction::Inverse] {
                let ir = TransformIr::build(
                    FftConfig::new(dimensions.clone())
                        .with_batch_count(batch_count)
                        .with_grouped_batch(0, 3)
                        .unwrap()
                        .with_grouped_batch(1, 3)
                        .unwrap()
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    context.device_profile(),
                )
                .unwrap();
                if prime == 94 {
                    let TransformIr::ComplexNd(nd) = &ir else {
                        panic!("CUDA grouped N94x8 should build ComplexNd");
                    };
                    let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                    let crate::OneDimFftIr::Recursive(recursive) = &outer.transform else {
                        panic!("CUDA grouped N94 higher axis should remain recursive");
                    };
                    let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) =
                        &recursive.root
                    else {
                        panic!("CUDA grouped N94 higher axis should keep a Cooley root");
                    };
                    let parent = root.pack_right.axis_batch_block.unwrap();
                    assert_eq!(parent.threads_per_transform, 48);
                    assert_eq!(parent.grouped_batch, 3);
                    assert_eq!([parent.local_size_x, parent.local_size_y], [3, 48]);
                    assert!(parent.transforms_on_x);
                    assert_eq!(root.twiddle_transpose.axis_batch_block, Some(parent));
                    assert_eq!(root.scatter_output.axis_batch_block, Some(parent));
                }
                let actual = context.execute_transform_complex32(&ir, &input).unwrap();
                let oracle_input = input
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected = ir.execute_complex_reference(&oracle_input).unwrap();
                let error = max_error32(&actual, &expected);
                assert!(
                    error <= 3.0e-3,
                    "CUDA grouped higher-axis p{prime} Rader error {direction:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn cuda_grouped_higher_axis_bluestein_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let dimensions = vec![103usize, 8usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let mut input = vec![Complex32::new(0.0, 0.0); tensor_len * batch_count];
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            input[base] = Complex32::new(1.0 + 0.125 * batch as f32, -0.0625);
            input[base + 51 * 8 + 3] = Complex32::new(-0.375, 0.1875 + 0.03125 * batch as f32);
        }
        for direction in [Direction::Forward, Direction::Inverse] {
            let ir = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap()
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_tuning(tuning),
                direction,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::ComplexNd(nd) = &ir else {
                panic!("CUDA grouped p103x8 Bluestein probe did not build ComplexNd");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let crate::OneDimFftIr::Bluestein(pipeline) = &outer.transform else {
                panic!("CUDA grouped p103 higher axis should remain Bluestein");
            };
            let block = pipeline.preprocess.axis_batch_block.unwrap();
            assert_eq!([block.local_size_x, block.local_size_y], [3, 128]);
            assert!(block.transforms_on_x);

            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected = ir.execute_complex_reference(&oracle_input).unwrap();
            let error = max_error32(&actual, &expected);
            assert!(
                error <= 5.0e-3,
                "CUDA grouped higher-axis Bluestein error {direction:?}: {error}"
            );
        }
    }

    #[test]
    fn cuda_higher_axis_bluestein_nd_r2r_auto_group_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut planning_profile = context.device_profile();
        if planning_profile.shared_memory_bytes < 48 * 1024
            || planning_profile.max_threads_per_block < 512
            || planning_profile.max_workgroup_size[0] < 4
            || planning_profile.max_workgroup_size[1] < 128
        {
            return;
        }
        planning_profile.shared_memory_bytes = 48 * 1024;
        planning_profile.shared_memory_pow2_bytes = 32 * 1024;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let dimensions = vec![103usize, 8usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let input = (0..tensor_len)
            .map(|index| {
                let x = index as f32;
                (0.021 * x).sin() + 0.15 * (0.013 * x).cos() + 0.0002 * x
            })
            .collect::<Vec<_>>();

        for direction in [Direction::Forward, Direction::Inverse] {
            let ir = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_tuning(tuning),
                direction,
                planning_profile,
            )
            .unwrap();
            let TransformIr::RealToRealNd(nd) = &ir else {
                panic!("CUDA p103x8 DCT-II probe did not build NdR2rIr");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let reduction = outer.transform.fft_reduction.as_deref().unwrap();
            let crate::OneDimFftIr::Bluestein(pipeline) = &reduction.fft else {
                panic!("CUDA p103 outer DCT-II reduction should use Bluestein");
            };
            let block = pipeline.preprocess.axis_batch_block.unwrap();
            assert!(block.grouped_batch > 1 && block.transforms_on_x);
            assert_eq!(block.threads_per_transform, 128);
            assert_eq!(pipeline.grouped_batch, 1);

            let actual = match context
                .execute_transform_f32(&ir, NativeTransformInput32::Real(&input))
                .unwrap()
            {
                NativeTransformOutput32::Real(values) => values,
                NativeTransformOutput32::Complex(_) => {
                    panic!("CUDA ND DCT-II returned complex output")
                }
            };
            let expected = ir
                .execute_r2r_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
                .unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual as f64 - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                error <= 2.0e-3 * dimensions[0] as f64,
                "CUDA auto-grouped higher-axis Bluestein ND DCT-II error {direction:?}: {error}"
            );
        }
    }

    #[test]
    fn cuda_multidimensional_zero_padding_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let dimensions = vec![3usize, 4usize];
        let batch_count = 2usize;
        let length = dimensions.iter().product::<usize>() * batch_count;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.069 * x).sin() + x * 0.0004, (0.039 * x).cos())
            })
            .collect::<Vec<_>>();
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for direction in [Direction::Forward, Direction::Inverse] {
                let config = FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_zero_padding(1, 2, 4)
                    .unwrap();
                let ir = TransformIr::build(config, direction, context.device_profile()).unwrap();
                let TransformIr::ComplexNd(nd) = &ir else {
                    panic!("CUDA ND zero-padded mixed-storage plan did not build ComplexNd");
                };
                assert!(nd.zero_pad_pass.is_some());
                let actual = context.execute_transform_complex32(&ir, &input).unwrap();
                let oracle_input = input
                    .iter()
                    .map(|value| {
                        if precision == Precision::F16StorageF32Compute {
                            Complex64::new(
                                crate::Binary16::from_f32(value.re).to_f32() as f64,
                                crate::Binary16::from_f32(value.im).to_f32() as f64,
                            )
                        } else {
                            Complex64::new(value.re as f64, value.im as f64)
                        }
                    })
                    .collect::<Vec<_>>();
                let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
                for value in &mut expected {
                    if precision == Precision::F16StorageF32Compute {
                        value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                        value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                    } else {
                        value.re = value.re as f32 as f64;
                        value.im = value.im as f32 as f64;
                    }
                }
                let error = max_error32(&actual, &expected);
                let tolerance = if precision == Precision::F16StorageF32Compute {
                    1.2e-3 * dimensions.iter().product::<usize>() as f64
                } else {
                    4.0e-5 * dimensions.iter().product::<usize>() as f64
                };
                assert!(
                    error <= tolerance,
                    "CUDA ND zero-padded mixed-storage error {direction:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn cuda_zero_padding_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            for (length, direction) in [
                (64usize, Direction::Forward),
                (64, Direction::Inverse),
                (49_152, Direction::Forward),
            ] {
                let left = length / 2;
                let config = FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_zero_padding(0, left, length)
                    .unwrap();
                let ir = TransformIr::build(config, direction, context.device_profile()).unwrap();
                if length == 49_152 {
                    let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir
                    else {
                        panic!("CUDA zero-padded multi-upload case did not remain recursive");
                    };
                    assert_eq!(
                        recursive
                            .stockham_upload_schedule
                            .as_ref()
                            .unwrap()
                            .upload_count,
                        2
                    );
                    assert!(recursive.four_step_plan.is_some());
                }
                let input = (0..length)
                    .map(|index| {
                        let x = index as f32;
                        Complex32::new(0.21 * (0.031 * x).sin(), 0.17 * (0.019 * x).cos())
                    })
                    .collect::<Vec<_>>();
                let actual = context.execute_transform_complex32(&ir, &input).unwrap();
                let oracle_input = input
                    .iter()
                    .map(|value| {
                        if precision == Precision::F16StorageF32Compute {
                            Complex64::new(
                                crate::Binary16::from_f32(value.re).to_f32() as f64,
                                crate::Binary16::from_f32(value.im).to_f32() as f64,
                            )
                        } else {
                            Complex64::new(value.re as f64, value.im as f64)
                        }
                    })
                    .collect::<Vec<_>>();
                let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
                for value in &mut expected {
                    if precision == Precision::F16StorageF32Compute {
                        value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                        value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                    } else {
                        value.re = value.re as f32 as f64;
                        value.im = value.im as f32 as f64;
                    }
                }
                let error = max_error32(&actual, &expected);
                let tolerance = if precision == Precision::F16StorageF32Compute {
                    1.0e-3 * length as f64
                } else {
                    3.0e-5 * length as f64
                };
                assert!(
                    error <= tolerance,
                    "CUDA zero-pad mixed-storage error N={length} {direction:?}: {error}"
                );
                if direction == Direction::Inverse {
                    assert!(
                        actual[left..]
                            .iter()
                            .all(|value| value.re == 0.0 && value.im == 0.0)
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_four_step_mixed_storage_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 49_152usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.037 * x).sin() + x * 0.00003, (0.017 * x).cos())
            })
            .collect::<Vec<_>>();
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            let ir = TransformIr::build(
                FftConfig::new(vec![length]).with_precision(precision),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("CUDA mixed-storage N=6144 did not build recursive C2C");
            };
            assert_eq!(
                recursive
                    .stockham_upload_schedule
                    .as_ref()
                    .expect("N=6144 should use the upload scheduler")
                    .upload_count,
                2
            );
            assert!(recursive.four_step_plan.is_some());

            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| {
                    if precision == Precision::F16StorageF32Compute {
                        Complex64::new(
                            crate::Binary16::from_f32(value.re).to_f32() as f64,
                            crate::Binary16::from_f32(value.im).to_f32() as f64,
                        )
                    } else {
                        Complex64::new(value.re as f64, value.im as f64)
                    }
                })
                .collect::<Vec<_>>();
            let mut expected = ir.execute_complex_reference(&oracle_input).unwrap();
            if precision == Precision::F16StorageF32Compute {
                for value in &mut expected {
                    value.re = crate::Binary16::from_f32(value.re as f32).to_f32() as f64;
                    value.im = crate::Binary16::from_f32(value.im as f32).to_f32() as f64;
                }
            }
            let error = max_error32(&actual, &expected);
            let tolerance = if precision == Precision::F16StorageF32Compute {
                1.0e-3 * length as f64
            } else {
                3.0e-5 * length as f64
            };
            assert!(
                error <= tolerance,
                "CUDA Four-step mixed-storage error {error}"
            );
        }
    }

    #[test]
    fn cuda_recursive_mixed_storage_cooley_matches_quantized_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 256usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.053 * x).sin() + x * 0.00017, (0.021 * x).cos())
            })
            .collect::<Vec<_>>();
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            if precision == Precision::F64ComputeF32Storage
                && !context.device_profile().supports_f64
            {
                continue;
            }
            let mut profile = context.device_profile();
            profile.vendor = crate::GpuVendor::Other(0xD00D);
            profile.shared_memory_bytes = 64;
            profile.shared_memory_pow2_bytes = 64;
            let ir = TransformIr::build(
                FftConfig::new(vec![length]).with_precision(precision),
                Direction::Forward,
                profile,
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("forced mixed-storage CUDA plan did not build recursive C2C");
            };
            assert!(matches!(
                recursive.root,
                crate::RecursiveFftNodeIr::CooleyTukey(_)
            ));
            let actual = context.execute_transform_complex32(&ir, &input).unwrap();
            let oracle_input = input
                .iter()
                .map(|value| match precision {
                    Precision::F16StorageF32Compute => Complex64::new(
                        crate::Binary16::from_f32(value.re).to_f32() as f64,
                        crate::Binary16::from_f32(value.im).to_f32() as f64,
                    ),
                    Precision::F64ComputeF32Storage => {
                        Complex64::new(value.re as f64, value.im as f64)
                    }
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            let expected = ir
                .execute_complex_reference(&oracle_input)
                .unwrap()
                .into_iter()
                .map(|value| match precision {
                    Precision::F16StorageF32Compute => Complex32::new(
                        crate::Binary16::from_f32(value.re as f32).to_f32(),
                        crate::Binary16::from_f32(value.im as f32).to_f32(),
                    ),
                    Precision::F64ComputeF32Storage => {
                        Complex32::new(value.re as f32, value.im as f32)
                    }
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0f32, f32::max);
            let tolerance = match precision {
                Precision::F16StorageF32Compute => 1.0e-3 * length as f32,
                Precision::F64ComputeF32Storage => 2.0e-5 * length as f32,
                _ => unreachable!(),
            };
            assert!(
                error <= tolerance,
                "CUDA recursive {precision:?} error {error} > {tolerance}"
            );
        }
    }

    #[test]
    fn cuda_real_gpu_f64_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 64usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.09 * x).sin() + x * 0.0003, (0.03 * x).cos())
            })
            .collect::<Vec<_>>();
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(Precision::F64),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let actual = context.execute_transform_complex64(&ir, &input).unwrap();
        let expected = ir.execute_complex_reference(&input).unwrap();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(error <= 2.0e-10 * length as f64, "CUDA F64 error {error}");

        let real_len = 16usize;
        let real_input = (0..real_len)
            .map(|index| {
                let x = index as f64;
                (0.051 * x).sin() + 0.13 * (0.017 * x).cos()
            })
            .collect::<Vec<_>>();
        let real = TransformIr::build(
            FftConfig::new(vec![real_len])
                .with_precision(Precision::F64)
                .with_transform(crate::TransformKind::RealToComplex),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let spectrum = match context
            .execute_transform_f64(&real, NativeTransformInput64::Real(&real_input))
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => panic!("CUDA F64 R2C returned real output"),
        };
        let expected = real.execute_r2c_reference(&real_input).unwrap();
        let error = spectrum
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            error <= 4.0e-10 * real_len as f64,
            "CUDA F64 R2C error {error}"
        );
    }
    #[test]
    fn double_double_r2r_all_fft_families_match_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 9usize;
        let batch_count = 7usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.13 * x).sin() + 0.17 * (0.07 * x).cos() + 0.001 * x,
                    (index + 1) as f64 * 5.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_grouped_batch(0, 3)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let inverse = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_inverse_normalization(true)
                .with_grouped_batch(0, 3)
                .unwrap(),
            Direction::Inverse,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::RealToRealDoubleDouble(forward_ir) = &forward else {
            panic!("DD DCT-II must build DoubleDoubleR2rIr");
        };
        let TransformIr::RealToRealDoubleDouble(inverse_ir) = &inverse else {
            panic!("DD DCT-II inverse must build DoubleDoubleR2rIr");
        };
        assert!(matches!(
            forward_ir.algorithm,
            crate::DoubleDoubleR2rAlgorithm::FftReduction { .. }
        ));
        assert!(matches!(
            inverse_ir.algorithm,
            crate::DoubleDoubleR2rAlgorithm::FftReduction { .. }
        ));
        assert_eq!(
            crate::ProgramIr::double_double_r2r(forward_ir)
                .unwrap()
                .passes
                .len(),
            1
        );
        assert_eq!(
            crate::ProgramIr::double_double_r2r(inverse_ir)
                .unwrap()
                .passes
                .len(),
            1
        );
        let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
        let actual = context
            .execute_double_double_r2r(forward_ir, &input)
            .unwrap();
        let dd_error = |actual: DoubleDouble, expected: DoubleDouble| {
            let delta = (actual - expected).abs();
            delta.hi.abs() + delta.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-19,
            "CUDA DD DCT-II mismatch on {}: {forward_error:e}",
            context.device_name()
        );
        let restored = context
            .execute_double_double_r2r(inverse_ir, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 5.0e-17,
            "CUDA DD DCT-II/III round trip mismatch on {}: {round_trip_error:e}",
            context.device_name()
        );

        for family_transform in [
            crate::TransformKind::Dct(crate::DctType::I),
            crate::TransformKind::Dct(crate::DctType::IV),
            crate::TransformKind::Dst(crate::DstType::I),
            crate::TransformKind::Dst(crate::DstType::II),
            crate::TransformKind::Dst(crate::DstType::III),
            crate::TransformKind::Dst(crate::DstType::IV),
        ] {
            let family = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(family_transform)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::RealToRealDoubleDouble(family_ir) = &family else {
                panic!("DD DCT/DST must build DoubleDoubleR2rIr");
            };
            assert!(matches!(
                family_ir.algorithm,
                crate::DoubleDoubleR2rAlgorithm::FftReduction { .. }
            ));
            let family_expected = family.execute_double_double_r2r_reference(&input).unwrap();
            let family_actual = context
                .execute_transform_double_double_r2r(&family, &input)
                .unwrap();
            let family_error = family_actual
                .iter()
                .copied()
                .zip(family_expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                family_error <= 5.0e-19,
                "CUDA DD {family_transform:?} mismatch on {}: {family_error:e}",
                context.device_name()
            );
            let family_inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(family_transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let family_restored = context
                .execute_transform_double_double_r2r(&family_inverse, &family_actual)
                .unwrap();
            let family_round_trip_error = family_restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                family_round_trip_error <= 5.0e-17,
                "DD {family_transform:?} round trip mismatch on {}: {family_round_trip_error:e}",
                context.device_name()
            );
        }

        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        for f64_transform in [
            crate::TransformKind::Dct(crate::DctType::I),
            crate::TransformKind::Dst(crate::DstType::I),
            crate::TransformKind::Dst(crate::DstType::II),
            crate::TransformKind::Dct(crate::DctType::II),
            crate::TransformKind::Dct(crate::DctType::IV),
            crate::TransformKind::Dst(crate::DstType::IV),
        ] {
            let f64_forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(f64_transform)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(f64_transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            assert!(matches!(
                f64_forward,
                TransformIr::RealToRealDoubleDouble(_)
            ));
            assert!(matches!(
                f64_inverse,
                TransformIr::RealToRealDoubleDouble(_)
            ));
            let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_forward_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_forward_error <= 3.0e-12,
                "CUDA DD/F64 {f64_transform:?} mismatch on {}: {f64_forward_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_round_trip_error <= 3.0e-13,
                "CUDA DD/F64 {f64_transform:?} round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn cuda_double_double_dct2_n512_wide_stockham_fusion_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 512usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.017 * x).sin() + 0.13 * (0.031 * x).cos() + 0.00011 * x,
                    (index + 1) as f64 * 3.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let build = |precision, direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Precision::DoubleDouble, Direction::Forward);
        let inverse = build(Precision::DoubleDouble, Direction::Inverse);
        for transform in [&forward, &inverse] {
            let TransformIr::RealToRealDoubleDouble(ir) = transform else {
                panic!("CUDA DD DCT-II N512 must build DoubleDoubleR2rIr");
            };
            let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &ir.algorithm else {
                panic!("CUDA DD DCT-II N512 must use FFT reduction");
            };
            assert!(matches!(
                fft.as_ref(),
                crate::DoubleDoubleOneDimIr::Stockham(_)
            ));
            assert_eq!(
                crate::ProgramIr::double_double_r2r(ir)
                    .unwrap()
                    .passes
                    .len(),
                1
            );
        }
        let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double_r2r(&forward, &input)
            .unwrap();
        let dd_error = |actual: DoubleDouble, expected: DoubleDouble| {
            let delta = (actual - expected).abs();
            delta.hi.abs() + delta.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-16,
            "CUDA DD DCT-II N512 error {forward_error:e}"
        );
        let restored = context
            .execute_transform_double_double_r2r(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-14,
            "CUDA DD DCT-II/III N512 round-trip error {round_trip_error:e}"
        );

        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_forward = build(Precision::DoubleDoubleF64Storage, Direction::Forward);
        let f64_inverse = build(Precision::DoubleDoubleF64Storage, Direction::Inverse);
        for transform in [&f64_forward, &f64_inverse] {
            let TransformIr::RealToRealDoubleDouble(ir) = transform else {
                panic!("CUDA DD/F64 DCT-II N512 must build DoubleDoubleR2rIr");
            };
            assert_eq!(
                crate::ProgramIr::double_double_r2r(ir)
                    .unwrap()
                    .passes
                    .len(),
                1
            );
        }
        let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 2.0e-10,
            "CUDA DD/F64 DCT-II N512 error {f64_forward_error:e}"
        );
        let f64_restored = context
            .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_round_trip_error <= 2.0e-11,
            "CUDA DD/F64 DCT-II/III N512 round-trip error {f64_round_trip_error:e}"
        );
    }

    #[test]
    fn double_double_even_type_iv_half_size_round_trip_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 10usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.17 * x).sin() + 0.09 * (0.23 * x).cos() - 0.003 * x,
                    (index + 1) as f64 * 9.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let dd_error = |actual: DoubleDouble, expected: DoubleDouble| {
            let delta = (actual - expected).abs();
            delta.hi.abs() + delta.lo.abs()
        };

        for transform in [
            crate::TransformKind::Dct(crate::DctType::IV),
            crate::TransformKind::Dst(crate::DstType::IV),
        ] {
            let forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            for ir in [&forward, &inverse] {
                let TransformIr::RealToRealDoubleDouble(ir) = ir else {
                    panic!("CUDA DD even Type-IV must build DoubleDoubleR2rIr");
                };
                assert!(matches!(
                    &ir.algorithm,
                    crate::DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize { fft_len, .. }
                        if *fft_len == length / 2
                ));
            }
            let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double_r2r(&forward, &input)
                .unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 5.0e-18,
                "CUDA DD even Type-IV {transform:?} mismatch on {}: {forward_error:e}",
                context.device_name()
            );
            let padded_forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_zero_padding(0, 1, 4)
                    .unwrap()
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let padded_expected = padded_forward
                .execute_double_double_r2r_reference(&input)
                .unwrap();
            let padded_actual = context
                .execute_transform_double_double_r2r(&padded_forward, &input)
                .unwrap();
            let padded_error = padded_actual
                .iter()
                .copied()
                .zip(padded_expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                padded_error <= 5.0e-18,
                "CUDA DD padded even Type-IV {transform:?} mismatch on {}: {padded_error:e}",
                context.device_name()
            );
            let restored = context
                .execute_transform_double_double_r2r(&inverse, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 5.0e-16,
                "CUDA DD even Type-IV {transform:?} round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );

            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(transform)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            for ir in [&f64_forward, &f64_inverse] {
                let TransformIr::RealToRealDoubleDouble(ir) = ir else {
                    panic!("CUDA DD/F64 even Type-IV must build DoubleDoubleR2rIr");
                };
                assert!(matches!(
                    &ir.algorithm,
                    crate::DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize { fft_len, .. }
                        if *fft_len == length / 2
                ));
            }
            let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_forward_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_forward_error <= 5.0e-12,
                "CUDA DD/F64 even Type-IV {transform:?} mismatch on {}: {f64_forward_error:e}",
                context.device_name()
            );
            let f64_padded_forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(transform)
                    .with_zero_padding(0, 1, 4)
                    .unwrap()
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_padded_expected = f64_padded_forward
                .execute_r2r_reference(&f64_input)
                .unwrap();
            let f64_padded_actual = context
                .execute_transform_double_double_r2r_f64_storage(&f64_padded_forward, &f64_input)
                .unwrap();
            let f64_padded_error = f64_padded_actual
                .iter()
                .zip(&f64_padded_expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_padded_error <= 5.0e-12,
                "CUDA DD/F64 padded even Type-IV {transform:?} mismatch on {}: {f64_padded_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_round_trip_error <= 5.0e-13,
                "CUDA DD/F64 even Type-IV {transform:?} round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn double_double_nd_r2r_fft_families_match_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![3usize, 4usize];
        let batch_count = 7usize;
        let elements = dimensions.iter().product::<usize>() * batch_count;
        let input = (0..elements)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.11 * x).sin() + 0.19 * (0.05 * x).cos() + 0.002 * x,
                    (index + 1) as f64 * 7.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let dd_error = |actual: DoubleDouble, expected: DoubleDouble| {
            let delta = (actual - expected).abs();
            delta.hi.abs() + delta.lo.abs()
        };

        for transform in [
            crate::TransformKind::Dct(crate::DctType::I),
            crate::TransformKind::Dst(crate::DstType::I),
            crate::TransformKind::Dct(crate::DctType::II),
            crate::TransformKind::Dst(crate::DstType::III),
            crate::TransformKind::Dct(crate::DctType::IV),
            crate::TransformKind::Dst(crate::DstType::IV),
        ] {
            let forward = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let inverse = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let TransformIr::RealToRealNdDoubleDouble(forward_ir) = &forward else {
                panic!("DD ND R2R must build DoubleDoubleNdR2rIr");
            };
            for axis in &forward_ir.axes {
                let even_type_iv = matches!(
                    transform,
                    crate::TransformKind::Dct(crate::DctType::IV)
                        | crate::TransformKind::Dst(crate::DstType::IV)
                ) && axis.axis_len.is_multiple_of(2);
                if even_type_iv {
                    assert!(matches!(
                        &axis.transform.algorithm,
                        crate::DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize { fft_len, .. }
                            if *fft_len == axis.axis_len / 2
                    ));
                } else {
                    assert!(matches!(
                        axis.transform.algorithm,
                        crate::DoubleDoubleR2rAlgorithm::FftReduction { .. }
                    ));
                }
            }
            let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double_r2r(&forward, &input)
                .unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 2.0e-17,
                "CUDA DD ND {transform:?} mismatch on {}: {forward_error:e}",
                context.device_name()
            );
            let restored = context
                .execute_transform_double_double_r2r(&inverse, &actual)
                .unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error <= 2.0e-16,
                "CUDA DD ND {transform:?} round trip mismatch on {}: {round_trip_error:e}",
                context.device_name()
            );

            let f64_forward = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(transform)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_inverse = TransformIr::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_transform(transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_forward_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_forward_error <= 2.0e-11,
                "CUDA DD/F64 ND {transform:?} mismatch on {}: {f64_forward_error:e}",
                context.device_name()
            );
            let f64_restored = context
                .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_round_trip_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_round_trip_error <= 3.0e-12,
                "CUDA DD/F64 ND {transform:?} round trip mismatch on {}: {f64_round_trip_error:e}",
                context.device_name()
            );
        }
    }

    #[test]
    fn ordinary_r2r_padding_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 12usize;
        let batch_count = 7usize;
        let left = 3usize;
        let right = 6usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                (0.11 * x).sin() + 0.19 * (0.047 * x).cos() + 0.0007 * x
            })
            .collect::<Vec<_>>();
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_zero_padding(0, left, right)
                    .unwrap(),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };

        let forward = build(Direction::Forward);
        let expected = forward
            .execute_r2r_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
            .unwrap();
        let actual = match context
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA padded DCT returned complex output")
            }
        };
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 3.0e-3 * length as f64,
            "CUDA padded DCT-II forward mismatch: {forward_error}"
        );

        let inverse = build(Direction::Inverse);
        let actual64 = actual.iter().map(|value| *value as f64).collect::<Vec<_>>();
        let inverse_expected = inverse.execute_r2r_reference(&actual64).unwrap();
        let restored = match context
            .execute_transform_f32(&inverse, NativeTransformInput32::Real(&actual))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("CUDA padded inverse DCT returned complex output")
            }
        };
        let inverse_error = restored
            .iter()
            .zip(&inverse_expected)
            .map(|(actual, expected)| (*actual as f64 - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 3.0e-3 * length as f64,
            "CUDA padded inverse DCT-II mismatch: {inverse_error}"
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + left..base + right]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }

    #[test]
    fn odd_iv_r2r_padding_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 9usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let left = 2usize;
        let right = 5usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f32;
                (0.13 * x).sin() + 0.17 * (0.041 * x).cos() + 0.0005 * x
            })
            .collect::<Vec<_>>();
        for transform in [
            crate::TransformKind::Dct(crate::DctType::IV),
            crate::TransformKind::Dst(crate::DstType::IV),
        ] {
            let build = |direction| {
                TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_transform(transform)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_zero_padding(0, left, right)
                        .unwrap(),
                    direction,
                    context.device_profile(),
                )
                .unwrap()
            };
            let forward = build(Direction::Forward);
            let TransformIr::RealToReal(forward_r2r) = &forward else {
                panic!("CUDA padded odd-IV transform should build ordinary R2R IR");
            };
            assert_eq!(
                forward_r2r.fft_reduction.as_deref().unwrap().fft_len,
                2 * length
            );
            let expected = forward
                .execute_r2r_reference(&input.iter().map(|value| *value as f64).collect::<Vec<_>>())
                .unwrap();
            let actual = match context
                .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
                .unwrap()
            {
                NativeTransformOutput32::Real(values) => values,
                NativeTransformOutput32::Complex(_) => {
                    panic!("CUDA padded odd-IV transform returned complex output")
                }
            };
            let forward_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual as f64 - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 3.0e-3 * length as f64,
                "CUDA padded {transform:?} forward mismatch: {forward_error}"
            );

            let inverse = build(Direction::Inverse);
            let actual64 = actual.iter().map(|value| *value as f64).collect::<Vec<_>>();
            let inverse_expected = inverse.execute_r2r_reference(&actual64).unwrap();
            let restored = match context
                .execute_transform_f32(&inverse, NativeTransformInput32::Real(&actual))
                .unwrap()
            {
                NativeTransformOutput32::Real(values) => values,
                NativeTransformOutput32::Complex(_) => {
                    panic!("CUDA padded inverse odd-IV transform returned complex output")
                }
            };
            let inverse_error = restored
                .iter()
                .zip(&inverse_expected)
                .map(|(actual, expected)| (*actual as f64 - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                inverse_error <= 3.0e-3 * length as f64,
                "CUDA padded inverse {transform:?} mismatch: {inverse_error}"
            );
            for batch in 0..batch_count {
                let base = batch * length;
                assert!(
                    restored[base + left..base + right]
                        .iter()
                        .all(|value| *value == 0.0)
                );
            }
        }
    }

    #[test]
    fn one_dim_dd_c2c_padding_stockham_direct_fft_rader_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        #[derive(Clone, Copy)]
        enum ExpectedKind {
            Stockham,
            DirectRader,
            FftRader,
        }
        let cases = [
            (16usize, ExpectedKind::Stockham),
            (47usize, ExpectedKind::DirectRader),
            (83usize, ExpectedKind::DirectRader),
            (257usize, ExpectedKind::FftRader),
        ];
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let dd_error = |actual: ComplexDoubleDouble, expected: ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        for (length, expected_kind) in cases {
            let left = length / 4;
            let right = length / 2;
            let config = |precision, direction| {
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_tuning(crate::PlannerTuning::portable())
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_zero_padding(0, left, right)
                    .unwrap()
            };
            let forward = TransformIr::build(
                config(Precision::DoubleDouble, Direction::Forward),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let inverse = TransformIr::build(
                config(Precision::DoubleDouble, Direction::Inverse),
                Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            for transform in [&forward, &inverse] {
                let TransformIr::Complex1dDoubleDouble(inner) = transform else {
                    panic!("CUDA N={length} padded DD C2C did not use 1D DD IR");
                };
                match (inner, expected_kind) {
                    (crate::DoubleDoubleOneDimIr::Stockham(ir), ExpectedKind::Stockham) => {
                        assert!(ir.zero_pad_pass.is_some());
                    }
                    (crate::DoubleDoubleOneDimIr::DirectRader(ir), ExpectedKind::DirectRader) => {
                        assert!(ir.zero_pad_pass.is_some());
                    }
                    (crate::DoubleDoubleOneDimIr::FftRader(ir), ExpectedKind::FftRader) => {
                        assert!(ir.zero_pad_pass.is_some());
                        assert!(ir.forward_fft.zero_pad_pass().is_none());
                        assert!(ir.inverse_fft.zero_pad_pass().is_none());
                    }
                    _ => panic!("CUDA N={length} padded DD C2C selected unexpected algorithm"),
                }
            }
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    ComplexDoubleDouble::new(
                        DoubleDouble::from_parts(
                            (0.071 * x).sin() + 0.0007 * x,
                            (index + 1) as f64 * 8.0e-32,
                        ),
                        DoubleDouble::from_parts(
                            (0.039 * x).cos() - 0.0003 * x,
                            -(index as f64 + 1.0) * 5.0e-32,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let expected = forward.execute_double_double_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double(&forward, &input)
                .unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 5.0e-16,
                "CUDA N={length} padded DD C2C forward mismatch: {forward_error:e}"
            );
            let expected_inverse = inverse.execute_double_double_reference(&actual).unwrap();
            let restored = context
                .execute_transform_double_double(&inverse, &actual)
                .unwrap();
            let inverse_error = restored
                .iter()
                .copied()
                .zip(expected_inverse.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                inverse_error <= 5.0e-15,
                "CUDA N={length} padded DD C2C inverse mismatch: {inverse_error:e}"
            );
            for batch in 0..batch_count {
                let base = batch * length;
                assert!(
                    restored[base + left..base + right]
                        .iter()
                        .all(|value| *value == ComplexDoubleDouble::default())
                );
            }

            let f64_ir = TransformIr::build(
                config(Precision::DoubleDoubleF64Storage, Direction::Forward),
                Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(ComplexDoubleDouble::to_complex64)
                .collect::<Vec<_>>();
            let f64_expected = f64_ir.execute_complex_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_f64_storage(&f64_ir, &f64_input)
                .unwrap();
            let f64_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                f64_error <= 5.0e-11 * length as f64,
                "CUDA N={length} padded DD/F64 C2C mismatch: {f64_error:e}"
            );
        }
    }

    #[test]
    fn double_double_recursive_padding_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 17usize * 17;
        let batch_count = 3usize;
        let left = 97usize;
        let right = 151usize;
        let build = |direction| {
            TransformIr::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_grouped_batch(0, 2)
                    .unwrap()
                    .with_zero_padding(0, left, right)
                    .unwrap(),
                direction,
                context.device_profile(),
            )
            .unwrap()
        };
        let forward = build(Direction::Forward);
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(recursive)) =
            &forward
        else {
            panic!("CUDA padded p17^2 should route through DD recursive IR");
        };
        assert!(recursive.zero_pad_pass.is_some());
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.071 * x).sin() + 0.002 * x,
                        (index + 1) as f64 * 8.0e-32,
                    ),
                    DoubleDouble::from_parts(
                        (0.037 * x).cos() - 0.001 * x,
                        -(index as f64 + 1.0) * 5.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = forward.execute_double_double_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double(&forward, &input)
            .unwrap();
        let dd_error = |actual: ComplexDoubleDouble, expected: ComplexDoubleDouble| {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
        };
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 5.0e-17,
            "CUDA padded DD recursive forward mismatch: {forward_error:e}"
        );

        let inverse = build(Direction::Inverse);
        let expected_inverse = inverse.execute_double_double_reference(&actual).unwrap();
        let restored = context
            .execute_transform_double_double(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_inverse.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 5.0e-16,
            "CUDA padded DD recursive inverse mismatch: {inverse_error:e}"
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + left..base + right]
                    .iter()
                    .all(|value| *value == ComplexDoubleDouble::default())
            );
        }

        let f64_forward = TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_grouped_batch(0, 2)
                .unwrap()
                .with_zero_padding(0, left, right)
                .unwrap(),
            Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(
            f64_recursive,
        )) = &f64_forward
        else {
            panic!("CUDA padded DD/F64 p17^2 should route through recursive IR");
        };
        assert!(f64_recursive.zero_pad_pass.is_some());
        let f64_input = input
            .iter()
            .copied()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_complex_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 5.0e-11 * length as f64,
            "CUDA padded DD/F64 recursive forward mismatch: {f64_error:e}"
        );
    }

    #[test]
    fn double_double_r2r_padding_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 9usize;
        let batch_count = 7usize;
        let left = 3usize;
        let right = 6usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.097 * x).sin() + 0.17 * (0.043 * x).cos() + 0.0013 * x,
                    (index + 1) as f64 * 6.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let dd_error = |actual: DoubleDouble, expected: DoubleDouble| {
            let delta = (actual - expected).abs();
            delta.hi.abs() + delta.lo.abs()
        };
        let build = |precision, direction| {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, left, right)
                .unwrap();
            TransformIr::build(config, direction, context.device_profile()).unwrap()
        };

        let forward = build(Precision::DoubleDouble, Direction::Forward);
        let inverse = build(Precision::DoubleDouble, Direction::Inverse);
        let TransformIr::RealToRealDoubleDouble(forward_ir) = &forward else {
            panic!("CUDA padded DD DCT-II must build DoubleDoubleR2rIr");
        };
        let TransformIr::RealToRealDoubleDouble(inverse_ir) = &inverse else {
            panic!("CUDA padded DD DCT-II inverse must build DoubleDoubleR2rIr");
        };
        assert_eq!(
            crate::ProgramIr::double_double_r2r(forward_ir)
                .unwrap()
                .passes
                .len(),
            1
        );
        assert_eq!(
            crate::ProgramIr::double_double_r2r(inverse_ir)
                .unwrap()
                .passes
                .len(),
            1
        );
        let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
        let actual = context
            .execute_transform_double_double_r2r(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-17,
            "CUDA padded DD R2R forward mismatch: {forward_error:e}"
        );
        let expected_restored = inverse
            .execute_double_double_r2r_reference(&actual)
            .unwrap();
        let restored = context
            .execute_transform_double_double_r2r(&inverse, &actual)
            .unwrap();
        let inverse_error = restored
            .iter()
            .copied()
            .zip(expected_restored.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 2.0e-16,
            "CUDA padded DD R2R inverse mismatch: {inverse_error:e}"
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                restored[base + left..base + right]
                    .iter()
                    .all(|value| *value == DoubleDouble::ZERO)
            );
        }

        let f64_forward = build(Precision::DoubleDoubleF64Storage, Direction::Forward);
        let f64_inverse = build(Precision::DoubleDoubleF64Storage, Direction::Inverse);
        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
        let f64_actual = context
            .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
            .unwrap();
        let f64_forward_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_forward_error <= 2.0e-11,
            "CUDA padded DD/F64 R2R forward mismatch: {f64_forward_error:e}"
        );
        let f64_expected_restored = f64_inverse.execute_r2r_reference(&f64_actual).unwrap();
        let f64_restored = context
            .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
            .unwrap();
        let f64_inverse_error = f64_restored
            .iter()
            .zip(&f64_expected_restored)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_inverse_error <= 3.0e-12,
            "CUDA padded DD/F64 R2R inverse mismatch: {f64_inverse_error:e}"
        );
        for batch in 0..batch_count {
            let base = batch * length;
            assert!(
                f64_restored[base + left..base + right]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }

    #[test]
    fn double_double_nd_r2r_padding_matches_cpu_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        if !context.device_profile().supports_f64 {
            return;
        }
        let dimensions = vec![3usize, 4usize];
        let batch_count = 7usize;
        let elements = dimensions.iter().product::<usize>() * batch_count;
        let input = (0..elements)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.097 * x).sin() + 0.17 * (0.043 * x).cos() + 0.0013 * x,
                    (index + 1) as f64 * 6.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let dd_error = |actual: DoubleDouble, expected: DoubleDouble| {
            let delta = (actual - expected).abs();
            delta.hi.abs() + delta.lo.abs()
        };
        for transform in [
            crate::TransformKind::Dct(crate::DctType::II),
            crate::TransformKind::Dst(crate::DstType::III),
        ] {
            let build = |precision, direction| {
                let config = FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_transform(transform)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap()
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_zero_padding(1, 1, 3)
                    .unwrap();
                TransformIr::build(config, direction, context.device_profile()).unwrap()
            };
            let forward = build(Precision::DoubleDouble, Direction::Forward);
            let inverse = build(Precision::DoubleDouble, Direction::Inverse);
            let expected = forward.execute_double_double_r2r_reference(&input).unwrap();
            let actual = context
                .execute_transform_double_double_r2r(&forward, &input)
                .unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error <= 2.0e-17,
                "CUDA padded DD ND {transform:?} forward mismatch: {forward_error:e}"
            );
            let expected_restored = inverse
                .execute_double_double_r2r_reference(&expected)
                .unwrap();
            let restored = context
                .execute_transform_double_double_r2r(&inverse, &actual)
                .unwrap();
            let inverse_error = restored
                .iter()
                .copied()
                .zip(expected_restored.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                inverse_error <= 2.0e-16,
                "CUDA padded DD ND {transform:?} inverse mismatch: {inverse_error:e}"
            );

            let f64_forward = build(Precision::DoubleDoubleF64Storage, Direction::Forward);
            let f64_inverse = build(Precision::DoubleDoubleF64Storage, Direction::Inverse);
            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
            let f64_actual = context
                .execute_transform_double_double_r2r_f64_storage(&f64_forward, &f64_input)
                .unwrap();
            let f64_forward_error = f64_actual
                .iter()
                .zip(&f64_expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_forward_error <= 2.0e-11,
                "CUDA padded DD/F64 ND {transform:?} forward mismatch: {f64_forward_error:e}"
            );
            let f64_expected_restored = f64_inverse.execute_r2r_reference(&f64_expected).unwrap();
            let f64_restored = context
                .execute_transform_double_double_r2r_f64_storage(&f64_inverse, &f64_actual)
                .unwrap();
            let f64_inverse_error = f64_restored
                .iter()
                .zip(&f64_expected_restored)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_inverse_error <= 3.0e-12,
                "CUDA padded DD/F64 ND {transform:?} inverse mismatch: {f64_inverse_error:e}"
            );
        }
    }

    #[test]
    fn rader_internal_bluestein_child_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let build = |direction| {
            let config = FftConfig::new(vec![103])
                .with_tuning(tuning)
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = crate::FftPlan::build(config).unwrap();
            let one_dim =
                crate::OneDimFftIr::build(&plan, direction, context.device_profile()).unwrap();
            crate::TransformIr::Complex1d(one_dim)
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        let crate::TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward
        else {
            panic!("CUDA custom p103 should keep an outer recursive FFT-Rader node");
        };
        let crate::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
            panic!("CUDA custom p103 should keep an FFT-Rader root");
        };
        assert!(matches!(
            rader.forward_fft.as_ref(),
            crate::OneDimFftIr::Bluestein(_)
        ));

        let input = (0..103)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.071 * x).sin() + x * 0.0002,
                    (0.043 * x).cos() - x * 0.0001,
                )
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = forward.execute_complex_reference(&expected_input).unwrap();
        let actual = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re;
                let di = actual.im as f64 - expected.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-3 * 103.0,
            "CUDA nested Bluestein Rader error {forward_error}"
        );

        let restored = context
            .execute_transform_complex32(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re as f64;
                let di = actual.im as f64 - expected.im as f64;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 3.0e-3,
            "CUDA nested Bluestein Rader round-trip error {round_trip_error}"
        );
    }

    #[test]
    fn composite_rader_parent_with_bluestein_child_matches_reference_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let length = 2usize * 103;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let build = |direction| {
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            crate::TransformIr::Complex1d(
                crate::OneDimFftIr::build(&plan, direction, context.device_profile()).unwrap(),
            )
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        let crate::TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward
        else {
            panic!("CUDA N206 should keep the recursive 2 x p103 parent");
        };
        let crate::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
            panic!("CUDA N206 should use a Cooley parent");
        };
        assert_eq!(
            root.pack_right
                .axis_batch_block
                .unwrap()
                .threads_per_transform,
            103
        );
        let rader = match (&root.left, &root.right) {
            (crate::RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 103 => rader,
            (_, crate::RecursiveFftNodeIr::FftRader(rader)) if rader.prime == 103 => rader,
            _ => panic!("CUDA N206 should retain a p103 FFT-Rader child"),
        };
        assert!(matches!(
            rader.forward_fft.as_ref(),
            crate::OneDimFftIr::Bluestein(_)
        ));

        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.031 * x).sin() + x * 0.0002,
                    (0.047 * x).cos() - x * 0.0001,
                )
            })
            .collect::<Vec<_>>();
        let expected_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = forward.execute_complex_reference(&expected_input).unwrap();
        let actual = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re;
                let di = actual.im as f64 - expected.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(forward_error <= 2.0e-3 * length as f64);
        let restored = context
            .execute_transform_complex32(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re as f64;
                let di = actual.im as f64 - expected.im as f64;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(round_trip_error <= 4.0e-3);
    }

    #[test]
    fn cuda_p103_bluestein_four_step_component_or_skip() {
        let Some(context) = context_or_skip() else {
            return;
        };
        let actual_profile = context.device_profile();
        if actual_profile.max_threads_per_block < 560
            || actual_profile.max_workgroup_size[0] < 16
            || actual_profile.max_workgroup_size[1] < 35
        {
            return;
        }
        let length = 64usize * 103;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let mut planner_profile = actual_profile;
        planner_profile.shared_memory_bytes = 32 * 1024;
        planner_profile.shared_memory_pow2_bytes = 32 * 1024;
        planner_profile.max_threads_per_block = planner_profile.max_threads_per_block.min(1024);
        planner_profile.max_workgroup_size[0] = planner_profile.max_workgroup_size[0].min(1024);
        planner_profile.max_workgroup_size[1] = planner_profile.max_workgroup_size[1].min(1024);
        planner_profile.coalesced_memory_bytes = 32;
        let build = |direction| {
            let plan = crate::FftPlan::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            crate::TransformIr::Complex1d(
                crate::OneDimFftIr::build(&plan, direction, planner_profile).unwrap(),
            )
        };
        let forward = build(Direction::Forward);
        let inverse = build(Direction::Inverse);
        let crate::TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &forward
        else {
            panic!("CUDA N6592 should keep the two-upload recursive Rader path");
        };
        assert_eq!(
            recursive
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![64, 103]
        );
        let high = recursive
            .four_step_plan
            .as_ref()
            .unwrap()
            .uploads
            .iter()
            .find(|upload| upload.axis_upload_id == 1)
            .unwrap();
        let block = high.axis_block.unwrap();
        assert_eq!(block.threads_per_transform, 35);
        assert_eq!(block.grouped_batch, 16);
        assert_eq!([block.local_size_x, block.local_size_y], [16, 35]);
        let uploads = recursive.four_step_rader_upload_nodes().unwrap().unwrap();
        let crate::RecursiveFftNodeIr::FftRader(rader) = &uploads[0] else {
            panic!("CUDA N6592 upload1 should remain p103 FFT-Rader");
        };
        assert_eq!(rader.caller_axis_batch_block, Some(block));
        assert!(matches!(
            rader.forward_fft.as_ref(),
            crate::OneDimFftIr::Bluestein(_)
        ));

        let mut input = vec![Complex32::new(0.0, 0.0); length];
        input[1] = Complex32::new(1.0, 0.0);
        let actual = context
            .execute_transform_complex32(&forward, &input)
            .unwrap();
        let forward_error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = Complex64::new(angle.cos(), angle.sin());
                let dr = value.re as f64 - expected.re;
                let di = value.im as f64 - expected.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 1.0e-2,
            "CUDA N6592 error {forward_error:e}"
        );
        let restored = context
            .execute_transform_complex32(&inverse, &actual)
            .unwrap();
        let round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re as f64;
                let di = actual.im as f64 - expected.im as f64;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(round_trip_error <= 4.0e-3);
    }
}
