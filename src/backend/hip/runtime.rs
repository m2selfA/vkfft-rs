//! Dynamically loaded HIP Runtime + HIPRTC correctness execution.
//!
//! The implementation consumes the same backend-neutral `ProgramIr` used by CUDA and
//! OpenCL. HIP/HIPRTC remain optional runtime dependencies; when ROCm or a device is
//! absent, probing is fail-soft. Real-device goldens are conditional on AMD hardware.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use libloading::Library;

#[cfg(all(windows, feature = "opencl-runtime"))]
use crate::backend::opencl::runtime::OpenClExecutionContext;

use crate::backend::native::{NativeProgramSource, NativeShaderSource};
use crate::backend::native_runtime::{
    NativeCompiledPassResourceReport, NativeCompiledResourceMetrics, NativeRuntimeAvailability,
    NativeTransformInput32, NativeTransformInput64, NativeTransformOutput32,
    NativeTransformOutput64, PreparedProgramStorage, load_first_library, prepare_program_complex32,
    prepare_program_complex64, runtime_error, unavailable,
};
use crate::complex::{Complex32, Complex64};
use crate::config::{Backend, DeviceProfile, GpuVendor, SubgroupProfile};
use crate::error::{Result, VkFftError};
use crate::program_ir::ProgramAllocationKind;
use crate::{ScalarType, TransformIr};

const HIP_SUCCESS: c_int = 0;
const HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
const HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
const HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
const HIP_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
const HIP_MODULE_CACHE_MAX_ENTRIES: usize = 64;

type HipInit = unsafe extern "C" fn(c_uint) -> c_int;
type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type HipRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type HipDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type HipStream = *mut c_void;
type HipStreamCreate = unsafe extern "C" fn(*mut HipStream) -> c_int;
type HipStreamDestroy = unsafe extern "C" fn(HipStream) -> c_int;
type HipStreamSynchronize = unsafe extern "C" fn(HipStream) -> c_int;
type HipDeviceSynchronize = unsafe extern "C" fn() -> c_int;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> c_int;
type HipModuleUnload = unsafe extern "C" fn(HipModule) -> c_int;
type HipModuleGetFunction =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> c_int;
type HipFuncGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, HipFunction) -> c_int;
type HipModuleLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

/// Native HIP lowering emits every typed shared allocation as a static `__shared__`
/// declaration in the code object. `hipModuleLaunchKernel`'s dynamic-shared argument
/// must therefore stay zero; `NativeShaderSource::required_shared_memory_bytes` is
/// resource-accounting metadata for this backend, not a second allocation request.
const HIP_DYNAMIC_SHARED_MEMORY_BYTES: c_uint = 0;

type HiprtcProgram = *mut c_void;
type HiprtcCreateProgram = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type HiprtcDestroyProgram = unsafe extern "C" fn(*mut HiprtcProgram) -> c_int;
type HiprtcCompileProgram =
    unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> c_int;
type HiprtcGetCodeSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> c_int;
type HiprtcGetCode = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> c_int;
type HiprtcGetProgramLogSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> c_int;
type HiprtcGetProgramLog = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> c_int;

const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const HIPRTC_SUCCESS: c_int = 0;
const HIPRTC_CACHE_MAX_ENTRIES: usize = 64;
const HIPRTC_CACHE_MAX_BYTES: usize = 32 * 1024 * 1024;
pub const HIPRTC_CODE_ARCHIVE_VERSION: u32 = 2;
const HIPRTC_CODE_ARCHIVE_MAGIC: &[u8; 8] = b"VKFTHIPC";
const HIPRTC_CODE_ARCHIVE_COMMIT_BYTES: usize = 40;
const HIPRTC_CODE_ARCHIVE_HEADER_BYTES: usize =
    8 + 4 + HIPRTC_CODE_ARCHIVE_COMMIT_BYTES + 4 + 8 + 8 + 8;

const HIP_RUNTIME_LIBRARY_CANDIDATES: &[&str] = &[
    "amdhip64.dll",
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "/opt/rocm/lib/libamdhip64.so",
];
const HIPRTC_LIBRARY_CANDIDATES: &[&str] = &[
    "libhiprtc.so",
    "libhiprtc.so.7",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
    "/opt/rocm/lib/libhiprtc.so",
];

#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_LIBRARY_CANDIDATES: &[&str] = &["amd_comgr.dll"];
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_SUCCESS: c_int = 0;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_LANGUAGE_OPENCL_2_0: c_int = 0x2;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_DATA_KIND_SOURCE: c_int = 0x1;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_DATA_KIND_DIAGNOSTIC: c_int = 0x4;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_DATA_KIND_LOG: c_int = 0x5;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_DATA_KIND_BC: c_int = 0x6;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_DATA_KIND_RELOCATABLE: c_int = 0x7;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_DATA_KIND_EXECUTABLE: c_int = 0x8;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE: c_int = 0x6;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE: c_int = 0x9;
#[cfg(all(windows, feature = "opencl-runtime"))]
const AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC: c_int = 0xF;

#[cfg(all(windows, feature = "opencl-runtime"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AmdComgrHandle {
    handle: u64,
}

#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrCreateDataSet = unsafe extern "C" fn(*mut AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrDestroyDataSet = unsafe extern "C" fn(AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrCreateData = unsafe extern "C" fn(c_int, *mut AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrReleaseData = unsafe extern "C" fn(AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrSetData = unsafe extern "C" fn(AmdComgrHandle, usize, *const c_void) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrSetDataName = unsafe extern "C" fn(AmdComgrHandle, *const c_char) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrDataSetAdd = unsafe extern "C" fn(AmdComgrHandle, AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrCreateActionInfo = unsafe extern "C" fn(*mut AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrDestroyActionInfo = unsafe extern "C" fn(AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrActionInfoSetLanguage = unsafe extern "C" fn(AmdComgrHandle, c_int) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrActionInfoSetIsaName = unsafe extern "C" fn(AmdComgrHandle, *const c_char) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrActionInfoSetOptionList =
    unsafe extern "C" fn(AmdComgrHandle, *const *const c_char, usize) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrActionInfoSetLogging = unsafe extern "C" fn(AmdComgrHandle, bool) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrDoAction =
    unsafe extern "C" fn(c_int, AmdComgrHandle, AmdComgrHandle, AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrActionDataCount = unsafe extern "C" fn(AmdComgrHandle, c_int, *mut usize) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrActionDataGetData =
    unsafe extern "C" fn(AmdComgrHandle, c_int, usize, *mut AmdComgrHandle) -> c_int;
#[cfg(all(windows, feature = "opencl-runtime"))]
type AmdComgrGetData = unsafe extern "C" fn(AmdComgrHandle, *mut usize, *mut c_void) -> c_int;

// ROCm 5.0 reordered hipDeviceAttribute_t into CUDA-compatible and AMD-specific
// ranges. Keep the pre-5 ABI for an unversioned legacy runtime, while all ROCm 5+
// SONAMEs use the stable CUDA-compatible values.
const HIP_VERSION_5_0: c_int = 50_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HipDeviceAttributeAbi {
    max_block_dim_x: c_int,
    max_block_dim_y: c_int,
    max_block_dim_z: c_int,
    max_threads_per_block: c_int,
    max_shared_memory_per_block: c_int,
    warp_size: c_int,
}

const HIP_DEVICE_ATTRIBUTES_PRE_5: HipDeviceAttributeAbi = HipDeviceAttributeAbi {
    max_block_dim_x: 1,
    max_block_dim_y: 2,
    max_block_dim_z: 3,
    max_threads_per_block: 0,
    max_shared_memory_per_block: 7,
    warp_size: 9,
};

const HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE: HipDeviceAttributeAbi = HipDeviceAttributeAbi {
    max_block_dim_x: 26,
    max_block_dim_y: 27,
    max_block_dim_z: 28,
    max_threads_per_block: 56,
    max_shared_memory_per_block: 74,
    warp_size: 87,
};

const fn hip_device_attribute_abi(runtime_version: c_int) -> HipDeviceAttributeAbi {
    if runtime_version >= HIP_VERSION_5_0 {
        HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE
    } else {
        HIP_DEVICE_ATTRIBUTES_PRE_5
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HipDeviceAttributes {
    shared_memory_bytes: usize,
    max_threads_per_block: usize,
    max_block_dim: [usize; 3],
    warp_size: usize,
}

fn hip_profile_from_attributes(attributes: HipDeviceAttributes) -> Result<DeviceProfile> {
    if attributes.shared_memory_bytes == 0
        || attributes.max_threads_per_block == 0
        || attributes.max_block_dim.contains(&0)
        || attributes.warp_size == 0
        || !attributes.warp_size.is_power_of_two()
        || attributes.warp_size > attributes.max_threads_per_block
    {
        return Err(runtime_error(
            Backend::Hip,
            "HIP device attributes contain an invalid shared-memory, workgroup, or warp limit",
        ));
    }
    Ok(DeviceProfile {
        backend: Backend::Hip,
        vendor: GpuVendor::Amd,
        shared_memory_bytes: attributes.shared_memory_bytes,
        shared_memory_pow2_bytes: floor_power_of_two(attributes.shared_memory_bytes),
        max_threads_per_block: attributes.max_threads_per_block,
        max_workgroup_size: attributes.max_block_dim,
        coalesced_memory_bytes: 32,
        shared_banks: 32,
        // HIP-supported AMD architectures expose double arithmetic; exact throughput
        // is architecture-dependent but does not change the scheduler capability bit.
        supports_f64: true,
        subgroup: SubgroupProfile {
            size: attributes.warp_size,
            min_size: attributes.warp_size,
            max_size: attributes.warp_size,
            required_size_compute_supported: false,
            compute_supported: true,
            basic_supported: true,
            shuffle_supported: true,
            shuffle_relative_supported: true,
            compute_full_subgroups: true,
        },
    })
}

struct HipRuntimeApi {
    _library: Library,
    init: HipInit,
    get_device_count: HipGetDeviceCount,
    set_device: HipSetDevice,
    device_get_name: HipDeviceGetName,
    runtime_get_version: HipRuntimeGetVersion,
    device_get_attribute: HipDeviceGetAttribute,
    malloc: HipMalloc,
    free: HipFree,
    memcpy: HipMemcpy,
    stream_create: HipStreamCreate,
    stream_destroy: HipStreamDestroy,
    stream_synchronize: HipStreamSynchronize,
    device_synchronize: HipDeviceSynchronize,
    module_load_data: HipModuleLoadData,
    module_unload: HipModuleUnload,
    module_get_function: HipModuleGetFunction,
    func_get_attribute: Option<HipFuncGetAttribute>,
    module_launch_kernel: HipModuleLaunchKernel,
}

impl HipRuntimeApi {
    fn load() -> Result<Self> {
        let (library, _) = load_first_library(Backend::Hip, HIP_RUNTIME_LIBRARY_CANDIDATES)?;
        Ok(Self {
            init: load_symbol(&library, b"hipInit\0")?,
            get_device_count: load_symbol(&library, b"hipGetDeviceCount\0")?,
            set_device: load_symbol(&library, b"hipSetDevice\0")?,
            device_get_name: load_symbol(&library, b"hipDeviceGetName\0")?,
            runtime_get_version: load_symbol(&library, b"hipRuntimeGetVersion\0")?,
            device_get_attribute: load_symbol(&library, b"hipDeviceGetAttribute\0")?,
            malloc: load_symbol(&library, b"hipMalloc\0")?,
            free: load_symbol(&library, b"hipFree\0")?,
            memcpy: load_symbol(&library, b"hipMemcpy\0")?,
            stream_create: load_symbol(&library, b"hipStreamCreate\0")?,
            stream_destroy: load_symbol(&library, b"hipStreamDestroy\0")?,
            stream_synchronize: load_symbol(&library, b"hipStreamSynchronize\0")?,
            device_synchronize: load_symbol(&library, b"hipDeviceSynchronize\0")?,
            module_load_data: load_symbol(&library, b"hipModuleLoadData\0")?,
            module_unload: load_symbol(&library, b"hipModuleUnload\0")?,
            module_get_function: load_symbol(&library, b"hipModuleGetFunction\0")?,
            func_get_attribute: load_optional_symbol(&library, b"hipFuncGetAttribute\0"),
            module_launch_kernel: load_symbol(&library, b"hipModuleLaunchKernel\0")?,
            _library: library,
        })
    }
}

impl HipRuntimeApi {
    fn runtime_version(&self) -> Result<c_int> {
        let mut version = 0;
        check_hip(
            unsafe { (self.runtime_get_version)(&mut version) },
            "hipRuntimeGetVersion",
        )?;
        if version <= 0 {
            return Err(runtime_error(
                Backend::Hip,
                format!("hipRuntimeGetVersion returned invalid version {version}"),
            ));
        }
        Ok(version)
    }

    fn device_attribute(
        &self,
        device_index: usize,
        attribute: c_int,
        label: &'static str,
    ) -> Result<usize> {
        let device = c_int::try_from(device_index).map_err(|_| VkFftError::ValueOutOfRange {
            field: "HIP device index",
        })?;
        let mut value = 0;
        check_hip(
            unsafe { (self.device_get_attribute)(&mut value, attribute, device) },
            label,
        )?;
        usize::try_from(value).map_err(|_| {
            runtime_error(
                Backend::Hip,
                format!("{label} returned negative value {value}"),
            )
        })
    }

    fn device_attributes(&self, device_index: usize) -> Result<HipDeviceAttributes> {
        let version = self.runtime_version()?;
        let abi = hip_device_attribute_abi(version);
        Ok(HipDeviceAttributes {
            shared_memory_bytes: self.device_attribute(
                device_index,
                abi.max_shared_memory_per_block,
                "hipDeviceGetAttribute(MaxSharedMemoryPerBlock)",
            )?,
            max_threads_per_block: self.device_attribute(
                device_index,
                abi.max_threads_per_block,
                "hipDeviceGetAttribute(MaxThreadsPerBlock)",
            )?,
            max_block_dim: [
                self.device_attribute(
                    device_index,
                    abi.max_block_dim_x,
                    "hipDeviceGetAttribute(MaxBlockDimX)",
                )?,
                self.device_attribute(
                    device_index,
                    abi.max_block_dim_y,
                    "hipDeviceGetAttribute(MaxBlockDimY)",
                )?,
                self.device_attribute(
                    device_index,
                    abi.max_block_dim_z,
                    "hipDeviceGetAttribute(MaxBlockDimZ)",
                )?,
            ],
            warp_size: self.device_attribute(
                device_index,
                abi.warp_size,
                "hipDeviceGetAttribute(WarpSize)",
            )?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HiprtcCompileMode {
    Default,
    DisableFpContraction,
}

impl HiprtcCompileMode {
    const fn for_scalar(scalar: ScalarType) -> Self {
        if matches!(scalar, ScalarType::DoubleDouble) {
            Self::DisableFpContraction
        } else {
            Self::Default
        }
    }

    const fn archive_tag(self) -> u8 {
        match self {
            Self::Default => 0,
            Self::DisableFpContraction => 1,
        }
    }

    fn from_archive_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::Default),
            1 => Ok(Self::DisableFpContraction),
            _ => Err(runtime_error(
                Backend::Hip,
                format!("HIPRTC code archive compile-mode tag {tag} is unsupported"),
            )),
        }
    }

    const fn clang_options(self) -> &'static [&'static str] {
        match self {
            Self::Default => &["--std=c++14"],
            Self::DisableFpContraction => &["--std=c++14", "-ffp-contract=off"],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct HiprtcCodeCacheKey {
    source: String,
    compile_mode: HiprtcCompileMode,
}

impl HiprtcCodeCacheKey {
    fn new(source: &str, compile_mode: HiprtcCompileMode) -> Self {
        Self {
            source: source.to_owned(),
            compile_mode,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiprtcCodeArchiveIdentity {
    pub runtime_version: i32,
    pub device_name: String,
    pub warp_size: usize,
}

/// Versioned persistent form of HIPRTC's source-to-code cache. The archive skips
/// recompilation only; loaded HIP modules/functions remain ticket-local. Artifacts are
/// accepted only for the same runtime/device/wave identity and fixed upstream baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiprtcCodeArchive {
    pub identity: HiprtcCodeArchiveIdentity,
    entries: Vec<(HiprtcCodeCacheKey, Vec<u8>)>,
}

impl HiprtcCodeArchive {
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn encode(&self) -> Vec<u8> {
        let commit = crate::UPSTREAM_VKFFT_COMMIT.as_bytes();
        debug_assert_eq!(commit.len(), HIPRTC_CODE_ARCHIVE_COMMIT_BYTES);
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(HIPRTC_CODE_ARCHIVE_MAGIC);
        bytes.extend_from_slice(&HIPRTC_CODE_ARCHIVE_VERSION.to_le_bytes());
        bytes.extend_from_slice(commit);
        bytes.extend_from_slice(&self.identity.runtime_version.to_le_bytes());
        bytes.extend_from_slice(&(self.identity.warp_size as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.identity.device_name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.identity.device_name.as_bytes());
        for (key, code) in entries {
            bytes.extend_from_slice(&(key.source.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(code.len() as u64).to_le_bytes());
            bytes.push(key.compile_mode.archive_tag());
            bytes.extend_from_slice(key.source.as_bytes());
            bytes.extend_from_slice(code);
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let malformed = |message: &str| runtime_error(Backend::Hip, message);
        if bytes.len() < HIPRTC_CODE_ARCHIVE_HEADER_BYTES {
            return Err(malformed("HIPRTC code archive is truncated"));
        }
        if &bytes[..8] != HIPRTC_CODE_ARCHIVE_MAGIC {
            return Err(malformed("HIPRTC code archive magic does not match"));
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| malformed("HIPRTC code archive version is malformed"))?,
        );
        if version != HIPRTC_CODE_ARCHIVE_VERSION {
            return Err(malformed(
                "HIPRTC code archive schema version is unsupported",
            ));
        }
        let commit_start = 12;
        let commit_end = commit_start + HIPRTC_CODE_ARCHIVE_COMMIT_BYTES;
        if &bytes[commit_start..commit_end] != crate::UPSTREAM_VKFFT_COMMIT.as_bytes() {
            return Err(malformed(
                "HIPRTC code archive upstream commit does not match",
            ));
        }
        let mut offset = commit_end;
        let runtime_end = offset + 4;
        let runtime_version = i32::from_le_bytes(
            bytes[offset..runtime_end]
                .try_into()
                .map_err(|_| malformed("HIPRTC code archive runtime version is malformed"))?,
        );
        offset = runtime_end;
        let warp_size = hiprtc_archive_usize(bytes, &mut offset, "warp-size")?;
        let device_name_len = hiprtc_archive_usize(bytes, &mut offset, "device-name length")?;
        let entry_count = hiprtc_archive_usize(bytes, &mut offset, "entry count")?;
        if entry_count > HIPRTC_CACHE_MAX_ENTRIES {
            return Err(malformed(
                "HIPRTC code archive exceeds the cache entry limit",
            ));
        }
        let device_name =
            hiprtc_archive_string(bytes, &mut offset, device_name_len, "device name")?;
        let mut decoded = HashMap::<HiprtcCodeCacheKey, Vec<u8>>::with_capacity(entry_count);
        let mut total_bytes = 0usize;
        for _ in 0..entry_count {
            let source_len = hiprtc_archive_usize(bytes, &mut offset, "source length")?;
            let code_len = hiprtc_archive_usize(bytes, &mut offset, "code length")?;
            total_bytes = total_bytes
                .checked_add(source_len)
                .and_then(|value| value.checked_add(code_len))
                .ok_or_else(|| malformed("HIPRTC code archive payload size overflows"))?;
            if total_bytes > HIPRTC_CACHE_MAX_BYTES {
                return Err(malformed(
                    "HIPRTC code archive exceeds the cache byte limit",
                ));
            }
            let mode = HiprtcCompileMode::from_archive_tag(
                hiprtc_archive_take(bytes, &mut offset, 1, "compile mode")?[0],
            )?;
            let source = hiprtc_archive_string(bytes, &mut offset, source_len, "source")?;
            let code = hiprtc_archive_take(bytes, &mut offset, code_len, "code")?.to_vec();
            let key = HiprtcCodeCacheKey {
                source,
                compile_mode: mode,
            };
            if decoded.insert(key, code).is_some() {
                return Err(malformed(
                    "HIPRTC code archive contains duplicate source/mode entries",
                ));
            }
        }
        if offset != bytes.len() {
            return Err(malformed("HIPRTC code archive has trailing bytes"));
        }
        let mut entries = decoded.into_iter().collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        Ok(Self {
            identity: HiprtcCodeArchiveIdentity {
                runtime_version,
                device_name,
                warp_size,
            },
            entries,
        })
    }
}

fn hiprtc_archive_take<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    len: usize,
    field: &str,
) -> Result<&'a [u8]> {
    let end = offset.checked_add(len).ok_or_else(|| {
        runtime_error(
            Backend::Hip,
            format!("HIPRTC code archive {field} length overflows"),
        )
    })?;
    if end > bytes.len() {
        return Err(runtime_error(
            Backend::Hip,
            format!("HIPRTC code archive {field} is truncated"),
        ));
    }
    let value = &bytes[*offset..end];
    *offset = end;
    Ok(value)
}

fn hiprtc_archive_usize(bytes: &[u8], offset: &mut usize, field: &str) -> Result<usize> {
    let raw = hiprtc_archive_take(bytes, offset, 8, field)?;
    let value = u64::from_le_bytes(raw.try_into().map_err(|_| {
        runtime_error(
            Backend::Hip,
            format!("HIPRTC code archive {field} is malformed"),
        )
    })?);
    usize::try_from(value).map_err(|_| {
        runtime_error(
            Backend::Hip,
            format!("HIPRTC code archive {field} does not fit this platform"),
        )
    })
}

fn hiprtc_archive_string(
    bytes: &[u8],
    offset: &mut usize,
    len: usize,
    field: &str,
) -> Result<String> {
    String::from_utf8(hiprtc_archive_take(bytes, offset, len, field)?.to_vec()).map_err(|_| {
        runtime_error(
            Backend::Hip,
            format!("HIPRTC code archive {field} is not valid UTF-8"),
        )
    })
}

#[derive(Debug)]
struct HiprtcCodeCache {
    entries: HashMap<HiprtcCodeCacheKey, Vec<u8>>,
    insertion_order: VecDeque<HiprtcCodeCacheKey>,
    total_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl HiprtcCodeCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            total_bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    fn get(&self, key: &HiprtcCodeCacheKey) -> Option<Vec<u8>> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: HiprtcCodeCacheKey, code: &[u8]) {
        let entry_bytes = key.source.len().saturating_add(code.len());
        if self.max_entries == 0 || entry_bytes > self.max_bytes {
            return;
        }
        if let Some(previous) = self.entries.remove(&key) {
            self.total_bytes = self
                .total_bytes
                .saturating_sub(key.source.len().saturating_add(previous.len()));
            self.insertion_order.retain(|candidate| candidate != &key);
        }
        while !self.insertion_order.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.total_bytes.saturating_add(entry_bytes) > self.max_bytes)
        {
            let oldest = self
                .insertion_order
                .pop_front()
                .expect("non-empty HIPRTC cache order");
            if let Some(previous) = self.entries.remove(&oldest) {
                self.total_bytes = self
                    .total_bytes
                    .saturating_sub(oldest.source.len().saturating_add(previous.len()));
            }
        }
        if self.entries.len() < self.max_entries
            && self.total_bytes.saturating_add(entry_bytes) <= self.max_bytes
        {
            self.total_bytes = self.total_bytes.saturating_add(entry_bytes);
            self.insertion_order.push_back(key.clone());
            self.entries.insert(key, code.to_vec());
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
        self.total_bytes = 0;
    }
}

struct HiprtcApi {
    _library: Library,
    create_program: HiprtcCreateProgram,
    destroy_program: HiprtcDestroyProgram,
    compile_program: HiprtcCompileProgram,
    get_code_size: HiprtcGetCodeSize,
    get_code: HiprtcGetCode,
    get_log_size: HiprtcGetProgramLogSize,
    get_log: HiprtcGetProgramLog,
}

impl HiprtcApi {
    fn load() -> Result<Self> {
        let (library, _) = load_first_library(Backend::Hip, HIPRTC_LIBRARY_CANDIDATES)?;
        Ok(Self {
            create_program: load_symbol(&library, b"hiprtcCreateProgram\0")?,
            destroy_program: load_symbol(&library, b"hiprtcDestroyProgram\0")?,
            compile_program: load_symbol(&library, b"hiprtcCompileProgram\0")?,
            get_code_size: load_symbol(&library, b"hiprtcGetCodeSize\0")?,
            get_code: load_symbol(&library, b"hiprtcGetCode\0")?,
            get_log_size: load_symbol(&library, b"hiprtcGetProgramLogSize\0")?,
            get_log: load_symbol(&library, b"hiprtcGetProgramLog\0")?,
            _library: library,
        })
    }

    fn compile_code(&self, source: &str, mode: HiprtcCompileMode) -> Result<Vec<u8>> {
        let source = CString::new(source)
            .map_err(|_| runtime_error(Backend::Hip, "HIP source contains an interior NUL byte"))?;
        let name = CString::new("vkfft_rs.hip").expect("static CString");
        let mut program = ptr::null_mut();
        check_hiprtc(
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
            "hiprtcCreateProgram",
        )?;
        let option_strings = mode
            .clang_options()
            .iter()
            .map(|option| CString::new(*option).expect("static HIPRTC option"))
            .collect::<Vec<_>>();
        let options = option_strings
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        let option_count = c_int::try_from(options.len()).map_err(|_| {
            runtime_error(
                Backend::Hip,
                "HIPRTC compile option count does not fit c_int",
            )
        })?;
        let compile = unsafe { (self.compile_program)(program, option_count, options.as_ptr()) };
        if compile != HIPRTC_SUCCESS {
            let log = self.program_log(program);
            unsafe {
                (self.destroy_program)(&mut program);
            }
            return Err(VkFftError::ShaderCompilation(format!(
                "HIPRTC returned {compile}: {log}"
            )));
        }
        let mut size = 0usize;
        check_hiprtc(
            unsafe { (self.get_code_size)(program, &mut size) },
            "hiprtcGetCodeSize",
        )?;
        let mut code = vec![0u8; size];
        check_hiprtc(
            unsafe { (self.get_code)(program, code.as_mut_ptr().cast()) },
            "hiprtcGetCode",
        )?;
        check_hiprtc(
            unsafe { (self.destroy_program)(&mut program) },
            "hiprtcDestroyProgram",
        )?;
        Ok(code)
    }

    fn program_log(&self, program: HiprtcProgram) -> String {
        let mut size = 0usize;
        if unsafe { (self.get_log_size)(program, &mut size) } != HIPRTC_SUCCESS || size == 0 {
            return "no HIPRTC log available".to_owned();
        }
        let mut bytes = vec![0u8; size];
        if unsafe { (self.get_log)(program, bytes.as_mut_ptr().cast()) } != HIPRTC_SUCCESS {
            return "failed to read HIPRTC log".to_owned();
        }
        CStr::from_bytes_until_nul(&bytes)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(all(windows, feature = "opencl-runtime"))]
struct AmdComgrApi {
    _library: Library,
    create_data_set: AmdComgrCreateDataSet,
    destroy_data_set: AmdComgrDestroyDataSet,
    create_data: AmdComgrCreateData,
    release_data: AmdComgrReleaseData,
    set_data: AmdComgrSetData,
    set_data_name: AmdComgrSetDataName,
    data_set_add: AmdComgrDataSetAdd,
    create_action_info: AmdComgrCreateActionInfo,
    destroy_action_info: AmdComgrDestroyActionInfo,
    action_info_set_language: AmdComgrActionInfoSetLanguage,
    action_info_set_isa_name: AmdComgrActionInfoSetIsaName,
    action_info_set_option_list: AmdComgrActionInfoSetOptionList,
    action_info_set_logging: AmdComgrActionInfoSetLogging,
    do_action: AmdComgrDoAction,
    action_data_count: AmdComgrActionDataCount,
    action_data_get_data: AmdComgrActionDataGetData,
    get_data: AmdComgrGetData,
}

#[cfg(all(windows, feature = "opencl-runtime"))]
impl AmdComgrApi {
    fn load() -> Result<Self> {
        let (library, _) = load_first_library(Backend::Hip, AMD_COMGR_LIBRARY_CANDIDATES)?;
        Ok(Self {
            create_data_set: load_symbol(&library, b"amd_comgr_create_data_set\0")?,
            destroy_data_set: load_symbol(&library, b"amd_comgr_destroy_data_set\0")?,
            create_data: load_symbol(&library, b"amd_comgr_create_data\0")?,
            release_data: load_symbol(&library, b"amd_comgr_release_data\0")?,
            set_data: load_symbol(&library, b"amd_comgr_set_data\0")?,
            set_data_name: load_symbol(&library, b"amd_comgr_set_data_name\0")?,
            data_set_add: load_symbol(&library, b"amd_comgr_data_set_add\0")?,
            create_action_info: load_symbol(&library, b"amd_comgr_create_action_info\0")?,
            destroy_action_info: load_symbol(&library, b"amd_comgr_destroy_action_info\0")?,
            action_info_set_language: load_symbol(
                &library,
                b"amd_comgr_action_info_set_language\0",
            )?,
            action_info_set_isa_name: load_symbol(
                &library,
                b"amd_comgr_action_info_set_isa_name\0",
            )?,
            action_info_set_option_list: load_symbol(
                &library,
                b"amd_comgr_action_info_set_option_list\0",
            )?,
            action_info_set_logging: load_symbol(&library, b"amd_comgr_action_info_set_logging\0")?,
            do_action: load_symbol(&library, b"amd_comgr_do_action\0")?,
            action_data_count: load_symbol(&library, b"amd_comgr_action_data_count\0")?,
            action_data_get_data: load_symbol(&library, b"amd_comgr_action_data_get_data\0")?,
            get_data: load_symbol(&library, b"amd_comgr_get_data\0")?,
            _library: library,
        })
    }
}

#[cfg(all(windows, feature = "opencl-runtime"))]
fn check_comgr(status: c_int, operation: &str) -> Result<()> {
    if status == AMD_COMGR_SUCCESS {
        Ok(())
    } else {
        Err(VkFftError::ShaderCompilation(format!(
            "AMD COMGR {operation} returned status {status}"
        )))
    }
}

#[cfg(all(windows, feature = "opencl-runtime"))]
fn amd_comgr_action_messages(
    api: &AmdComgrApi,
    data_set: AmdComgrHandle,
    kind: c_int,
) -> Vec<String> {
    let mut count = 0usize;
    if unsafe { (api.action_data_count)(data_set, kind, &mut count) } != AMD_COMGR_SUCCESS {
        return Vec::new();
    }
    let mut messages = Vec::new();
    for index in 0..count {
        let mut data = AmdComgrHandle::default();
        if unsafe { (api.action_data_get_data)(data_set, kind, index, &mut data) }
            != AMD_COMGR_SUCCESS
        {
            continue;
        }
        let message = (|| {
            let mut bytes = 0usize;
            if unsafe { (api.get_data)(data, &mut bytes, ptr::null_mut()) } != AMD_COMGR_SUCCESS
                || bytes == 0
            {
                return None;
            }
            let mut buffer = vec![0u8; bytes];
            if unsafe { (api.get_data)(data, &mut bytes, buffer.as_mut_ptr().cast::<c_void>()) }
                != AMD_COMGR_SUCCESS
            {
                return None;
            }
            buffer.truncate(bytes);
            let text = String::from_utf8_lossy(&buffer);
            let text = text.trim_matches('\0').trim();
            (!text.is_empty()).then(|| text.to_owned())
        })();
        unsafe {
            (api.release_data)(data);
        }
        if let Some(message) = message {
            messages.push(message);
        }
    }
    messages
}

#[cfg(all(windows, feature = "opencl-runtime"))]
fn check_comgr_action(
    api: &AmdComgrApi,
    status: c_int,
    output: AmdComgrHandle,
    operation: &str,
) -> Result<()> {
    if status == AMD_COMGR_SUCCESS {
        return Ok(());
    }
    let mut messages = amd_comgr_action_messages(api, output, AMD_COMGR_DATA_KIND_LOG);
    messages.extend(amd_comgr_action_messages(
        api,
        output,
        AMD_COMGR_DATA_KIND_DIAGNOSTIC,
    ));
    let detail = if messages.is_empty() {
        "no COMGR action log was produced".to_owned()
    } else {
        messages.join("\n")
    };
    Err(VkFftError::ShaderCompilation(format!(
        "AMD COMGR {operation} returned status {status}: {detail}"
    )))
}

#[cfg(any(test, all(windows, feature = "opencl-runtime")))]
fn amd_comgr_opencl_source(source: &str) -> String {
    let mut prefixed = String::from(
        "typedef unsigned short ushort;\ntypedef unsigned int uint;\ntypedef uint cl_mem_fence_flags;\ntypedef unsigned long ulong;\ntypedef float float2 __attribute__((ext_vector_type(2)));\ntypedef float float4 __attribute__((ext_vector_type(4)));\n#define CLK_LOCAL_MEM_FENCE 1u\n#define CLK_GLOBAL_MEM_FENCE 2u\ninline float as_float(uint value) { union { uint bits; float scalar; } bitcast; bitcast.bits = value; return bitcast.scalar; }\ninline uint as_uint(float value) { union { uint bits; float scalar; } bitcast; bitcast.scalar = value; return bitcast.bits; }\nfloat sub_group_broadcast(float, uint) __attribute__((overloadable, convergent));\nuint get_sub_group_local_id(void) __attribute__((overloadable, convergent));\nuint get_sub_group_id(void) __attribute__((overloadable, convergent));\n",
    );
    if source.contains("double2") || source.contains("double4") {
        prefixed.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        prefixed.push_str(
            "double sub_group_broadcast(double, uint) __attribute__((overloadable, convergent));\n",
        );
        prefixed.push_str(
            "typedef double double2 __attribute__((ext_vector_type(2)));\ntypedef double double4 __attribute__((ext_vector_type(4)));\n",
        );
    }
    prefixed.push_str(source);
    prefixed
}

#[cfg(all(windows, feature = "opencl-runtime"))]
struct AmdComgrCompiler {
    api: AmdComgrApi,
    isa_name: CString,
}

#[cfg(all(windows, feature = "opencl-runtime"))]
impl AmdComgrCompiler {
    fn load(isa_name: &str) -> Result<Self> {
        let isa_name = CString::new(isa_name).map_err(|_| {
            unavailable(
                Backend::Hip,
                "AMD COMGR ISA name contains an interior NUL byte",
            )
        })?;
        Ok(Self {
            api: AmdComgrApi::load()?,
            isa_name,
        })
    }

    fn compile_opencl_executable(&self, source: &str) -> Result<Vec<u8>> {
        let source = amd_comgr_opencl_source(source);
        let source_name = c"vkfft.cl";
        let mut data_sets = Vec::<AmdComgrHandle>::new();
        let mut data_objects = Vec::<AmdComgrHandle>::new();
        let mut action_info = None::<AmdComgrHandle>;
        let result = (|| {
            let mut input = AmdComgrHandle::default();
            check_comgr(
                unsafe { (self.api.create_data_set)(&mut input) },
                "create input data set",
            )?;
            data_sets.push(input);

            let mut source_data = AmdComgrHandle::default();
            check_comgr(
                unsafe { (self.api.create_data)(AMD_COMGR_DATA_KIND_SOURCE, &mut source_data) },
                "create source data",
            )?;
            data_objects.push(source_data);
            check_comgr(
                unsafe {
                    (self.api.set_data)(source_data, source.len(), source.as_ptr().cast::<c_void>())
                },
                "set source data",
            )?;
            check_comgr(
                unsafe { (self.api.set_data_name)(source_data, source_name.as_ptr()) },
                "set source name",
            )?;
            check_comgr(
                unsafe { (self.api.data_set_add)(input, source_data) },
                "add source data",
            )?;

            let mut info = AmdComgrHandle::default();
            check_comgr(
                unsafe { (self.api.create_action_info)(&mut info) },
                "create action info",
            )?;
            action_info = Some(info);
            check_comgr(
                unsafe { (self.api.action_info_set_logging)(info, true) },
                "enable action logging",
            )?;
            check_comgr(
                unsafe { (self.api.action_info_set_language)(info, AMD_COMGR_LANGUAGE_OPENCL_2_0) },
                "set OpenCL 2.0 language",
            )?;
            check_comgr(
                unsafe { (self.api.action_info_set_isa_name)(info, self.isa_name.as_ptr()) },
                "set target ISA",
            )?;

            let declare_builtins = c"-fdeclare-opencl-builtins";
            let clang = c"-Xclang";
            let compile_options = [clang.as_ptr(), declare_builtins.as_ptr()];
            check_comgr(
                unsafe {
                    (self.api.action_info_set_option_list)(
                        info,
                        compile_options.as_ptr(),
                        compile_options.len(),
                    )
                },
                "set OpenCL compiler options",
            )?;
            let mut bitcode = AmdComgrHandle::default();
            check_comgr(
                unsafe { (self.api.create_data_set)(&mut bitcode) },
                "create bitcode data set",
            )?;
            data_sets.push(bitcode);
            check_comgr_action(
                &self.api,
                unsafe {
                    (self.api.do_action)(
                        AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                        info,
                        input,
                        bitcode,
                    )
                },
                bitcode,
                "compile OpenCL source with device libraries",
            )?;
            Self::require_single_kind(&self.api, bitcode, AMD_COMGR_DATA_KIND_BC, "bitcode")?;

            let optimize = c"-O2";
            let codegen_options = [optimize.as_ptr()];
            check_comgr(
                unsafe {
                    (self.api.action_info_set_option_list)(
                        info,
                        codegen_options.as_ptr(),
                        codegen_options.len(),
                    )
                },
                "set codegen options",
            )?;
            let mut relocatable = AmdComgrHandle::default();
            check_comgr(
                unsafe { (self.api.create_data_set)(&mut relocatable) },
                "create relocatable data set",
            )?;
            data_sets.push(relocatable);
            check_comgr_action(
                &self.api,
                unsafe {
                    (self.api.do_action)(
                        AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                        info,
                        bitcode,
                        relocatable,
                    )
                },
                relocatable,
                "codegen bitcode to relocatable",
            )?;
            Self::require_single_kind(
                &self.api,
                relocatable,
                AMD_COMGR_DATA_KIND_RELOCATABLE,
                "relocatable",
            )?;

            check_comgr(
                unsafe { (self.api.action_info_set_option_list)(info, ptr::null(), 0) },
                "clear link options",
            )?;
            let mut executable = AmdComgrHandle::default();
            check_comgr(
                unsafe { (self.api.create_data_set)(&mut executable) },
                "create executable data set",
            )?;
            data_sets.push(executable);
            check_comgr_action(
                &self.api,
                unsafe {
                    (self.api.do_action)(
                        AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                        info,
                        relocatable,
                        executable,
                    )
                },
                executable,
                "link relocatable to executable",
            )?;
            Self::require_single_kind(
                &self.api,
                executable,
                AMD_COMGR_DATA_KIND_EXECUTABLE,
                "executable",
            )?;

            let mut executable_data = AmdComgrHandle::default();
            check_comgr(
                unsafe {
                    (self.api.action_data_get_data)(
                        executable,
                        AMD_COMGR_DATA_KIND_EXECUTABLE,
                        0,
                        &mut executable_data,
                    )
                },
                "get executable data",
            )?;
            data_objects.push(executable_data);
            let mut bytes = 0usize;
            check_comgr(
                unsafe { (self.api.get_data)(executable_data, &mut bytes, ptr::null_mut()) },
                "get executable size",
            )?;
            if bytes == 0 {
                return Err(VkFftError::ShaderCompilation(
                    "AMD COMGR produced an empty executable".to_owned(),
                ));
            }
            let mut output = vec![0u8; bytes];
            check_comgr(
                unsafe {
                    (self.api.get_data)(
                        executable_data,
                        &mut bytes,
                        output.as_mut_ptr().cast::<c_void>(),
                    )
                },
                "read executable data",
            )?;
            output.truncate(bytes);
            Ok(output)
        })();

        for data in data_objects.into_iter().rev() {
            unsafe {
                (self.api.release_data)(data);
            }
        }
        for set in data_sets.into_iter().rev() {
            unsafe {
                (self.api.destroy_data_set)(set);
            }
        }
        if let Some(info) = action_info {
            unsafe {
                (self.api.destroy_action_info)(info);
            }
        }
        result
    }

    fn require_single_kind(
        api: &AmdComgrApi,
        set: AmdComgrHandle,
        kind: c_int,
        label: &str,
    ) -> Result<()> {
        let mut count = 0usize;
        check_comgr(
            unsafe { (api.action_data_count)(set, kind, &mut count) },
            &format!("count {label} outputs"),
        )?;
        if count != 1 {
            return Err(VkFftError::ShaderCompilation(format!(
                "AMD COMGR produced {count} {label} object(s), expected exactly one"
            )));
        }
        Ok(())
    }
}

enum HipExecutionCompiler {
    Hiprtc(HiprtcApi),
    #[cfg(all(windows, feature = "opencl-runtime"))]
    AmdComgrNative(AmdComgrCompiler),
}

impl HipExecutionCompiler {
    fn load(_hip_device_count: usize, _hip_device_index: usize) -> Result<Self> {
        match HiprtcApi::load() {
            Ok(hiprtc) => Ok(Self::Hiprtc(hiprtc)),
            Err(hiprtc_error) => {
                #[cfg(all(windows, feature = "opencl-runtime"))]
                {
                    if let Ok(isa_name) =
                        amd_comgr_isa_for_hip_device(_hip_device_count, _hip_device_index)
                    {
                        if let Ok(compiler) = AmdComgrCompiler::load(&isa_name) {
                            return Ok(Self::AmdComgrNative(compiler));
                        }
                    }
                }
                Err(unavailable(
                    Backend::Hip,
                    format!(
                        "HIPRTC is unavailable ({hiprtc_error}); the AMD COMGR fallback requires Windows, opencl-runtime, amd_comgr.dll, and an AMD OpenCL device set whose count/order matches the HIP device set"
                    ),
                ))
            }
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Hiprtc(_) => "HIPRTC",
            #[cfg(all(windows, feature = "opencl-runtime"))]
            Self::AmdComgrNative(_) => "AMD COMGR native",
        }
    }

    fn hiprtc(&self) -> Option<&HiprtcApi> {
        match self {
            Self::Hiprtc(hiprtc) => Some(hiprtc),
            #[cfg(all(windows, feature = "opencl-runtime"))]
            Self::AmdComgrNative(_) => None,
        }
    }

    fn requires_shared_subgroup_fallback(&self) -> bool {
        match self {
            Self::Hiprtc(_) => false,
            #[cfg(all(windows, feature = "opencl-runtime"))]
            Self::AmdComgrNative(_) => true,
        }
    }
}

#[cfg(any(test, all(windows, feature = "opencl-runtime")))]
fn matched_amd_opencl_compiler_ordinal(
    hip_device_count: usize,
    hip_device_index: usize,
    amd_opencl_device_count: usize,
) -> Result<usize> {
    if hip_device_count == 0 || hip_device_index >= hip_device_count {
        return Err(unavailable(
            Backend::Hip,
            format!(
                "AMD COMGR fallback received HIP device index {hip_device_index} for {hip_device_count} HIP device(s)"
            ),
        ));
    }
    if amd_opencl_device_count != hip_device_count {
        return Err(unavailable(
            Backend::Hip,
            format!(
                "AMD COMGR fallback requires matching HIP/OpenCL AMD device counts, got {hip_device_count} HIP and {amd_opencl_device_count} AMD OpenCL device(s)"
            ),
        ));
    }
    Ok(hip_device_index)
}

#[cfg(any(test, all(windows, feature = "opencl-runtime")))]
fn amd_comgr_isa_from_device_name(device_name: &str) -> Result<String> {
    let architecture = device_name.trim();
    if !architecture.starts_with("gfx")
        || architecture.len() <= 3
        || !architecture
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'+' | b'-'))
    {
        return Err(unavailable(
            Backend::Hip,
            format!(
                "AMD COMGR fallback requires an OpenCL device name containing a gfx architecture, got {device_name:?}"
            ),
        ));
    }
    Ok(format!("amdgcn-amd-amdhsa--{architecture}"))
}

#[cfg(all(windows, feature = "opencl-runtime"))]
fn amd_comgr_isa_for_hip_device(
    hip_device_count: usize,
    hip_device_index: usize,
) -> Result<String> {
    let availability = OpenClExecutionContext::probe();
    let mut amd_contexts = Vec::new();
    for device_index in 0..availability.device_count {
        let Ok(context) = OpenClExecutionContext::new(device_index) else {
            continue;
        };
        if context.device_profile().vendor == GpuVendor::Amd {
            amd_contexts.push(context);
        }
    }
    let compiler_ordinal = matched_amd_opencl_compiler_ordinal(
        hip_device_count,
        hip_device_index,
        amd_contexts.len(),
    )?;
    let context = amd_contexts.get(compiler_ordinal).ok_or_else(|| {
        unavailable(
            Backend::Hip,
            "AMD COMGR fallback compiler ordinal disappeared after device enumeration",
        )
    })?;
    amd_comgr_isa_from_device_name(context.device_name())
}

fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            unavailable(
                Backend::Hip,
                format!(
                    "missing symbol {}: {error}",
                    String::from_utf8_lossy(name).trim_end_matches('\0')
                ),
            )
        })
}

fn load_optional_symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    unsafe { library.get::<T>(name) }.ok().map(|symbol| *symbol)
}

fn check_hip(code: c_int, operation: &'static str) -> Result<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(runtime_error(
            Backend::Hip,
            format!("{operation} returned hipError_t {code}"),
        ))
    }
}

fn check_hiprtc(code: c_int, operation: &'static str) -> Result<()> {
    if code == HIPRTC_SUCCESS {
        Ok(())
    } else {
        Err(runtime_error(
            Backend::Hip,
            format!("{operation} returned hiprtcResult {code}"),
        ))
    }
}

struct HipDeviceMemory<'a> {
    api: &'a HipRuntimeApi,
    ptr: *mut c_void,
    bytes: usize,
    owned: bool,
}

impl HipDeviceMemory<'_> {
    fn relinquish(&mut self) -> *mut c_void {
        self.owned = false;
        self.ptr
    }
}

impl Drop for HipDeviceMemory<'_> {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                (self.api.free)(self.ptr);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HipModuleCacheKey {
    source: String,
    entry_point: String,
    compile_mode: HiprtcCompileMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HipCachedModule {
    module: HipModule,
    function: HipFunction,
}

#[derive(Debug)]
struct HipModuleCache {
    entries: HashMap<HipModuleCacheKey, HipCachedModule>,
    max_entries: usize,
}

impl HipModuleCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, key: &HipModuleCacheKey) -> Option<HipCachedModule> {
        self.entries.get(key).copied()
    }

    fn insert_if_capacity(&mut self, key: HipModuleCacheKey, value: HipCachedModule) -> bool {
        if self.entries.len() >= self.max_entries || self.entries.contains_key(&key) {
            return false;
        }
        self.entries.insert(key, value);
        true
    }

    fn drain_modules(&mut self) -> impl Iterator<Item = HipModule> + '_ {
        self.entries.drain().map(|(_, cached)| cached.module)
    }
}

struct HipLoadedModule<'a> {
    api: &'a HipRuntimeApi,
    module: HipModule,
    function: HipFunction,
    owned: bool,
}

impl Drop for HipLoadedModule<'_> {
    fn drop(&mut self) {
        if self.owned && !self.module.is_null() {
            unsafe {
                (self.api.module_unload)(self.module);
            }
        }
    }
}

struct HipOwnedStream<'a> {
    api: &'a HipRuntimeApi,
    stream: HipStream,
    owned: bool,
}

impl HipOwnedStream<'_> {
    fn synchronize(&self) -> Result<()> {
        check_hip(
            unsafe { (self.api.stream_synchronize)(self.stream) },
            "hipStreamSynchronize",
        )
    }

    fn relinquish(&mut self) -> HipStream {
        self.owned = false;
        self.stream
    }
}

impl Drop for HipOwnedStream<'_> {
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

struct HipPendingProgram<'a> {
    stream: HipOwnedStream<'a>,
    context: &'a HipExecutionContext,
    prepared: PreparedProgramStorage,
    allocations: Vec<HipDeviceMemory<'a>>,
    pending_luts: Vec<(usize, Vec<u8>)>,
    _modules: Vec<HipLoadedModule<'a>>,
    completed: bool,
}

impl HipPendingProgram<'_> {
    fn finish(&mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        check_hip(
            unsafe { (self.context.runtime.set_device)(self.context.device_index as c_int) },
            "hipSetDevice",
        )?;
        self.stream.synchronize()?;
        let output = self
            .allocations
            .get(self.prepared.output_allocation.0)
            .ok_or(VkFftError::InvalidKernelIr(
                "HIP program is missing its output allocation",
            ))?;
        let output_bytes = self.prepared.output_bytes_mut()?;
        check_hip(
            unsafe {
                (self.context.runtime.memcpy)(
                    output_bytes.as_mut_ptr().cast(),
                    output.ptr,
                    output_bytes.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy(DtoH)",
        )?;
        for (index, key) in self.pending_luts.drain(..) {
            let allocation = self
                .allocations
                .get_mut(index)
                .ok_or(VkFftError::InvalidKernelIr(
                    "pending HIP LUT references a missing allocation",
                ))?;
            let ptr = allocation.relinquish();
            let mut cache = self
                .context
                .lut_cache
                .lock()
                .map_err(|_| runtime_error(Backend::Hip, "HIP LUT cache lock is poisoned"))?;
            if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
                entry.insert(ptr);
            } else {
                check_hip(
                    unsafe { (self.context.runtime.free)(ptr) },
                    "hipFree(duplicate LUT)",
                )?;
            }
        }
        {
            let mut pool = self.context.transient_buffer_pool.lock().map_err(|_| {
                runtime_error(Backend::Hip, "HIP transient buffer pool lock is poisoned")
            })?;
            for (allocation_plan, allocation) in self
                .prepared
                .memory_plan
                .allocations
                .iter()
                .zip(&mut self.allocations)
            {
                if allocation_plan.kind != ProgramAllocationKind::LookupTable {
                    let bytes = allocation.bytes;
                    let ptr = allocation.relinquish();
                    pool.entry(bytes).or_default().push(ptr);
                }
            }
        }
        {
            let stream = self.stream.relinquish();
            self.context
                .stream_pool
                .lock()
                .map_err(|_| runtime_error(Backend::Hip, "HIP stream pool lock is poisoned"))?
                .push(stream);
        }
        self.completed = true;
        Ok(())
    }
}

impl Drop for HipPendingProgram<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.stream.synchronize();
        }
        self.context
            .active_submissions
            .fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct HipProgramTicket32<'a> {
    pending: HipPendingProgram<'a>,
}

impl HipProgramTicket32<'_> {
    pub fn wait(mut self) -> Result<Vec<Complex32>> {
        self.pending.finish()?;
        self.pending.prepared.output_complex32()
    }
}

pub struct HipProgramTicket64<'a> {
    pending: HipPendingProgram<'a>,
}

impl HipProgramTicket64<'_> {
    pub fn wait(mut self) -> Result<Vec<Complex64>> {
        self.pending.finish()?;
        self.pending.prepared.output_complex64()
    }
}

pub struct HipExecutionContext {
    runtime: HipRuntimeApi,
    compiler: HipExecutionCompiler,
    device_index: usize,
    device_name: String,
    runtime_version: c_int,
    profile: DeviceProfile,
    hiprtc_code_cache: Mutex<HiprtcCodeCache>,
    module_cache: Mutex<HipModuleCache>,
    lut_cache: Mutex<HashMap<Vec<u8>, *mut c_void>>,
    transient_buffer_pool: Mutex<HashMap<usize, Vec<*mut c_void>>>,
    stream_pool: Mutex<Vec<HipStream>>,
    active_submissions: AtomicUsize,
}

impl HipExecutionContext {
    pub fn probe() -> NativeRuntimeAvailability {
        HipRuntimeAdapter::probe()
    }

    pub fn new(device_index: usize) -> Result<Self> {
        Self::new_internal(device_index)
    }

    pub fn new_with_hiprtc_code_archive(device_index: usize, encoded: &[u8]) -> Result<Self> {
        let context = Self::new_internal(device_index)?;
        context.restore_hiprtc_code_archive(encoded)?;
        Ok(context)
    }

    fn new_internal(device_index: usize) -> Result<Self> {
        let runtime = HipRuntimeApi::load()?;
        check_hip(unsafe { (runtime.init)(0) }, "hipInit")?;
        let mut count = 0;
        check_hip(
            unsafe { (runtime.get_device_count)(&mut count) },
            "hipGetDeviceCount",
        )?;
        if device_index >= count.max(0) as usize {
            return Err(unavailable(
                Backend::Hip,
                format!(
                    "requested device {device_index}, but only {} HIP device(s) are available",
                    count.max(0)
                ),
            ));
        }
        let compiler = HipExecutionCompiler::load(count.max(0) as usize, device_index)?;
        check_hip(
            unsafe { (runtime.set_device)(device_index as c_int) },
            "hipSetDevice",
        )?;
        let mut name = [0i8; 256];
        check_hip(
            unsafe {
                (runtime.device_get_name)(
                    name.as_mut_ptr(),
                    name.len() as c_int,
                    device_index as c_int,
                )
            },
            "hipDeviceGetName",
        )?;
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let runtime_version = runtime.runtime_version()?;
        let mut profile = hip_profile_from_attributes(runtime.device_attributes(device_index)?)?;
        if compiler.requires_shared_subgroup_fallback() {
            profile.subgroup.shuffle_supported = false;
            profile.subgroup.shuffle_relative_supported = false;
        }
        Ok(Self {
            runtime,
            compiler,
            device_index,
            device_name,
            runtime_version,
            profile,
            hiprtc_code_cache: Mutex::new(HiprtcCodeCache::new(
                HIPRTC_CACHE_MAX_ENTRIES,
                HIPRTC_CACHE_MAX_BYTES,
            )),
            module_cache: Mutex::new(HipModuleCache::new(HIP_MODULE_CACHE_MAX_ENTRIES)),
            lut_cache: Mutex::new(HashMap::new()),
            transient_buffer_pool: Mutex::new(HashMap::new()),
            stream_pool: Mutex::new(Vec::new()),
            active_submissions: AtomicUsize::new(0),
        })
    }

    pub const fn device_profile(&self) -> DeviceProfile {
        self.profile
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub const fn device_index(&self) -> usize {
        self.device_index
    }

    pub fn hiprtc_code_archive_identity(&self) -> HiprtcCodeArchiveIdentity {
        HiprtcCodeArchiveIdentity {
            runtime_version: self.runtime_version,
            device_name: self.device_name.clone(),
            warp_size: self.profile.subgroup.size,
        }
    }

    pub fn hiprtc_code_archive_data(&self) -> Result<Vec<u8>> {
        if self.compiler.hiprtc().is_none() {
            return Err(unavailable(
                Backend::Hip,
                "HIPRTC code archives are unavailable with the AMD COMGR native compiler fallback",
            ));
        }
        let cache = self
            .hiprtc_code_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIPRTC code cache lock is poisoned"))?;
        let entries = cache
            .entries
            .iter()
            .map(|(key, code)| (key.clone(), code.clone()))
            .collect::<Vec<_>>();
        Ok(HiprtcCodeArchive {
            identity: self.hiprtc_code_archive_identity(),
            entries,
        }
        .encode())
    }

    pub fn restore_hiprtc_code_archive(&self, encoded: &[u8]) -> Result<()> {
        if self.compiler.hiprtc().is_none() {
            return Err(unavailable(
                Backend::Hip,
                "HIPRTC code archives cannot be restored into the AMD COMGR native compiler fallback",
            ));
        }
        let archive = HiprtcCodeArchive::decode(encoded)?;
        let expected = self.hiprtc_code_archive_identity();
        if archive.identity != expected {
            return Err(runtime_error(
                Backend::Hip,
                format!(
                    "HIPRTC code archive targets {:?}, but the selected runtime/device reports {:?}",
                    archive.identity, expected
                ),
            ));
        }
        let mut cache = self
            .hiprtc_code_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIPRTC code cache lock is poisoned"))?;
        for (key, code) in archive.entries {
            if let Some(existing) = cache.entries.get(&key) {
                if existing != &code {
                    return Err(runtime_error(
                        Backend::Hip,
                        "HIPRTC code archive conflicts with an existing source/mode entry",
                    ));
                }
                continue;
            }
            cache.insert(key, &code);
        }
        Ok(())
    }

    pub fn cached_module_count(&self) -> Result<usize> {
        self.module_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| runtime_error(Backend::Hip, "HIP module cache lock is poisoned"))
    }

    pub fn cached_lut_count(&self) -> Result<usize> {
        self.lut_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| runtime_error(Backend::Hip, "HIP LUT cache lock is poisoned"))
    }

    pub fn cached_transient_buffer_count(&self) -> Result<usize> {
        self.transient_buffer_pool
            .lock()
            .map(|pool| pool.values().map(Vec::len).sum())
            .map_err(|_| runtime_error(Backend::Hip, "HIP transient buffer pool lock is poisoned"))
    }

    pub fn pooled_stream_count(&self) -> Result<usize> {
        self.stream_pool
            .lock()
            .map(|pool| pool.len())
            .map_err(|_| runtime_error(Backend::Hip, "HIP stream pool lock is poisoned"))
    }

    pub fn cached_hiprtc_code_count(&self) -> Result<usize> {
        self.hiprtc_code_cache
            .lock()
            .map(|cache| cache.entries.len())
            .map_err(|_| runtime_error(Backend::Hip, "HIPRTC code cache lock is poisoned"))
    }

    fn compiled_function_resource_metrics(
        &self,
        function: HipFunction,
    ) -> Result<Option<NativeCompiledResourceMetrics>> {
        let Some(func_get_attribute) = self.runtime.func_get_attribute else {
            return Ok(None);
        };
        let attribute = |attribute: c_int, field: &'static str| -> Result<usize> {
            let mut value = 0;
            check_hip(
                unsafe { func_get_attribute(&mut value, attribute, function) },
                "hipFuncGetAttribute",
            )?;
            usize::try_from(value).map_err(|_| VkFftError::ValueOutOfRange { field })
        };
        Ok(Some(NativeCompiledResourceMetrics::Hip {
            registers_per_thread: attribute(
                HIP_FUNC_ATTRIBUTE_NUM_REGS,
                "HIP function register count",
            )?,
            static_shared_memory_bytes_per_block: attribute(
                HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                "HIP function static shared-memory bytes",
            )?,
            local_memory_bytes_per_thread: attribute(
                HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                "HIP function local-memory bytes",
            )?,
            max_threads_per_block: attribute(
                HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                "HIP function max threads per block",
            )?,
        }))
    }

    fn compile_shader_source(
        &self,
        shader: &NativeShaderSource,
        compile_mode: HiprtcCompileMode,
    ) -> Result<Vec<u8>> {
        match &self.compiler {
            HipExecutionCompiler::Hiprtc(hiprtc) => {
                let key = HiprtcCodeCacheKey::new(&shader.source, compile_mode);
                if let Some(code) = self
                    .hiprtc_code_cache
                    .lock()
                    .map_err(|_| runtime_error(Backend::Hip, "HIPRTC code cache lock is poisoned"))?
                    .get(&key)
                {
                    return Ok(code);
                }
                let code = hiprtc.compile_code(&shader.source, compile_mode)?;
                self.hiprtc_code_cache
                    .lock()
                    .map_err(|_| runtime_error(Backend::Hip, "HIPRTC code cache lock is poisoned"))?
                    .insert(key, &code);
                Ok(code)
            }
            #[cfg(all(windows, feature = "opencl-runtime"))]
            HipExecutionCompiler::AmdComgrNative(compiler) => {
                let fallback = shader.compiler_fallback_source.as_deref().ok_or(
                    VkFftError::InvalidKernelIr(
                        "HIP AMD COMGR fallback requires a paired OpenCL compiler source",
                    ),
                )?;
                compiler.compile_opencl_executable(fallback)
            }
        }
    }

    pub fn submit_program_complex32<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<HipProgramTicket32<'a>> {
        source.validate()?;
        if source.backend != Backend::Hip {
            return Err(VkFftError::InvalidKernelIr(
                "HIP F32-storage execution requires a HIP native program",
            ));
        }
        if source.program.scalar == ScalarType::F64 && !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "HIP runtime",
                precision: "f64 compute with f32 storage",
            });
        }
        let prepared = prepare_program_complex32(Backend::Hip, &source.program, input)?;
        Ok(HipProgramTicket32 {
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

    pub fn submit_program_complex64<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<HipProgramTicket64<'a>> {
        source.validate()?;
        if source.backend != Backend::Hip || source.program.scalar != ScalarType::F64 {
            return Err(VkFftError::InvalidKernelIr(
                "HIP F64 execution requires a HIP/F64 native program",
            ));
        }
        let prepared = prepare_program_complex64(Backend::Hip, &source.program, input)?;
        Ok(HipProgramTicket64 {
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

    crate::backend::native_runtime::impl_native_double_double_runtime_facade!(Backend::Hip, "HIP");
    crate::backend::native_runtime::impl_native_transform_convenience_facade!(
        Backend::Hip,
        HipProgramTicket32,
        HipProgramTicket64
    );

    pub fn execute_transform_f32(
        &self,
        ir: &TransformIr,
        input: NativeTransformInput32<'_>,
    ) -> Result<NativeTransformOutput32> {
        self.submit_transform_f32(ir, input)?.wait()
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
    ) -> Result<HipPendingProgram<'a>> {
        check_hip(
            unsafe { (self.runtime.set_device)(self.device_index as c_int) },
            "hipSetDevice",
        )?;
        let stream = self.take_stream()?;
        let mut pending_luts = Vec::new();
        let mut allocations = Vec::with_capacity(prepared.allocations.len());
        for (index, (allocation_plan, prepared_allocation)) in prepared
            .memory_plan
            .allocations
            .iter()
            .zip(&prepared.allocations)
            .enumerate()
        {
            let byte_len = prepared_allocation.byte_len;
            let host_bytes = prepared_allocation.host_bytes.as_deref();
            if allocation_plan.kind == ProgramAllocationKind::LookupTable {
                let bytes = host_bytes.ok_or(VkFftError::InvalidKernelIr(
                    "HIP lookup-table allocation is missing initialization bytes",
                ))?;
                let cached = self
                    .lut_cache
                    .lock()
                    .map_err(|_| runtime_error(Backend::Hip, "HIP LUT cache lock is poisoned"))?
                    .get(bytes)
                    .copied();
                if let Some(ptr) = cached {
                    allocations.push(HipDeviceMemory {
                        api: &self.runtime,
                        ptr,
                        bytes: byte_len,
                        owned: false,
                    });
                } else {
                    allocations.push(self.take_transient_and_upload(bytes)?);
                    pending_luts.push((index, bytes.to_vec()));
                }
            } else if let Some(bytes) = host_bytes {
                allocations.push(self.take_transient_and_upload(bytes)?);
            } else {
                allocations.push(self.take_transient(byte_len)?);
            }
        }

        let mut modules = Vec::with_capacity(source.shaders.len());
        for shader in &source.shaders {
            shader.validate()?;
            modules.push(self.load_module_function(shader)?);
        }

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
                            "HIP pass references a missing device allocation",
                        ))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut argument_pointers = argument_values
                .iter_mut()
                .map(|value| (value as *mut *mut c_void).cast::<c_void>())
                .collect::<Vec<_>>();
            check_hip(
                unsafe {
                    (self.runtime.module_launch_kernel)(
                        module.function,
                        shader.dispatch.x,
                        shader.dispatch.y,
                        shader.dispatch.z,
                        shader.workgroup_size.x,
                        shader.workgroup_size.y,
                        shader.workgroup_size.z,
                        HIP_DYNAMIC_SHARED_MEMORY_BYTES,
                        stream.stream,
                        argument_pointers.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "hipModuleLaunchKernel",
            )?;
        }
        self.active_submissions.fetch_add(1, Ordering::AcqRel);
        Ok(HipPendingProgram {
            stream,
            context: self,
            prepared,
            allocations,
            pending_luts,
            _modules: modules,
            completed: false,
        })
    }

    fn load_module_function<'a>(
        &'a self,
        shader: &NativeShaderSource,
    ) -> Result<HipLoadedModule<'a>> {
        let compile_mode = HiprtcCompileMode::for_scalar(shader.scalar);
        let key = HipModuleCacheKey {
            source: shader.source.clone(),
            entry_point: shader.entry_point.to_owned(),
            compile_mode,
        };
        if let Some(cached) = self
            .module_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP module cache lock is poisoned"))?
            .get(&key)
        {
            return Ok(HipLoadedModule {
                api: &self.runtime,
                module: cached.module,
                function: cached.function,
                owned: false,
            });
        }

        let code = self.compile_shader_source(shader, compile_mode)?;
        let mut module = ptr::null_mut();
        check_hip(
            unsafe { (self.runtime.module_load_data)(&mut module, code.as_ptr().cast()) },
            "hipModuleLoadData",
        )?;
        if module.is_null() {
            return Err(runtime_error(
                Backend::Hip,
                "hipModuleLoadData succeeded with a null module handle",
            ));
        }
        let mut loaded = HipLoadedModule {
            api: &self.runtime,
            module,
            function: ptr::null_mut(),
            owned: true,
        };
        let entry = CString::new(shader.entry_point).map_err(|_| {
            VkFftError::ShaderCompilation(
                "HIP kernel entry point contains an interior NUL byte".to_owned(),
            )
        })?;
        check_hip(
            unsafe {
                (self.runtime.module_get_function)(
                    &mut loaded.function,
                    loaded.module,
                    entry.as_ptr(),
                )
            },
            "hipModuleGetFunction",
        )?;
        if loaded.function.is_null() {
            return Err(runtime_error(
                Backend::Hip,
                "hipModuleGetFunction succeeded with a null function handle",
            ));
        }

        let cached_value = HipCachedModule {
            module: loaded.module,
            function: loaded.function,
        };
        let mut cache = self
            .module_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP module cache lock is poisoned"))?;
        if let Some(existing) = cache.get(&key) {
            drop(cache);
            drop(loaded);
            return Ok(HipLoadedModule {
                api: &self.runtime,
                module: existing.module,
                function: existing.function,
                owned: false,
            });
        }
        if cache.insert_if_capacity(key, cached_value) {
            loaded.owned = false;
        }
        Ok(loaded)
    }

    fn take_stream(&self) -> Result<HipOwnedStream<'_>> {
        if let Some(stream) = self
            .stream_pool
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP stream pool lock is poisoned"))?
            .pop()
        {
            return Ok(HipOwnedStream {
                api: &self.runtime,
                stream,
                owned: true,
            });
        }
        let mut stream = ptr::null_mut();
        check_hip(
            unsafe { (self.runtime.stream_create)(&mut stream) },
            "hipStreamCreate",
        )?;
        if stream.is_null() {
            return Err(runtime_error(
                Backend::Hip,
                "hipStreamCreate succeeded with a null stream",
            ));
        }
        Ok(HipOwnedStream {
            api: &self.runtime,
            stream,
            owned: true,
        })
    }

    fn take_transient<'a>(&'a self, bytes: usize) -> Result<HipDeviceMemory<'a>> {
        let pooled = self
            .transient_buffer_pool
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP transient buffer pool lock is poisoned"))?
            .get_mut(&bytes)
            .and_then(Vec::pop);
        let ptr = if let Some(ptr) = pooled {
            ptr
        } else {
            let mut ptr = ptr::null_mut();
            check_hip(
                unsafe { (self.runtime.malloc)(&mut ptr, bytes) },
                "hipMalloc",
            )?;
            if ptr.is_null() {
                return Err(runtime_error(
                    Backend::Hip,
                    "hipMalloc succeeded with a null pointer",
                ));
            }
            ptr
        };
        Ok(HipDeviceMemory {
            api: &self.runtime,
            ptr,
            bytes,
            owned: true,
        })
    }

    fn take_transient_and_upload<'a>(&'a self, bytes: &[u8]) -> Result<HipDeviceMemory<'a>> {
        let allocation = self.take_transient(bytes.len())?;
        check_hip(
            unsafe {
                (self.runtime.memcpy)(
                    allocation.ptr,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy(HtoD)",
        )?;
        Ok(allocation)
    }

    pub fn clear_runtime_caches(&self) -> Result<()> {
        let active = self.active_submissions.load(Ordering::Acquire);
        if active != 0 {
            return Err(runtime_error(
                Backend::Hip,
                format!(
                    "cannot clear HIP runtime caches while {active} submission(s) are in flight"
                ),
            ));
        }
        check_hip(
            unsafe { (self.runtime.set_device)(self.device_index as c_int) },
            "hipSetDevice",
        )?;
        check_hip(
            unsafe { (self.runtime.device_synchronize)() },
            "hipDeviceSynchronize",
        )?;
        let mut buffers = self.transient_buffer_pool.lock().map_err(|_| {
            runtime_error(Backend::Hip, "HIP transient buffer pool lock is poisoned")
        })?;
        for ptr in buffers.drain().flat_map(|(_, buffers)| buffers) {
            check_hip(unsafe { (self.runtime.free)(ptr) }, "hipFree")?;
        }
        drop(buffers);
        let mut modules = self
            .module_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP module cache lock is poisoned"))?;
        for module in modules.drain_modules() {
            check_hip(
                unsafe { (self.runtime.module_unload)(module) },
                "hipModuleUnload(cache)",
            )?;
        }
        drop(modules);
        let mut luts = self
            .lut_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP LUT cache lock is poisoned"))?;
        for ptr in luts.drain().map(|(_, ptr)| ptr) {
            check_hip(unsafe { (self.runtime.free)(ptr) }, "hipFree(LUT cache)")?;
        }
        drop(luts);
        let mut streams = self
            .stream_pool
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIP stream pool lock is poisoned"))?;
        for stream in streams.drain(..) {
            check_hip(
                unsafe { (self.runtime.stream_destroy)(stream) },
                "hipStreamDestroy",
            )?;
        }
        self.hiprtc_code_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Hip, "HIPRTC code cache lock is poisoned"))?
            .clear();
        Ok(())
    }
}

impl crate::backend::native_runtime::NativeProgramTicket32 for HipProgramTicket32<'_> {
    fn wait(self) -> Result<Vec<Complex32>> {
        HipProgramTicket32::wait(self)
    }
}

impl crate::backend::native_runtime::NativeProgramTicket64 for HipProgramTicket64<'_> {
    fn wait(self) -> Result<Vec<Complex64>> {
        HipProgramTicket64::wait(self)
    }
}

impl crate::backend::native_runtime::NativeAsyncRuntime for HipExecutionContext {
    type Ticket32<'a> = HipProgramTicket32<'a>;
    type Ticket64<'a> = HipProgramTicket64<'a>;

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

impl crate::backend::native_runtime::NativeRuntime for HipExecutionContext {
    fn backend(&self) -> Backend {
        Backend::Hip
    }

    fn device_profile(&self) -> DeviceProfile {
        HipExecutionContext::device_profile(self)
    }

    fn device_name(&self) -> &str {
        HipExecutionContext::device_name(self)
    }

    fn compiled_pass_resource_reports(
        &self,
        source: &NativeProgramSource,
    ) -> Result<Vec<NativeCompiledPassResourceReport>> {
        source.validate()?;
        if source.backend != Backend::Hip {
            return Err(VkFftError::InvalidKernelIr(
                "HIP compiled-resource reporting requires a HIP native program",
            ));
        }
        if self.runtime.func_get_attribute.is_none() {
            return Ok(Vec::new());
        }
        check_hip(
            unsafe { (self.runtime.set_device)(self.device_index as c_int) },
            "hipSetDevice",
        )?;
        let mut reports = Vec::with_capacity(source.shaders.len());
        for (pass, shader) in source.program.passes.iter().zip(&source.shaders) {
            let loaded = self.load_module_function(shader)?;
            let metrics = self
                .compiled_function_resource_metrics(loaded.function)?
                .ok_or(VkFftError::InvalidKernelIr(
                    "HIP function attributes disappeared after capability detection",
                ))?;
            reports.push(NativeCompiledPassResourceReport {
                pass_name: pass.name.clone(),
                metrics,
            });
        }
        Ok(reports)
    }

    fn execute_program_complex32(
        &self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        HipExecutionContext::execute_program_complex32(self, source, input)
    }

    fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        HipExecutionContext::execute_program_complex64(self, source, input)
    }
}

impl Drop for HipExecutionContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.runtime.set_device)(self.device_index as c_int);
            let _ = (self.runtime.device_synchronize)();
        }
        let buffers = match self.transient_buffer_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for ptr in buffers.drain().flat_map(|(_, buffers)| buffers) {
            unsafe {
                (self.runtime.free)(ptr);
            }
        }
        let modules = match self.module_cache.get_mut() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        for module in modules.drain_modules() {
            unsafe {
                (self.runtime.module_unload)(module);
            }
        }
        let luts = match self.lut_cache.get_mut() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        for ptr in luts.drain().map(|(_, ptr)| ptr) {
            unsafe {
                (self.runtime.free)(ptr);
            }
        }
        let streams = match self.stream_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for stream in streams.drain(..) {
            unsafe {
                (self.runtime.stream_destroy)(stream);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HipRuntimeAdapter {
    availability: NativeRuntimeAvailability,
}

impl HipRuntimeAdapter {
    pub fn probe() -> NativeRuntimeAvailability {
        let runtime = load_first_library(Backend::Hip, HIP_RUNTIME_LIBRARY_CANDIDATES);
        let Ok((library, library_name)) = runtime else {
            return NativeRuntimeAvailability {
                backend: Backend::Hip,
                loader_available: false,
                compiler_available: false,
                device_count: 0,
                detail: "HIP runtime library was not found".to_owned(),
            };
        };
        match hip_device_count(&library) {
            Ok(device_count) => {
                let compiler = HipExecutionCompiler::load(device_count, 0);
                let compiler_available = compiler.is_ok();
                let compiler_label = compiler
                    .as_ref()
                    .map(|compiler| compiler.label())
                    .unwrap_or("unavailable");
                NativeRuntimeAvailability {
                    backend: Backend::Hip,
                    loader_available: true,
                    compiler_available,
                    device_count,
                    detail: format!(
                        "{library_name} reports {device_count} HIP device(s); execution compiler={compiler_label}"
                    ),
                }
            }
            Err(error) => NativeRuntimeAvailability {
                backend: Backend::Hip,
                loader_available: true,
                compiler_available: false,
                device_count: 0,
                detail: error.to_string(),
            },
        }
    }

    pub fn new() -> Result<Self> {
        let availability = Self::probe();
        if !availability.available() {
            return Err(unavailable(Backend::Hip, availability.detail.clone()));
        }
        Ok(Self { availability })
    }

    pub const fn availability(&self) -> &NativeRuntimeAvailability {
        &self.availability
    }
}

fn hip_device_count(library: &Library) -> Result<usize> {
    // SAFETY: signatures follow the public HIP Runtime API; the library lives through
    // both calls and symbols are not retained afterward.
    unsafe {
        let init = library
            .get::<HipInit>(b"hipInit\0")
            .map_err(|error| unavailable(Backend::Hip, format!("missing hipInit: {error}")))?;
        let get_device_count = library
            .get::<HipGetDeviceCount>(b"hipGetDeviceCount\0")
            .map_err(|error| {
                unavailable(Backend::Hip, format!("missing hipGetDeviceCount: {error}"))
            })?;
        let init_result = init(0);
        if init_result != HIP_SUCCESS {
            return Err(VkFftError::NativeRuntime {
                backend: "HIP",
                message: format!("hipInit returned {init_result}"),
            });
        }
        let mut count = 0;
        let result = get_device_count(&mut count);
        if result != HIP_SUCCESS {
            return Err(VkFftError::NativeRuntime {
                backend: "HIP",
                message: format!("hipGetDeviceCount returned {result}"),
            });
        }
        Ok(count.max(0) as usize)
    }
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
    use crate::backend::NativeSourceBackend;

    #[test]
    fn hip_library_candidates_cover_rocm5_through_rocm7_sonames() {
        assert!(HIP_RUNTIME_LIBRARY_CANDIDATES.contains(&"amdhip64.dll"));
        for major in [5, 6, 7] {
            assert!(
                HIP_RUNTIME_LIBRARY_CANDIDATES
                    .contains(&format!("libamdhip64.so.{major}").as_str())
            );
            assert!(HIPRTC_LIBRARY_CANDIDATES.contains(&format!("libhiprtc.so.{major}").as_str()));
        }
    }

    #[test]
    fn amd_comgr_fallback_maps_matching_multi_device_sets_by_ordinal() {
        assert_eq!(matched_amd_opencl_compiler_ordinal(1, 0, 1).unwrap(), 0);
        assert_eq!(matched_amd_opencl_compiler_ordinal(2, 0, 2).unwrap(), 0);
        assert_eq!(matched_amd_opencl_compiler_ordinal(2, 1, 2).unwrap(), 1);
        assert!(matched_amd_opencl_compiler_ordinal(2, 0, 1).is_err());
        assert!(matched_amd_opencl_compiler_ordinal(2, 0, 3).is_err());
        assert!(matched_amd_opencl_compiler_ordinal(2, 2, 2).is_err());
        assert!(matched_amd_opencl_compiler_ordinal(0, 0, 0).is_err());
        assert_eq!(
            amd_comgr_isa_from_device_name("gfx803").unwrap(),
            "amdgcn-amd-amdhsa--gfx803"
        );
        assert!(amd_comgr_isa_from_device_name("AMD Radeon RX 640").is_err());
        assert!(amd_comgr_isa_from_device_name("gfx").is_err());
    }

    #[test]
    fn amd_comgr_headerless_opencl_prelude_declares_required_builtins() {
        let f32_source = amd_comgr_opencl_source(
            "float2 value; barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);",
        );
        assert!(f32_source.contains("typedef unsigned int uint;"));
        assert!(f32_source.contains("typedef uint cl_mem_fence_flags;"));
        assert!(f32_source.contains("typedef float float2"));
        assert!(f32_source.contains("#define CLK_LOCAL_MEM_FENCE 1u"));
        assert!(f32_source.contains("#define CLK_GLOBAL_MEM_FENCE 2u"));
        assert!(f32_source.contains("inline float as_float(uint value)"));
        assert!(f32_source.contains("inline uint as_uint(float value)"));
        assert!(f32_source.contains("float sub_group_broadcast(float, uint)"));
        assert!(f32_source.contains("get_sub_group_local_id(void)"));
        assert!(f32_source.contains("get_sub_group_id(void)"));
        assert!(!f32_source.contains("opencl-c.h"));

        let f64_source = amd_comgr_opencl_source("double2 value;");
        assert!(f64_source.contains("cl_khr_fp64 : enable"));
        assert!(f64_source.contains("typedef double double2"));
        assert!(f64_source.contains("typedef double double4"));
        assert!(f64_source.contains("double sub_group_broadcast(double, uint)"));
    }

    #[test]
    fn hip_device_attribute_abi_tracks_rocm5_reorder() {
        assert_eq!(
            hip_device_attribute_abi(40_500_000),
            HIP_DEVICE_ATTRIBUTES_PRE_5
        );
        assert_eq!(HIP_DEVICE_ATTRIBUTES_PRE_5.max_threads_per_block, 0);
        assert_eq!(HIP_DEVICE_ATTRIBUTES_PRE_5.max_block_dim_x, 1);
        assert_eq!(HIP_DEVICE_ATTRIBUTES_PRE_5.max_shared_memory_per_block, 7);
        assert_eq!(HIP_DEVICE_ATTRIBUTES_PRE_5.warp_size, 9);

        for version in [50_000_000, 50_700_000, 60_000_000, 70_000_000] {
            assert_eq!(
                hip_device_attribute_abi(version),
                HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE
            );
        }
        assert_eq!(HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE.max_block_dim_x, 26);
        assert_eq!(
            HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE.max_threads_per_block,
            56
        );
        assert_eq!(
            HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE.max_shared_memory_per_block,
            74
        );
        assert_eq!(HIP_DEVICE_ATTRIBUTES_CUDA_COMPATIBLE.warp_size, 87);
    }

    #[test]
    fn hip_exact_device_attributes_build_wave32_and_wave64_profiles() {
        for warp_size in [32usize, 64usize] {
            let profile = hip_profile_from_attributes(HipDeviceAttributes {
                shared_memory_bytes: 65_536,
                max_threads_per_block: 1_024,
                max_block_dim: [1_024, 1_024, 1_024],
                warp_size,
            })
            .unwrap();
            assert_eq!(profile.backend, Backend::Hip);
            assert_eq!(profile.vendor, GpuVendor::Amd);
            assert_eq!(profile.shared_memory_bytes, 65_536);
            assert_eq!(profile.shared_memory_pow2_bytes, 65_536);
            assert_eq!(profile.max_threads_per_block, 1_024);
            assert_eq!(profile.max_workgroup_size, [1_024, 1_024, 1_024]);
            assert_eq!(profile.subgroup.size, warp_size);
            assert_eq!(profile.subgroup.min_size, warp_size);
            assert_eq!(profile.subgroup.max_size, warp_size);
            assert!(profile.subgroup.supports_full_subgroup_shuffle_compute());
        }
    }

    #[test]
    fn hip_exact_device_attributes_reject_invalid_limits() {
        let error = hip_profile_from_attributes(HipDeviceAttributes {
            shared_memory_bytes: 65_536,
            max_threads_per_block: 256,
            max_block_dim: [256, 256, 256],
            warp_size: 0,
        })
        .unwrap_err();
        assert!(matches!(error, VkFftError::NativeRuntime { .. }));
    }

    #[test]
    fn hip_static_shared_lowering_uses_zero_dynamic_launch_memory() {
        let profile = DeviceProfile::generic(Backend::Hip, GpuVendor::Amd);
        let transform = TransformIr::build(
            crate::FftConfig::new(vec![4096]).with_batch_count(8),
            crate::Direction::Forward,
            profile,
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::Hip)
            .lower_transform(&transform)
            .unwrap();
        let shared_shader = source
            .shaders
            .iter()
            .find(|shader| shader.required_shared_memory_bytes > 0)
            .expect("N4096 HIP Stockham should materialize static shared memory");
        assert!(shared_shader.source.contains("__shared__"));
        assert!(!shared_shader.source.contains("extern __shared__"));
        assert_eq!(HIP_DYNAMIC_SHARED_MEMORY_BYTES, 0);
    }

    #[test]
    fn hiprtc_code_archive_round_trips_modes_and_rejects_incompatible_headers() {
        let archive = HiprtcCodeArchive {
            identity: HiprtcCodeArchiveIdentity {
                runtime_version: 60_000_000,
                device_name: "fake-amd-gpu".to_owned(),
                warp_size: 64,
            },
            entries: vec![
                (
                    HiprtcCodeCacheKey::new("source-b", HiprtcCompileMode::DisableFpContraction),
                    vec![4, 5, 6],
                ),
                (
                    HiprtcCodeCacheKey::new("source-a", HiprtcCompileMode::Default),
                    vec![1, 2, 3],
                ),
            ],
        };
        let encoded = archive.encode();
        let decoded = HiprtcCodeArchive::decode(&encoded).unwrap();
        assert_eq!(decoded.identity, archive.identity);
        assert_eq!(decoded.entry_count(), 2);
        assert_eq!(
            decoded.entries[0],
            (
                HiprtcCodeCacheKey::new("source-a", HiprtcCompileMode::Default),
                vec![1, 2, 3]
            )
        );
        assert_eq!(
            decoded.entries[1],
            (
                HiprtcCodeCacheKey::new("source-b", HiprtcCompileMode::DisableFpContraction,),
                vec![4, 5, 6]
            )
        );
        assert_eq!(decoded.encode(), encoded);

        let mut bad_version = encoded.clone();
        bad_version[8..12].copy_from_slice(&(HIPRTC_CODE_ARCHIVE_VERSION + 1).to_le_bytes());
        assert!(HiprtcCodeArchive::decode(&bad_version).is_err());

        let mut bad_commit = encoded.clone();
        bad_commit[12] ^= 0x01;
        assert!(HiprtcCodeArchive::decode(&bad_commit).is_err());

        let first_mode_offset =
            HIPRTC_CODE_ARCHIVE_HEADER_BYTES + archive.identity.device_name.len() + 16;
        let mut bad_mode = encoded.clone();
        bad_mode[first_mode_offset] = 0xff;
        assert!(HiprtcCodeArchive::decode(&bad_mode).is_err());

        let mut trailing = encoded;
        trailing.push(0);
        assert!(HiprtcCodeArchive::decode(&trailing).is_err());
    }

    #[test]
    fn hiprtc_code_cache_is_bounded_and_separates_compile_modes() {
        let mut cache = HiprtcCodeCache::new(3, 64);
        let default_a = HiprtcCodeCacheKey::new("a", HiprtcCompileMode::Default);
        let dd_a = HiprtcCodeCacheKey::new("a", HiprtcCompileMode::DisableFpContraction);
        let default_b = HiprtcCodeCacheKey::new("b", HiprtcCompileMode::Default);
        let default_c = HiprtcCodeCacheKey::new("c", HiprtcCompileMode::Default);

        cache.insert(default_a.clone(), &[1, 2, 3]);
        cache.insert(dd_a.clone(), &[4, 5, 6]);
        cache.insert(default_b.clone(), &[7, 8, 9]);
        assert_eq!(cache.get(&default_a), Some(vec![1, 2, 3]));
        assert_eq!(cache.get(&dd_a), Some(vec![4, 5, 6]));

        cache.insert(default_c.clone(), &[10, 11, 12]);
        assert!(cache.get(&default_a).is_none());
        assert_eq!(cache.get(&dd_a), Some(vec![4, 5, 6]));
        assert_eq!(cache.get(&default_b), Some(vec![7, 8, 9]));
        assert_eq!(cache.get(&default_c), Some(vec![10, 11, 12]));

        cache.insert(dd_a.clone(), &[9, 8]);
        assert_eq!(cache.get(&dd_a), Some(vec![9, 8]));
        assert!(cache.entries.len() <= 3);
        assert!(cache.total_bytes <= 64);
        cache.clear();
        assert!(cache.entries.is_empty());
        assert_eq!(cache.total_bytes, 0);
    }

    #[test]
    fn hiprtc_double_double_compile_mode_disables_fp_contraction() {
        assert_eq!(
            HiprtcCompileMode::for_scalar(ScalarType::F32),
            HiprtcCompileMode::Default
        );
        assert_eq!(
            HiprtcCompileMode::for_scalar(ScalarType::DoubleDouble),
            HiprtcCompileMode::DisableFpContraction
        );
        assert!(
            !HiprtcCompileMode::Default
                .clang_options()
                .contains(&"-ffp-contract=off")
        );
        assert!(
            HiprtcCompileMode::DisableFpContraction
                .clang_options()
                .contains(&"-ffp-contract=off")
        );
    }

    #[test]
    fn hip_module_cache_is_bounded_and_deduplicates_source_entry() {
        let mut cache = HipModuleCache::new(2);
        let key_a = HipModuleCacheKey {
            source: "source-a".to_owned(),
            entry_point: "kernel-a".to_owned(),
            compile_mode: HiprtcCompileMode::Default,
        };
        let key_b = HipModuleCacheKey {
            source: "source-b".to_owned(),
            entry_point: "kernel-b".to_owned(),
            compile_mode: HiprtcCompileMode::DisableFpContraction,
        };
        let key_c = HipModuleCacheKey {
            source: "source-c".to_owned(),
            entry_point: "kernel-c".to_owned(),
            compile_mode: HiprtcCompileMode::Default,
        };
        let module_a = HipCachedModule {
            module: 1usize as HipModule,
            function: 11usize as HipFunction,
        };
        let module_b = HipCachedModule {
            module: 2usize as HipModule,
            function: 22usize as HipFunction,
        };
        let module_c = HipCachedModule {
            module: 3usize as HipModule,
            function: 33usize as HipFunction,
        };

        assert!(cache.insert_if_capacity(key_a.clone(), module_a));
        assert_eq!(cache.get(&key_a), Some(module_a));
        assert!(!cache.insert_if_capacity(key_a.clone(), module_c));
        assert!(cache.insert_if_capacity(key_b, module_b));
        assert!(!cache.insert_if_capacity(key_c, module_c));
        assert_eq!(cache.len(), 2);

        let mut modules = cache
            .drain_modules()
            .map(|module| module as usize)
            .collect::<Vec<_>>();
        modules.sort_unstable();
        assert_eq!(modules, vec![1, 2]);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn hip_probe_is_fail_soft_without_rocm() {
        let availability = HipRuntimeAdapter::probe();
        assert_eq!(availability.backend, Backend::Hip);
        if let Ok(adapter) = HipRuntimeAdapter::new() {
            assert!(adapter.availability().available());
            let context = HipExecutionContext::new(0).unwrap();
            assert_eq!(context.device_profile().backend, Backend::Hip);
            assert_eq!(context.device_profile().vendor, GpuVendor::Amd);
        }
    }

    #[test]
    fn hip_execution_context_implements_native_async_runtime_contract() {
        fn assert_async_runtime<T: crate::backend::native_runtime::NativeAsyncRuntime>() {}
        fn assert_ticket32<T: crate::backend::native_runtime::NativeProgramTicket32>() {}
        fn assert_ticket64<T: crate::backend::native_runtime::NativeProgramTicket64>() {}

        assert_async_runtime::<HipExecutionContext>();
        assert_ticket32::<HipProgramTicket32<'static>>();
        assert_ticket64::<HipProgramTicket64<'static>>();
        let _ = HipExecutionContext::submit_transform_complex32;
        let _ = HipExecutionContext::execute_transform_complex32;
        let _ = HipExecutionContext::submit_transform_complex64;
        let _ = HipExecutionContext::execute_transform_complex64;
        let _ = HipExecutionContext::submit_transform_f32;
        let _ = HipExecutionContext::submit_transform_f64;
    }

    #[test]
    fn hip_compiled_resource_report_matches_real_kernel_or_skip() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        let transform = TransformIr::build(
            crate::FftConfig::new(vec![96]).with_batch_count(2),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::Hip)
            .lower_transform(&transform)
            .unwrap();
        let reports =
            crate::backend::NativeRuntime::compiled_pass_resource_reports(&context, &source)
                .unwrap();
        if context.runtime.func_get_attribute.is_none() {
            assert!(reports.is_empty());
            return;
        }
        assert_eq!(reports.len(), source.program.passes.len());
        for ((pass, shader), report) in source
            .program
            .passes
            .iter()
            .zip(&source.shaders)
            .zip(&reports)
        {
            assert_eq!(report.pass_name, pass.name);
            let NativeCompiledResourceMetrics::Hip {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread: _,
                max_threads_per_block,
            } = report.metrics
            else {
                panic!("HIP compiled-resource report changed backend metric kind");
            };
            let workgroup_threads = (shader.workgroup_size.x as usize)
                * (shader.workgroup_size.y as usize)
                * (shader.workgroup_size.z as usize);
            assert!(registers_per_thread > 0);
            assert!(max_threads_per_block >= workgroup_threads);
            assert!(
                static_shared_memory_bytes_per_block
                    <= context.device_profile().shared_memory_bytes
            );
        }
    }

    #[test]
    fn hip_double_double_typed_facade_matches_reference_or_skip() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 15usize;
        let full_ir = TransformIr::build(
            crate::FftConfig::new(vec![length]).with_precision(crate::Precision::DoubleDouble),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let full_input = (0..length)
            .map(|index| {
                let x = index as f64;
                crate::ComplexDoubleDouble::new(
                    crate::DoubleDouble::from_parts(
                        (0.17 * x).sin(),
                        (index as f64 + 1.0) * 1.0e-30,
                    ),
                    crate::DoubleDouble::from_parts(
                        (0.11 * x).cos(),
                        -(index as f64 + 1.0) * 7.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let full_expected = full_ir
            .execute_double_double_reference(&full_input)
            .unwrap();
        let full_actual = context
            .execute_transform_double_double(&full_ir, &full_input)
            .unwrap();
        let full_error = full_actual
            .iter()
            .copied()
            .zip(full_expected.iter().copied())
            .map(|(actual, expected)| {
                let re = (actual.re - expected.re).abs();
                let im = (actual.im - expected.im).abs();
                re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            full_error <= 1.0e-20,
            "HIP full-DD N15 mismatch on {}: {full_error:e}",
            context.device_name()
        );

        let f64_ir = TransformIr::build(
            crate::FftConfig::new(vec![length])
                .with_precision(crate::Precision::DoubleDoubleF64Storage),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let f64_input = full_input
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
            "HIP DD/F64 N15 mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn hip_two_high_level_tickets_can_be_in_flight_before_wait_or_skip() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        let build = |length: usize, phase: f32| {
            let ir = TransformIr::build(
                crate::FftConfig::new(vec![length]),
                crate::Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let input = (0..length)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new((phase * x).sin() + x * 0.0001, (0.037 * x).cos())
                })
                .collect::<Vec<_>>();
            (ir, input)
        };
        let (ir_a, input_a) = build(64, 0.021);
        let (ir_b, input_b) = build(96, 0.017);
        let ticket_a = context
            .submit_transform_f32(&ir_a, NativeTransformInput32::Complex(&input_a))
            .unwrap();
        let ticket_b = context
            .submit_transform_f32(&ir_b, NativeTransformInput32::Complex(&input_b))
            .unwrap();
        assert!(context.clear_runtime_caches().is_err());

        for (ir, input, output) in [
            (&ir_b, &input_b, ticket_b.wait().unwrap()),
            (&ir_a, &input_a, ticket_a.wait().unwrap()),
        ] {
            let NativeTransformOutput32::Complex(actual) = output else {
                panic!("async HIP C2C returned real output");
            };
            let expected_input = input
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect::<Vec<_>>();
            let expected = ir.execute_complex_reference(&expected_input).unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| {
                    (actual.re as f64 - expected.re).hypot(actual.im as f64 - expected.im)
                })
                .fold(0.0f64, f64::max);
            assert!(
                error <= 2.0e-3 * input.len() as f64,
                "HIP async N{} error {error:e}",
                input.len()
            );
        }
        assert!(context.pooled_stream_count().unwrap() >= 2);
        assert!(context.cached_transient_buffer_count().unwrap() > 0);
    }

    #[test]
    fn hip_amd_comgr_native_fallback_stockham_matches_reference_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_HIP_AMD_COMGR_FALLBACK").is_some();
        if HiprtcApi::load().is_ok() {
            assert!(
                !require,
                "strict AMD COMGR fallback gate found HIPRTC instead"
            );
            return;
        }
        let availability = HipExecutionContext::probe();
        if !availability.available() || !availability.detail.contains("AMD COMGR native") {
            assert!(
                !require,
                "strict AMD COMGR fallback gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let device_count = if require {
            availability.device_count
        } else {
            availability.device_count.min(1)
        };
        assert!(device_count > 0);
        for device_index in 0..device_count {
            let context = HipExecutionContext::new(device_index).unwrap();
            assert_eq!(context.device_profile().vendor, GpuVendor::Amd);
            assert_eq!(context.compiler.label(), "AMD COMGR native");
            assert!(!context.device_profile().subgroup.shuffle_supported);
            assert!(!context.device_profile().subgroup.shuffle_relative_supported);
            assert_eq!(context.cached_hiprtc_code_count().unwrap(), 0);
            assert!(context.hiprtc_code_archive_data().is_err());

            for (length, phase, imag_phase, linear_term) in [
                (64usize, 0.17f32, 0.11f32, 0.0f32),
                (96usize, 0.017f32, 0.037f32, 0.0001f32),
            ] {
                let ir = TransformIr::build(
                    crate::FftConfig::new(vec![length]),
                    crate::Direction::Forward,
                    context.device_profile(),
                )
                .unwrap();
                let input = (0..length)
                    .map(|index| {
                        let x = index as f32;
                        Complex32::new((phase * x).sin() + x * linear_term, (imag_phase * x).cos())
                    })
                    .collect::<Vec<_>>();
                let actual = context
                    .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
                    .unwrap();
                let NativeTransformOutput32::Complex(actual) = actual else {
                    panic!("HIP AMD COMGR fallback C2C transform returned a real output");
                };
                let oracle_input = input
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected =
                    crate::reference::dft(&oracle_input, crate::Direction::Forward, false);
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| {
                        (actual.re as f64 - expected.re).hypot(actual.im as f64 - expected.im)
                    })
                    .fold(0.0f64, f64::max);
                assert!(
                    error < 2.0e-3 * length as f64,
                    "HIP AMD COMGR fallback N{length} F32 error {error:e} on {}",
                    context.device_name()
                );
            }
            let length = 4096usize;
            let batch_count = 8usize;
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f32;
                    Complex32::new((0.0031 * x).sin() + x * 1.0e-7, (0.0023 * x).cos())
                })
                .collect::<Vec<_>>();
            let forward_ir = TransformIr::build(
                crate::FftConfig::new(vec![length]).with_batch_count(batch_count),
                crate::Direction::Forward,
                context.device_profile(),
            )
            .unwrap();
            let inverse_ir = TransformIr::build(
                crate::FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_inverse_normalization(true),
                crate::Direction::Inverse,
                context.device_profile(),
            )
            .unwrap();
            let NativeTransformOutput32::Complex(spectrum) = context
                .execute_transform_f32(&forward_ir, NativeTransformInput32::Complex(&input))
                .unwrap()
            else {
                panic!("HIP AMD COMGR N4096 forward returned real output");
            };
            let NativeTransformOutput32::Complex(restored) = context
                .execute_transform_f32(&inverse_ir, NativeTransformInput32::Complex(&spectrum))
                .unwrap()
            else {
                panic!("HIP AMD COMGR N4096 inverse returned real output");
            };
            let round_trip_error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| {
                    (actual.re as f64 - expected.re as f64)
                        .hypot(actual.im as f64 - expected.im as f64)
                })
                .fold(0.0f64, f64::max);
            assert!(
                round_trip_error <= 2.0e-4,
                "HIP AMD COMGR N4096 batch8 static-shared round-trip error {round_trip_error:e} on {}",
                context.device_name()
            );

            assert_eq!(context.cached_hiprtc_code_count().unwrap(), 0);
        }
    }

    #[test]
    fn hip_runtime_caches_luts_and_transients_or_skip() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        assert_eq!(context.cached_module_count().unwrap(), 0);
        let uses_hiprtc = context.compiler.hiprtc().is_some();
        assert_eq!(context.cached_lut_count().unwrap(), 0);
        assert_eq!(context.cached_transient_buffer_count().unwrap(), 0);

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
            crate::FftConfig::new(vec![length]).with_tuning(tuning),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::Hip)
            .lower_transform(&ir)
            .unwrap();

        let first = context.execute_program_complex32(&source, &input).unwrap();
        let first_counts = (
            context.cached_hiprtc_code_count().unwrap(),
            context.cached_module_count().unwrap(),
            context.cached_lut_count().unwrap(),
            context.cached_transient_buffer_count().unwrap(),
        );
        if uses_hiprtc {
            assert!(first_counts.0 > 0);
        } else {
            assert_eq!(first_counts.0, 0);
        }
        assert!(first_counts.1 > 0);
        assert!(first_counts.2 > 0);
        assert!(first_counts.3 > 0);

        let second = context.execute_program_complex32(&source, &input).unwrap();
        let second_counts = (
            context.cached_hiprtc_code_count().unwrap(),
            context.cached_module_count().unwrap(),
            context.cached_lut_count().unwrap(),
            context.cached_transient_buffer_count().unwrap(),
        );
        assert_eq!(second, first);
        assert_eq!(second_counts, first_counts);

        context.clear_runtime_caches().unwrap();
        assert_eq!(context.cached_hiprtc_code_count().unwrap(), 0);
        assert_eq!(context.cached_module_count().unwrap(), 0);
        assert_eq!(context.cached_lut_count().unwrap(), 0);
        assert_eq!(context.cached_transient_buffer_count().unwrap(), 0);
        assert_eq!(context.pooled_stream_count().unwrap(), 0);
    }

    #[test]
    fn hiprtc_code_cache_reuses_repeat_transform_or_skip() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        if context.compiler.hiprtc().is_none() {
            return;
        }
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let input = vec![Complex32::new(1.0, 0.0); length];
        let first_output = context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        let NativeTransformOutput32::Complex(first_output) = first_output else {
            panic!("HIP cache test C2C returned real output");
        };
        let first = context.cached_hiprtc_code_count().unwrap();
        assert!(first > 0);
        let archive_data = context.hiprtc_code_archive_data().unwrap();
        let archive = HiprtcCodeArchive::decode(&archive_data).unwrap();
        assert_eq!(archive.identity, context.hiprtc_code_archive_identity());
        assert_eq!(archive.entry_count(), first);
        let mut wrong_identity = archive.clone();
        wrong_identity.identity.runtime_version += 1;
        assert!(
            context
                .restore_hiprtc_code_archive(&wrong_identity.encode())
                .is_err()
        );

        context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        assert_eq!(context.cached_hiprtc_code_count().unwrap(), first);

        context.clear_runtime_caches().unwrap();
        assert_eq!(context.cached_hiprtc_code_count().unwrap(), 0);
        context.restore_hiprtc_code_archive(&archive_data).unwrap();
        assert_eq!(context.cached_hiprtc_code_count().unwrap(), first);
        let restored_output = context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        let NativeTransformOutput32::Complex(restored_output) = restored_output else {
            panic!("restored HIP cache C2C returned real output");
        };
        assert_eq!(restored_output, first_output);
        assert_eq!(context.cached_hiprtc_code_count().unwrap(), first);
    }

    #[test]
    fn hip_f16_storage_f32_compute_matches_cpu_when_device_available() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length])
                .with_precision(crate::Precision::F16StorageF32Compute),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.15 * x).sin(), (0.09 * x).cos())
            })
            .collect::<Vec<_>>();
        let actual = context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        let NativeTransformOutput32::Complex(actual) = actual else {
            panic!("HIP F16-storage C2C transform returned a real output");
        };
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
            "HIP F16-storage error {error}"
        );
    }

    #[test]
    fn hip_f64_compute_f32_storage_matches_cpu_when_device_available() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        if !context.device_profile().supports_f64 {
            return;
        }
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length])
                .with_precision(crate::Precision::F64ComputeF32Storage),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.15 * x).sin(), (0.09 * x).cos())
            })
            .collect::<Vec<_>>();
        let actual = context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        let NativeTransformOutput32::Complex(actual) = actual else {
            panic!("HIP mixed-storage C2C transform returned a real output");
        };
        let oracle_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&oracle_input).unwrap();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re;
                let di = actual.im as f64 - expected.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-5 * length as f64,
            "HIP mixed-storage error {error}"
        );
    }

    #[test]
    fn hip_stockham_matches_cpu_when_device_available() {
        if !HipExecutionContext::probe().available() {
            return;
        }
        let context = HipExecutionContext::new(0).unwrap();
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.17 * x).sin(), (0.11 * x).cos())
            })
            .collect::<Vec<_>>();
        let actual = context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        let NativeTransformOutput32::Complex(actual) = actual else {
            panic!("HIP C2C transform returned a real output");
        };
        let oracle_input = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = crate::reference::dft(&oracle_input, crate::Direction::Forward, false);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                let dr = actual.re as f64 - expected.re;
                let di = actual.im as f64 - expected.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(error < 2.0e-3 * length as f64, "HIP F32 error {error}");
    }
}
