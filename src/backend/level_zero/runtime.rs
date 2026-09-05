//! Level Zero runtime capability adapter.
//!
//! The loader/device topology is queried dynamically. Native source execution prefers
//! an OpenCL-C -> SPIR-V compiler path such as `ocloc`; when that compiler is absent and
//! `opencl-runtime` is enabled, a matching Intel OpenCL device can supply native modules.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::ffi::CString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use libloading::Library;

#[cfg(feature = "opencl-runtime")]
use crate::backend::opencl::runtime::OpenClExecutionContext;

use crate::backend::native::NativeProgramSource;
use crate::backend::native_runtime::{
    NativeCompiledPassResourceReport, NativeCompiledResourceMetrics, NativeRuntimeAvailability,
    NativeTransformInput32, NativeTransformInput64, NativeTransformOutput32,
    NativeTransformOutput64, PreparedProgramStorage, load_first_library, prepare_program_complex32,
    prepare_program_complex64, runtime_error, unavailable,
};
use crate::complex::{Complex32, Complex64};
use crate::config::{Backend, DeviceProfile, GpuVendor, SubgroupProfile};
use crate::error::{Result, VkFftError};
use crate::program_ir::{
    ProgramAllocationId, ProgramAllocationKind, ProgramMemoryPlan, ProgramPass,
};
use crate::{ScalarType, TransformIr};

const ZE_RESULT_SUCCESS: c_int = 0;
const LEVEL_ZERO_MODULE_CACHE_MAX_ENTRIES: usize = 64;
const LEVEL_ZERO_KERNEL_POOL_MAX_INSTANCES: usize = 256;
const LEVEL_ZERO_COMMAND_LIST_POOL_MAX_INSTANCES: usize = 64;
const ZE_DEVICE_TYPE_GPU: c_uint = 1;
const LEVEL_ZERO_LOADER_CANDIDATES: &[&str] = &[
    "ze_loader.dll",
    "libze_loader.so.1",
    "libze_loader.so",
    "/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
];
const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: c_uint = 0x3;
const ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES: c_uint = 0x4;
const ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES: c_uint = 0x5;
const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES: c_uint = 0x6;
const ZE_STRUCTURE_TYPE_CONTEXT_DESC: c_uint = 0x0d;
const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: c_uint = 0x0e;
const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: c_uint = 0x0f;
const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: c_uint = 0x15;
const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: c_uint = 0x16;
const ZE_STRUCTURE_TYPE_MODULE_DESC: c_uint = 0x1b;
const ZE_STRUCTURE_TYPE_KERNEL_DESC: c_uint = 0x1d;
const ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES: c_uint = 0x1e;
const ZE_MODULE_FORMAT_IL_SPIRV: c_uint = 0;
#[cfg(feature = "opencl-runtime")]
const ZE_MODULE_FORMAT_NATIVE: c_uint = 1;
const ZE_COMMAND_QUEUE_MODE_DEFAULT: c_uint = 0;
const ZE_COMMAND_QUEUE_PRIORITY_NORMAL: c_uint = 0;
const ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE: c_uint = 1 << 0;
const ZE_DEVICE_MODULE_FLAG_FP64: c_uint = 1 << 1;
const ZE_SUBGROUPSIZE_COUNT: usize = 8;
const ZE_MAX_DEVICE_UUID_SIZE: usize = 16;
const ZE_MAX_DEVICE_NAME: usize = 256;
const ZE_MAX_NATIVE_KERNEL_UUID_SIZE: usize = 16;
const SPIRV_MAGIC: u32 = 0x0723_0203;
const OCLOC_LOG_LIMIT: usize = 8 * 1024;
const OCLOC_CACHE_MAX_ENTRIES: usize = 64;
const OCLOC_CACHE_MAX_BYTES: usize = 32 * 1024 * 1024;
pub const OCLOC_SPIRV_ARCHIVE_VERSION: u32 = 1;
const OCLOC_SPIRV_ARCHIVE_MAGIC: &[u8; 8] = b"VKFTOCLC";
const OCLOC_SPIRV_ARCHIVE_COMMIT_BYTES: usize = 40;
const OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES: usize = 16;
const OCLOC_SPIRV_ARCHIVE_HEADER_BYTES: usize =
    8 + 4 + OCLOC_SPIRV_ARCHIVE_COMMIT_BYTES + OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES + 8 + 8;
const LEVEL_ZERO_BUILD_LOG_LIMIT: usize = 64 * 1024;
static OCLOC_WORKSPACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

type ZeDriverHandle = *mut c_void;
type ZeDeviceHandle = *mut c_void;
type ZeContextHandle = *mut c_void;
type ZeCommandQueueHandle = *mut c_void;
type ZeCommandListHandle = *mut c_void;
type ZeModuleHandle = *mut c_void;
type ZeModuleBuildLogHandle = *mut c_void;
type ZeKernelHandle = *mut c_void;
type ZeEventHandle = *mut c_void;
type ZeFenceHandle = *mut c_void;
type ZeInit = unsafe extern "C" fn(c_uint) -> c_int;
type ZeDriverGet = unsafe extern "C" fn(*mut c_uint, *mut ZeDriverHandle) -> c_int;
type ZeDeviceGet = unsafe extern "C" fn(ZeDriverHandle, *mut c_uint, *mut ZeDeviceHandle) -> c_int;
type ZeDeviceGetProperties = unsafe extern "C" fn(ZeDeviceHandle, *mut ZeDeviceProperties) -> c_int;
type ZeDeviceGetComputeProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut ZeDeviceComputeProperties) -> c_int;
type ZeDeviceGetModuleProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut ZeDeviceModuleProperties) -> c_int;
type ZeDeviceGetCommandQueueGroupProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut c_uint, *mut ZeCommandQueueGroupProperties) -> c_int;
type ZeContextCreate =
    unsafe extern "C" fn(ZeDriverHandle, *const ZeContextDesc, *mut ZeContextHandle) -> c_int;
type ZeContextDestroy = unsafe extern "C" fn(ZeContextHandle) -> c_int;
type ZeMemAllocDevice = unsafe extern "C" fn(
    ZeContextHandle,
    *const ZeDeviceMemAllocDesc,
    usize,
    usize,
    ZeDeviceHandle,
    *mut *mut c_void,
) -> c_int;
type ZeMemAllocHost = unsafe extern "C" fn(
    ZeContextHandle,
    *const ZeHostMemAllocDesc,
    usize,
    usize,
    *mut *mut c_void,
) -> c_int;
type ZeMemFree = unsafe extern "C" fn(ZeContextHandle, *mut c_void) -> c_int;
type ZeModuleCreate = unsafe extern "C" fn(
    ZeContextHandle,
    ZeDeviceHandle,
    *const ZeModuleDesc,
    *mut ZeModuleHandle,
    *mut ZeModuleBuildLogHandle,
) -> c_int;
type ZeModuleDestroy = unsafe extern "C" fn(ZeModuleHandle) -> c_int;
type ZeModuleBuildLogGetString =
    unsafe extern "C" fn(ZeModuleBuildLogHandle, *mut usize, *mut c_char) -> c_int;
type ZeModuleBuildLogDestroy = unsafe extern "C" fn(ZeModuleBuildLogHandle) -> c_int;
type ZeKernelCreate =
    unsafe extern "C" fn(ZeModuleHandle, *const ZeKernelDesc, *mut ZeKernelHandle) -> c_int;
type ZeKernelDestroy = unsafe extern "C" fn(ZeKernelHandle) -> c_int;
type ZeKernelSetArgumentValue =
    unsafe extern "C" fn(ZeKernelHandle, c_uint, usize, *const c_void) -> c_int;
type ZeKernelSetGroupSize = unsafe extern "C" fn(ZeKernelHandle, c_uint, c_uint, c_uint) -> c_int;
type ZeKernelGetProperties = unsafe extern "C" fn(ZeKernelHandle, *mut ZeKernelProperties) -> c_int;
type ZeCommandQueueCreate = unsafe extern "C" fn(
    ZeContextHandle,
    ZeDeviceHandle,
    *const ZeCommandQueueDesc,
    *mut ZeCommandQueueHandle,
) -> c_int;
type ZeCommandQueueDestroy = unsafe extern "C" fn(ZeCommandQueueHandle) -> c_int;
type ZeCommandQueueExecuteCommandLists = unsafe extern "C" fn(
    ZeCommandQueueHandle,
    c_uint,
    *const ZeCommandListHandle,
    ZeFenceHandle,
) -> c_int;
type ZeCommandQueueSynchronize = unsafe extern "C" fn(ZeCommandQueueHandle, u64) -> c_int;
type ZeCommandListCreate = unsafe extern "C" fn(
    ZeContextHandle,
    ZeDeviceHandle,
    *const ZeCommandListDesc,
    *mut ZeCommandListHandle,
) -> c_int;
type ZeCommandListDestroy = unsafe extern "C" fn(ZeCommandListHandle) -> c_int;
type ZeCommandListReset = unsafe extern "C" fn(ZeCommandListHandle) -> c_int;
type ZeCommandListClose = unsafe extern "C" fn(ZeCommandListHandle) -> c_int;
type ZeCommandListAppendMemoryCopy = unsafe extern "C" fn(
    ZeCommandListHandle,
    *mut c_void,
    *const c_void,
    usize,
    ZeEventHandle,
    c_uint,
    *const ZeEventHandle,
) -> c_int;
type ZeCommandListAppendBarrier =
    unsafe extern "C" fn(ZeCommandListHandle, ZeEventHandle, c_uint, *const ZeEventHandle) -> c_int;
type ZeCommandListAppendLaunchKernel = unsafe extern "C" fn(
    ZeCommandListHandle,
    ZeKernelHandle,
    *const ZeGroupCount,
    ZeEventHandle,
    c_uint,
    *const ZeEventHandle,
) -> c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeDeviceProperties {
    stype: c_uint,
    p_next: *mut c_void,
    device_type: c_uint,
    vendor_id: u32,
    device_id: u32,
    flags: c_uint,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_mem_alloc_size: u64,
    max_hardware_contexts: u32,
    max_command_queue_priority: u32,
    num_threads_per_eu: u32,
    physical_eu_simd_width: u32,
    num_eus_per_subslice: u32,
    num_subslices_per_slice: u32,
    num_slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [u8; ZE_MAX_DEVICE_UUID_SIZE],
    name: [u8; ZE_MAX_DEVICE_NAME],
}

impl ZeDeviceProperties {
    const fn query() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
            p_next: core::ptr::null_mut(),
            device_type: 0,
            vendor_id: 0,
            device_id: 0,
            flags: 0,
            subdevice_id: 0,
            core_clock_rate: 0,
            max_mem_alloc_size: 0,
            max_hardware_contexts: 0,
            max_command_queue_priority: 0,
            num_threads_per_eu: 0,
            physical_eu_simd_width: 0,
            num_eus_per_subslice: 0,
            num_subslices_per_slice: 0,
            num_slices: 0,
            timer_resolution: 0,
            timestamp_valid_bits: 0,
            kernel_timestamp_valid_bits: 0,
            uuid: [0; ZE_MAX_DEVICE_UUID_SIZE],
            name: [0; ZE_MAX_DEVICE_NAME],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeDeviceComputeProperties {
    stype: c_uint,
    p_next: *mut c_void,
    max_total_group_size: u32,
    max_group_size_x: u32,
    max_group_size_y: u32,
    max_group_size_z: u32,
    max_group_count_x: u32,
    max_group_count_y: u32,
    max_group_count_z: u32,
    max_shared_local_memory: u32,
    num_sub_group_sizes: u32,
    sub_group_sizes: [u32; ZE_SUBGROUPSIZE_COUNT],
}

impl ZeDeviceComputeProperties {
    const fn query() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
            p_next: core::ptr::null_mut(),
            max_total_group_size: 0,
            max_group_size_x: 0,
            max_group_size_y: 0,
            max_group_size_z: 0,
            max_group_count_x: 0,
            max_group_count_y: 0,
            max_group_count_z: 0,
            max_shared_local_memory: 0,
            num_sub_group_sizes: 0,
            sub_group_sizes: [0; ZE_SUBGROUPSIZE_COUNT],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeDeviceModuleProperties {
    stype: c_uint,
    p_next: *mut c_void,
    spirv_version_supported: u32,
    flags: c_uint,
    fp16flags: c_uint,
    fp32flags: c_uint,
    fp64flags: c_uint,
    max_arguments_size: u32,
    printf_buffer_size: u32,
    native_kernel_supported: [u8; ZE_MAX_NATIVE_KERNEL_UUID_SIZE],
}

impl ZeDeviceModuleProperties {
    const fn query() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES,
            p_next: core::ptr::null_mut(),
            spirv_version_supported: 0,
            flags: 0,
            fp16flags: 0,
            fp32flags: 0,
            fp64flags: 0,
            max_arguments_size: 0,
            printf_buffer_size: 0,
            native_kernel_supported: [0; ZE_MAX_NATIVE_KERNEL_UUID_SIZE],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeCommandQueueGroupProperties {
    stype: c_uint,
    p_next: *mut c_void,
    flags: c_uint,
    max_memory_fill_pattern_size: usize,
    num_queues: u32,
}

impl ZeCommandQueueGroupProperties {
    const fn query() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
            p_next: core::ptr::null_mut(),
            flags: 0,
            max_memory_fill_pattern_size: 0,
            num_queues: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeContextDesc {
    stype: c_uint,
    p_next: *const c_void,
    flags: c_uint,
}

impl ZeContextDesc {
    const fn default_runtime() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            p_next: core::ptr::null(),
            flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeDeviceMemAllocDesc {
    stype: c_uint,
    p_next: *const c_void,
    flags: c_uint,
    ordinal: c_uint,
}

impl ZeDeviceMemAllocDesc {
    const fn default_runtime() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: core::ptr::null(),
            flags: 0,
            ordinal: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeHostMemAllocDesc {
    stype: c_uint,
    p_next: *const c_void,
    flags: c_uint,
}

impl ZeHostMemAllocDesc {
    const fn default_runtime() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
            p_next: core::ptr::null(),
            flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeModuleDesc {
    stype: c_uint,
    p_next: *const c_void,
    format: c_uint,
    input_size: usize,
    p_input_module: *const u8,
    p_build_flags: *const c_char,
    p_constants: *const c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeKernelDesc {
    stype: c_uint,
    p_next: *const c_void,
    flags: c_uint,
    p_kernel_name: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeKernelUuid {
    kid: [u8; ZE_MAX_NATIVE_KERNEL_UUID_SIZE],
    mid: [u8; ZE_MAX_NATIVE_KERNEL_UUID_SIZE],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeKernelProperties {
    stype: c_uint,
    p_next: *mut c_void,
    num_kernel_args: u32,
    required_group_size_x: u32,
    required_group_size_y: u32,
    required_group_size_z: u32,
    required_num_subgroups: u32,
    required_subgroup_size: u32,
    max_subgroup_size: u32,
    max_num_subgroups: u32,
    local_mem_size: u32,
    private_mem_size: u32,
    spill_mem_size: u32,
    uuid: ZeKernelUuid,
}

impl ZeKernelProperties {
    const fn query() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES,
            p_next: core::ptr::null_mut(),
            num_kernel_args: 0,
            required_group_size_x: 0,
            required_group_size_y: 0,
            required_group_size_z: 0,
            required_num_subgroups: 0,
            required_subgroup_size: 0,
            max_subgroup_size: 0,
            max_num_subgroups: 0,
            local_mem_size: 0,
            private_mem_size: 0,
            spill_mem_size: 0,
            uuid: ZeKernelUuid {
                kid: [0; ZE_MAX_NATIVE_KERNEL_UUID_SIZE],
                mid: [0; ZE_MAX_NATIVE_KERNEL_UUID_SIZE],
            },
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeCommandQueueDesc {
    stype: c_uint,
    p_next: *const c_void,
    ordinal: c_uint,
    index: c_uint,
    flags: c_uint,
    mode: c_uint,
    priority: c_uint,
}

impl ZeCommandQueueDesc {
    const fn compute(ordinal: u32) -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            p_next: core::ptr::null(),
            ordinal,
            index: 0,
            flags: 0,
            mode: ZE_COMMAND_QUEUE_MODE_DEFAULT,
            priority: ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ZeCommandListDesc {
    stype: c_uint,
    p_next: *const c_void,
    command_queue_group_ordinal: c_uint,
    flags: c_uint,
}

impl ZeCommandListDesc {
    const fn compute(ordinal: u32) -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            p_next: core::ptr::null(),
            command_queue_group_ordinal: ordinal,
            flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ZeGroupCount {
    group_count_x: c_uint,
    group_count_y: c_uint,
    group_count_z: c_uint,
}

#[derive(Debug, Clone, Copy)]
struct LevelZeroDeviceSelection {
    driver: ZeDriverHandle,
    device: ZeDeviceHandle,
}

#[derive(Clone, Copy)]
struct LevelZeroExecutionFns {
    context_create: ZeContextCreate,
    context_destroy: ZeContextDestroy,
    mem_alloc_device: ZeMemAllocDevice,
    mem_alloc_host: ZeMemAllocHost,
    mem_free: ZeMemFree,
    module_create: ZeModuleCreate,
    module_destroy: ZeModuleDestroy,
    module_build_log_get_string: ZeModuleBuildLogGetString,
    module_build_log_destroy: ZeModuleBuildLogDestroy,
    kernel_create: ZeKernelCreate,
    kernel_destroy: ZeKernelDestroy,
    kernel_set_argument_value: ZeKernelSetArgumentValue,
    kernel_set_group_size: ZeKernelSetGroupSize,
    kernel_get_properties: Option<ZeKernelGetProperties>,
    command_queue_create: ZeCommandQueueCreate,
    command_queue_destroy: ZeCommandQueueDestroy,
    command_queue_execute_command_lists: ZeCommandQueueExecuteCommandLists,
    command_queue_synchronize: ZeCommandQueueSynchronize,
    command_list_create: ZeCommandListCreate,
    command_list_destroy: ZeCommandListDestroy,
    command_list_reset: ZeCommandListReset,
    command_list_close: ZeCommandListClose,
    command_list_append_memory_copy: ZeCommandListAppendMemoryCopy,
    command_list_append_barrier: ZeCommandListAppendBarrier,
    command_list_append_launch_kernel: ZeCommandListAppendLaunchKernel,
}

struct LevelZeroRuntimeApi {
    _library: Library,
    fns: LevelZeroExecutionFns,
}

impl LevelZeroRuntimeApi {
    fn from_library(library: Library) -> Result<Self> {
        // SAFETY: every symbol is copied as a function pointer while `library` remains
        // owned by the returned API for at least as long as any call can occur.
        unsafe {
            macro_rules! load {
                ($ty:ty, $name:literal) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|error| {
                            unavailable(Backend::LevelZero, format!("missing {}: {error}", $name))
                        })?
                };
            }
            macro_rules! load_optional {
                ($ty:ty, $name:literal) => {
                    library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .ok()
                        .map(|symbol| *symbol)
                };
            }
            let fns = LevelZeroExecutionFns {
                context_create: load!(ZeContextCreate, "zeContextCreate"),
                context_destroy: load!(ZeContextDestroy, "zeContextDestroy"),
                mem_alloc_device: load!(ZeMemAllocDevice, "zeMemAllocDevice"),
                mem_alloc_host: load!(ZeMemAllocHost, "zeMemAllocHost"),
                mem_free: load!(ZeMemFree, "zeMemFree"),
                module_create: load!(ZeModuleCreate, "zeModuleCreate"),
                module_destroy: load!(ZeModuleDestroy, "zeModuleDestroy"),
                module_build_log_get_string: load!(
                    ZeModuleBuildLogGetString,
                    "zeModuleBuildLogGetString"
                ),
                module_build_log_destroy: load!(ZeModuleBuildLogDestroy, "zeModuleBuildLogDestroy"),
                kernel_create: load!(ZeKernelCreate, "zeKernelCreate"),
                kernel_destroy: load!(ZeKernelDestroy, "zeKernelDestroy"),
                kernel_set_argument_value: load!(
                    ZeKernelSetArgumentValue,
                    "zeKernelSetArgumentValue"
                ),
                kernel_set_group_size: load!(ZeKernelSetGroupSize, "zeKernelSetGroupSize"),
                kernel_get_properties: load_optional!(
                    ZeKernelGetProperties,
                    "zeKernelGetProperties"
                ),
                command_queue_create: load!(ZeCommandQueueCreate, "zeCommandQueueCreate"),
                command_queue_destroy: load!(ZeCommandQueueDestroy, "zeCommandQueueDestroy"),
                command_queue_execute_command_lists: load!(
                    ZeCommandQueueExecuteCommandLists,
                    "zeCommandQueueExecuteCommandLists"
                ),
                command_queue_synchronize: load!(
                    ZeCommandQueueSynchronize,
                    "zeCommandQueueSynchronize"
                ),
                command_list_create: load!(ZeCommandListCreate, "zeCommandListCreate"),
                command_list_destroy: load!(ZeCommandListDestroy, "zeCommandListDestroy"),
                command_list_reset: load!(ZeCommandListReset, "zeCommandListReset"),
                command_list_close: load!(ZeCommandListClose, "zeCommandListClose"),
                command_list_append_memory_copy: load!(
                    ZeCommandListAppendMemoryCopy,
                    "zeCommandListAppendMemoryCopy"
                ),
                command_list_append_barrier: load!(
                    ZeCommandListAppendBarrier,
                    "zeCommandListAppendBarrier"
                ),
                command_list_append_launch_kernel: load!(
                    ZeCommandListAppendLaunchKernel,
                    "zeCommandListAppendLaunchKernel"
                ),
            };
            Ok(Self {
                _library: library,
                fns,
            })
        }
    }
}

fn check_level_zero(code: c_int, operation: &'static str) -> Result<()> {
    if code == ZE_RESULT_SUCCESS {
        Ok(())
    } else {
        Err(runtime_error(
            Backend::LevelZero,
            format!("{operation} returned ze_result_t 0x{:08x}", code as u32),
        ))
    }
}

struct LevelZeroAllocation<'a> {
    fns: &'a LevelZeroExecutionFns,
    context: ZeContextHandle,
    ptr: *mut c_void,
    bytes: usize,
    owned: bool,
}

impl LevelZeroAllocation<'_> {
    fn relinquish(&mut self) -> *mut c_void {
        self.owned = false;
        self.ptr
    }
}

impl Drop for LevelZeroAllocation<'_> {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                (self.fns.mem_free)(self.context, self.ptr);
            }
        }
    }
}

#[derive(Debug)]
struct LevelZeroModuleCache {
    entries: HashMap<Vec<u8>, ZeModuleHandle>,
    max_entries: usize,
}

impl LevelZeroModuleCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, spirv: &[u8]) -> Option<ZeModuleHandle> {
        self.entries.get(spirv).copied()
    }

    fn insert_if_capacity(&mut self, spirv: &[u8], module: ZeModuleHandle) -> bool {
        if self.entries.len() >= self.max_entries || self.entries.contains_key(spirv) {
            return false;
        }
        self.entries.insert(spirv.to_vec(), module);
        true
    }

    fn drain_modules(&mut self) -> impl Iterator<Item = ZeModuleHandle> + '_ {
        self.entries.drain().map(|(_, module)| module)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LevelZeroKernelPoolKey {
    source: String,
    entry_point: String,
}

#[derive(Debug)]
struct LevelZeroKernelPool {
    entries: HashMap<LevelZeroKernelPoolKey, Vec<ZeKernelHandle>>,
    total_instances: usize,
    max_instances: usize,
}

impl LevelZeroKernelPool {
    fn new(max_instances: usize) -> Self {
        Self {
            entries: HashMap::new(),
            total_instances: 0,
            max_instances,
        }
    }

    fn len(&self) -> usize {
        self.total_instances
    }

    fn take(&mut self, key: &LevelZeroKernelPoolKey) -> Option<ZeKernelHandle> {
        let (kernel, remove_entry) = {
            let kernels = self.entries.get_mut(key)?;
            let kernel = kernels.pop()?;
            (kernel, kernels.is_empty())
        };
        self.total_instances = self.total_instances.saturating_sub(1);
        if remove_entry {
            self.entries.remove(key);
        }
        Some(kernel)
    }

    fn insert_if_capacity(&mut self, key: LevelZeroKernelPoolKey, kernel: ZeKernelHandle) -> bool {
        if self.total_instances >= self.max_instances {
            return false;
        }
        self.entries.entry(key).or_default().push(kernel);
        self.total_instances += 1;
        true
    }

    fn drain_kernels(&mut self) -> Vec<ZeKernelHandle> {
        let kernels = self
            .entries
            .drain()
            .flat_map(|(_, kernels)| kernels)
            .collect::<Vec<_>>();
        self.total_instances = 0;
        kernels
    }
}

struct LevelZeroLoadedKernel<'a> {
    fns: &'a LevelZeroExecutionFns,
    module: ZeModuleHandle,
    kernel: ZeKernelHandle,
    owns_module: bool,
    owns_kernel: bool,
    pool_key: Option<LevelZeroKernelPoolKey>,
}

impl LevelZeroLoadedKernel<'_> {
    fn relinquish_kernel(&mut self) -> ZeKernelHandle {
        self.owns_kernel = false;
        self.kernel
    }
}

impl Drop for LevelZeroLoadedKernel<'_> {
    fn drop(&mut self) {
        unsafe {
            if self.owns_kernel && !self.kernel.is_null() {
                (self.fns.kernel_destroy)(self.kernel);
            }
            if self.owns_module && !self.module.is_null() {
                (self.fns.module_destroy)(self.module);
            }
        }
    }
}

struct LevelZeroCommandListGuard<'a> {
    fns: &'a LevelZeroExecutionFns,
    handle: ZeCommandListHandle,
    owned: bool,
}

impl LevelZeroCommandListGuard<'_> {
    fn relinquish(&mut self) -> ZeCommandListHandle {
        self.owned = false;
        self.handle
    }
}

impl Drop for LevelZeroCommandListGuard<'_> {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe {
                (self.fns.command_list_destroy)(self.handle);
            }
        }
    }
}

struct LevelZeroOwnedQueue<'a> {
    fns: &'a LevelZeroExecutionFns,
    handle: ZeCommandQueueHandle,
    owned: bool,
}

impl LevelZeroOwnedQueue<'_> {
    fn synchronize(&self) -> Result<()> {
        check_level_zero(
            unsafe { (self.fns.command_queue_synchronize)(self.handle, u64::MAX) },
            "zeCommandQueueSynchronize",
        )
    }

    fn relinquish(&mut self) -> ZeCommandQueueHandle {
        self.owned = false;
        self.handle
    }
}

impl Drop for LevelZeroOwnedQueue<'_> {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe {
                (self.fns.command_queue_synchronize)(self.handle, u64::MAX);
                (self.fns.command_queue_destroy)(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

struct LevelZeroPendingProgram<'a> {
    queue: LevelZeroOwnedQueue<'a>,
    context: &'a LevelZeroExecutionContext,
    prepared: PreparedProgramStorage,
    device_allocations: Vec<LevelZeroAllocation<'a>>,
    host_allocations: Vec<Option<LevelZeroAllocation<'a>>>,
    pending_luts: Vec<(usize, Vec<u8>)>,
    _command_list: LevelZeroCommandListGuard<'a>,
    _kernels: Vec<LevelZeroLoadedKernel<'a>>,
    completed: bool,
}

impl LevelZeroPendingProgram<'_> {
    fn finish(&mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        self.queue.synchronize()?;
        let output_host = self
            .host_allocations
            .get(self.prepared.output_allocation.0)
            .and_then(Option::as_ref)
            .ok_or(VkFftError::InvalidKernelIr(
                "Level Zero program is missing its output host allocation",
            ))?;
        let output_bytes = self.prepared.output_bytes_mut()?;
        unsafe {
            ptr::copy_nonoverlapping(
                output_host.ptr.cast::<u8>(),
                output_bytes.as_mut_ptr(),
                output_bytes.len(),
            );
        }
        for (index, key) in self.pending_luts.drain(..) {
            let allocation =
                self.device_allocations
                    .get_mut(index)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "pending Level Zero LUT references a missing device allocation",
                    ))?;
            let ptr = allocation.relinquish();
            let mut cache = self.context.lut_cache_lock()?;
            if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
                entry.insert(ptr);
            } else {
                check_level_zero(
                    unsafe { (self.context.runtime.fns.mem_free)(self.context.context, ptr) },
                    "zeMemFree(duplicate LUT)",
                )?;
            }
        }
        {
            let mut pool = self.context.device_buffer_pool.lock().map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero device buffer pool lock is poisoned",
                )
            })?;
            for (allocation_plan, allocation) in self
                .prepared
                .memory_plan
                .allocations
                .iter()
                .zip(&mut self.device_allocations)
            {
                if allocation_plan.kind != ProgramAllocationKind::LookupTable {
                    let bytes = allocation.bytes;
                    pool.entry(bytes).or_default().push(allocation.relinquish());
                }
            }
        }
        {
            let mut pool = self.context.host_buffer_pool.lock().map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero host buffer pool lock is poisoned",
                )
            })?;
            for allocation in self.host_allocations.iter_mut().flatten() {
                let bytes = allocation.bytes;
                pool.entry(bytes).or_default().push(allocation.relinquish());
            }
        }
        self.context.recycle_command_list(&mut self._command_list);
        for kernel in &mut self._kernels {
            self.context.recycle_kernel(kernel)?;
        }
        {
            let queue = self.queue.relinquish();
            self.context
                .queue_pool
                .lock()
                .map_err(|_| {
                    runtime_error(Backend::LevelZero, "Level Zero queue pool lock is poisoned")
                })?
                .push(queue);
        }
        self.completed = true;
        Ok(())
    }
}

impl Drop for LevelZeroPendingProgram<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.queue.synchronize();
        }
        self.context
            .active_submissions
            .fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct LevelZeroProgramTicket32<'a> {
    pending: LevelZeroPendingProgram<'a>,
}

impl LevelZeroProgramTicket32<'_> {
    pub fn wait(mut self) -> Result<Vec<Complex32>> {
        self.pending.finish()?;
        self.pending.prepared.output_complex32()
    }
}

pub struct LevelZeroProgramTicket64<'a> {
    pending: LevelZeroPendingProgram<'a>,
}

impl LevelZeroProgramTicket64<'_> {
    pub fn wait(mut self) -> Result<Vec<Complex64>> {
        self.pending.finish()?;
        self.pending.prepared.output_complex64()
    }
}

struct CompiledLevelZeroModule {
    format: c_uint,
    bytes: Vec<u8>,
}

enum LevelZeroExecutionCompiler {
    Ocloc(LevelZeroOclocCompiler),
    #[cfg(feature = "opencl-runtime")]
    IntelOpenClNative(Box<OpenClExecutionContext>),
}

impl LevelZeroExecutionCompiler {
    fn new_for_device(device_name: &str) -> Result<Self> {
        if let Ok(compiler) = LevelZeroOclocCompiler::new() {
            return Ok(Self::Ocloc(compiler));
        }
        #[cfg(feature = "opencl-runtime")]
        {
            let availability = OpenClExecutionContext::probe();
            for device_index in 0..availability.device_count {
                let Ok(context) = OpenClExecutionContext::new(device_index) else {
                    continue;
                };
                if context.device_profile().vendor == GpuVendor::Intel
                    && context.device_name().eq_ignore_ascii_case(device_name)
                {
                    return Ok(Self::IntelOpenClNative(Box::new(context)));
                }
            }
        }
        Err(unavailable(
            Backend::LevelZero,
            format!(
                "no Level Zero compiler is available for {device_name}: ocloc was not found and no matching Intel OpenCL compiler device was discovered"
            ),
        ))
    }

    fn compile(&self, source: &str) -> Result<CompiledLevelZeroModule> {
        match self {
            Self::Ocloc(compiler) => Ok(CompiledLevelZeroModule {
                format: ZE_MODULE_FORMAT_IL_SPIRV,
                bytes: compiler.compile_opencl_c_to_spirv(source)?,
            }),
            #[cfg(feature = "opencl-runtime")]
            Self::IntelOpenClNative(context) => Ok(CompiledLevelZeroModule {
                format: ZE_MODULE_FORMAT_NATIVE,
                bytes: context.compile_source_to_native_binary(source)?,
            }),
        }
    }

    fn cache_archive_data(&self) -> Result<Vec<u8>> {
        match self {
            Self::Ocloc(compiler) => compiler.cache_archive_data(),
            #[cfg(feature = "opencl-runtime")]
            Self::IntelOpenClNative(_) => Err(unavailable(
                Backend::LevelZero,
                "OCLOC SPIR-V cache archives are unavailable with the Intel OpenCL native fallback",
            )),
        }
    }

    fn restore_cache_archive(&self, encoded: &[u8]) -> Result<()> {
        match self {
            Self::Ocloc(compiler) => compiler.restore_cache_archive(encoded),
            #[cfg(feature = "opencl-runtime")]
            Self::IntelOpenClNative(_) => Err(unavailable(
                Backend::LevelZero,
                "OCLOC SPIR-V cache archives cannot be restored into the Intel OpenCL native fallback",
            )),
        }
    }
}

pub struct LevelZeroExecutionContext {
    runtime: LevelZeroRuntimeApi,
    compiler: LevelZeroExecutionCompiler,
    selection: LevelZeroDeviceSelection,
    context: ZeContextHandle,
    queue_pool: Mutex<Vec<ZeCommandQueueHandle>>,
    device_buffer_pool: Mutex<HashMap<usize, Vec<*mut c_void>>>,
    host_buffer_pool: Mutex<HashMap<usize, Vec<*mut c_void>>>,
    lut_cache: Mutex<HashMap<Vec<u8>, *mut c_void>>,
    module_cache: Mutex<LevelZeroModuleCache>,
    kernel_pool: Mutex<LevelZeroKernelPool>,
    command_list_pool: Mutex<Vec<ZeCommandListHandle>>,
    active_submissions: AtomicUsize,
    compute_queue_group_ordinal: u32,
    device_name: String,
    profile: DeviceProfile,
}

impl LevelZeroExecutionContext {
    pub fn probe() -> NativeRuntimeAvailability {
        let mut availability = LevelZeroRuntimeAdapter::probe();
        if availability.loader_available
            && !availability.compiler_available
            && availability.device_count > 0
            && let Ok((library, _)) =
                load_first_library(Backend::LevelZero, LEVEL_ZERO_LOADER_CANDIDATES)
            && let Ok(selection) = level_zero_device_selection(&library, 0)
            && let Ok(device_name) = level_zero_device_name_for_handle(&library, selection.device)
            && LevelZeroExecutionCompiler::new_for_device(&device_name).is_ok()
        {
            availability.compiler_available = true;
            availability.detail = format!(
                "{}; Intel OpenCL native compiler fallback available for {device_name}",
                availability.detail
            );
        }
        availability
    }

    pub fn new(device_index: usize) -> Result<Self> {
        let (library, _) = load_first_library(Backend::LevelZero, LEVEL_ZERO_LOADER_CANDIDATES)?;
        let selection = level_zero_device_selection(&library, device_index)?;
        let profile = level_zero_device_profile_for_handle(&library, selection.device)?;
        let compute_queue_group_ordinal =
            level_zero_compute_queue_group_ordinal_for_handle(&library, selection.device)?;
        let device_name = level_zero_device_name_for_handle(&library, selection.device)?;
        let compiler = LevelZeroExecutionCompiler::new_for_device(&device_name)?;
        let runtime = LevelZeroRuntimeApi::from_library(library)?;

        let context_desc = ZeContextDesc::default_runtime();
        let mut context = ptr::null_mut();
        let context_result =
            unsafe { (runtime.fns.context_create)(selection.driver, &context_desc, &mut context) };
        if context_result != ZE_RESULT_SUCCESS || context.is_null() {
            if !context.is_null() {
                unsafe {
                    (runtime.fns.context_destroy)(context);
                }
            }
            if context_result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeContextCreate succeeded with a null context handle",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeContextCreate returned ze_result_t 0x{:08x}",
                    context_result as u32
                ),
            ));
        }

        let queue_desc = ZeCommandQueueDesc::compute(compute_queue_group_ordinal);
        let mut queue = ptr::null_mut();
        let queue_result = unsafe {
            (runtime.fns.command_queue_create)(context, selection.device, &queue_desc, &mut queue)
        };
        if queue_result != ZE_RESULT_SUCCESS || queue.is_null() {
            unsafe {
                if !queue.is_null() {
                    (runtime.fns.command_queue_destroy)(queue);
                }
                (runtime.fns.context_destroy)(context);
            }
            if queue_result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeCommandQueueCreate succeeded with a null queue handle",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeCommandQueueCreate returned ze_result_t 0x{:08x}",
                    queue_result as u32
                ),
            ));
        }

        let mut execution = Self {
            runtime,
            compiler,
            selection,
            context,
            queue_pool: Mutex::new(vec![queue]),
            device_buffer_pool: Mutex::new(HashMap::new()),
            host_buffer_pool: Mutex::new(HashMap::new()),
            lut_cache: Mutex::new(HashMap::new()),
            module_cache: Mutex::new(LevelZeroModuleCache::new(
                LEVEL_ZERO_MODULE_CACHE_MAX_ENTRIES,
            )),
            kernel_pool: Mutex::new(LevelZeroKernelPool::new(
                LEVEL_ZERO_KERNEL_POOL_MAX_INSTANCES,
            )),
            command_list_pool: Mutex::new(Vec::new()),
            active_submissions: AtomicUsize::new(0),
            compute_queue_group_ordinal,
            device_name,
            profile,
        };
        if let Some(subgroup_size) = execution.proven_level_zero_subgroup_width() {
            execution.profile.subgroup =
                proven_level_zero_subgroup_profile(execution.profile.subgroup, subgroup_size);
        }
        Ok(execution)
    }

    pub const fn device_profile(&self) -> DeviceProfile {
        self.profile
    }

    pub fn new_with_ocloc_spirv_cache_archive(device_index: usize, encoded: &[u8]) -> Result<Self> {
        let context = Self::new(device_index)?;
        context.restore_ocloc_spirv_cache_archive(encoded)?;
        Ok(context)
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn proven_level_zero_subgroup_width(&self) -> Option<usize> {
        let physical = self.profile.subgroup;
        if self.profile.vendor != GpuVendor::Intel {
            return None;
        }
        let candidates = level_zero_subgroup_probe_candidates(physical);
        if candidates.is_empty() {
            return None;
        }
        let compiler_kind = match &self.compiler {
            LevelZeroExecutionCompiler::Ocloc(_) => "ocloc",
            #[cfg(feature = "opencl-runtime")]
            LevelZeroExecutionCompiler::IntelOpenClNative(_) => "intel-opencl-native",
        };
        static CACHE: OnceLock<Mutex<HashMap<String, Option<usize>>>> = OnceLock::new();
        let key = format!(
            "{}\0{}\0{}\0{}\0{}",
            self.device_name, compiler_kind, physical.size, physical.min_size, physical.max_size
        );
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(cache) = cache.lock()
            && let Some(proven) = cache.get(&key).copied()
        {
            return proven;
        }
        let proven = candidates.into_iter().find(|&subgroup_size| {
            self.probe_level_zero_subgroup_language_surface(subgroup_size)
                .unwrap_or(false)
        });
        if let Ok(mut cache) = cache.lock() {
            cache.insert(key, proven);
        }
        proven
    }

    fn probe_level_zero_subgroup_language_surface(&self, subgroup_size: usize) -> Result<bool> {
        if subgroup_size < 2 || !subgroup_size.is_power_of_two() {
            return Ok(false);
        }
        let max_local_x = self
            .profile
            .max_threads_per_block
            .min(self.profile.max_workgroup_size[0]);
        let local_size = subgroup_size
            .checked_mul(2)
            .filter(|size| *size <= max_local_x)
            .unwrap_or(subgroup_size);
        if local_size > max_local_x || !local_size.is_multiple_of(subgroup_size) {
            return Ok(false);
        }
        let source = format!(
            "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n\
#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n\
__attribute__((intel_reqd_sub_group_size({subgroup_size}))) __kernel __attribute__((reqd_work_group_size({local_size}, 1, 1))) void VkFFT_main(__global uint4* output) {{\n\
    uint gid = (uint)get_global_id(0);\n\
    uint lane = (uint)get_sub_group_local_id();\n\
    uint subgroup = (uint)get_sub_group_id();\n\
    uint width = (uint)get_sub_group_size();\n\
    uint shuffled = intel_sub_group_shuffle(lane, width - 1u - lane);\n\
    output[gid] = (uint4)(lane, subgroup, width, shuffled);\n\
}}\n"
        );
        let compiled = self.compiler.compile(&source)?;
        let module = self.create_module(compiled.format, &compiled.bytes)?;
        let entry = CString::new("VkFFT_main").expect("static Level Zero subgroup entry point");
        let local_size_u32 =
            u32::try_from(local_size).map_err(|_| VkFftError::ArithmeticOverflow {
                operation: "Level Zero subgroup probe local size",
            })?;
        let workgroup = crate::kernel_ir::WorkgroupSize {
            x: local_size_u32,
            y: 1,
            z: 1,
        };
        let kernel = self.create_kernel_from_module(module, true, None, &entry, workgroup)?;
        let output_bytes = local_size
            .checked_mul(4 * core::mem::size_of::<u32>())
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Level Zero subgroup probe output bytes",
            })?;
        let device = self.allocate_device(output_bytes)?;
        let host = self.allocate_host(output_bytes)?;
        let queue = self.take_queue()?;
        let command_list = self.create_command_list()?;
        let device_ptr = device.ptr;
        check_level_zero(
            unsafe {
                (self.runtime.fns.kernel_set_argument_value)(
                    kernel.kernel,
                    0,
                    core::mem::size_of::<*mut c_void>(),
                    (&device_ptr as *const *mut c_void).cast(),
                )
            },
            "zeKernelSetArgumentValue(subgroup probe)",
        )?;
        let groups = ZeGroupCount {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        };
        check_level_zero(
            unsafe {
                (self.runtime.fns.command_list_append_launch_kernel)(
                    command_list.handle,
                    kernel.kernel,
                    &groups,
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                )
            },
            "zeCommandListAppendLaunchKernel(subgroup probe)",
        )?;
        check_level_zero(
            unsafe {
                (self.runtime.fns.command_list_append_barrier)(
                    command_list.handle,
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                )
            },
            "zeCommandListAppendBarrier(subgroup probe -> readback)",
        )?;
        check_level_zero(
            unsafe {
                (self.runtime.fns.command_list_append_memory_copy)(
                    command_list.handle,
                    host.ptr,
                    device.ptr.cast_const(),
                    output_bytes,
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                )
            },
            "zeCommandListAppendMemoryCopy(subgroup probe DtoH)",
        )?;
        check_level_zero(
            unsafe { (self.runtime.fns.command_list_close)(command_list.handle) },
            "zeCommandListClose(subgroup probe)",
        )?;
        let lists = [command_list.handle];
        check_level_zero(
            unsafe {
                (self.runtime.fns.command_queue_execute_command_lists)(
                    queue.handle,
                    1,
                    lists.as_ptr(),
                    ptr::null_mut(),
                )
            },
            "zeCommandQueueExecuteCommandLists(subgroup probe)",
        )?;
        queue.synchronize()?;
        let output = unsafe { std::slice::from_raw_parts(host.ptr.cast::<u8>(), output_bytes) };
        Ok(validate_level_zero_subgroup_probe_output(
            output,
            subgroup_size,
            local_size,
        ))
    }

    pub fn ocloc_spirv_cache_archive_data(&self) -> Result<Vec<u8>> {
        self.compiler.cache_archive_data()
    }

    pub fn restore_ocloc_spirv_cache_archive(&self, encoded: &[u8]) -> Result<()> {
        self.compiler.restore_cache_archive(encoded)
    }

    pub const fn compute_queue_group_ordinal(&self) -> u32 {
        self.compute_queue_group_ordinal
    }

    pub fn pooled_queue_count(&self) -> Result<usize> {
        self.queue_pool.lock().map(|pool| pool.len()).map_err(|_| {
            runtime_error(Backend::LevelZero, "Level Zero queue pool lock is poisoned")
        })
    }

    pub fn cached_device_buffer_count(&self) -> Result<usize> {
        self.device_buffer_pool
            .lock()
            .map(|pool| pool.values().map(Vec::len).sum())
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero device buffer pool lock is poisoned",
                )
            })
    }

    pub fn cached_host_buffer_count(&self) -> Result<usize> {
        self.host_buffer_pool
            .lock()
            .map(|pool| pool.values().map(Vec::len).sum())
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero host buffer pool lock is poisoned",
                )
            })
    }

    pub fn cached_lut_count(&self) -> Result<usize> {
        self.lut_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| runtime_error(Backend::LevelZero, "Level Zero LUT cache lock is poisoned"))
    }

    fn lut_cache_lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<Vec<u8>, *mut c_void>>> {
        self.lut_cache
            .lock()
            .map_err(|_| runtime_error(Backend::LevelZero, "Level Zero LUT cache lock is poisoned"))
    }

    pub fn pooled_command_list_count(&self) -> Result<usize> {
        self.command_list_pool
            .lock()
            .map(|pool| pool.len())
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero command-list pool lock is poisoned",
                )
            })
    }

    pub fn pooled_kernel_instance_count(&self) -> Result<usize> {
        self.kernel_pool.lock().map(|pool| pool.len()).map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero kernel pool lock is poisoned",
            )
        })
    }

    pub fn cached_module_count(&self) -> Result<usize> {
        self.module_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero module cache lock is poisoned",
                )
            })
    }

    pub fn submit_program_complex32<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<LevelZeroProgramTicket32<'a>> {
        source.validate()?;
        if source.backend != Backend::LevelZero {
            return Err(VkFftError::InvalidKernelIr(
                "Level Zero F32-storage execution requires a Level Zero native program",
            ));
        }
        if source.program.scalar == ScalarType::F64 && !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Level Zero runtime",
                precision: "f64 compute with f32 storage",
            });
        }
        let prepared = prepare_program_complex32(Backend::LevelZero, &source.program, input)?;
        Ok(LevelZeroProgramTicket32 {
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
    ) -> Result<LevelZeroProgramTicket64<'a>> {
        source.validate()?;
        if source.backend != Backend::LevelZero || source.program.scalar != ScalarType::F64 {
            return Err(VkFftError::InvalidKernelIr(
                "Level Zero F64 execution requires a Level Zero/F64 native program",
            ));
        }
        if !self.profile.supports_f64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Level Zero runtime",
                precision: "f64",
            });
        }
        let prepared = prepare_program_complex64(Backend::LevelZero, &source.program, input)?;
        Ok(LevelZeroProgramTicket64 {
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

    crate::backend::native_runtime::impl_native_double_double_runtime_facade!(
        Backend::LevelZero,
        "Level Zero"
    );
    crate::backend::native_runtime::impl_native_transform_convenience_facade!(
        Backend::LevelZero,
        LevelZeroProgramTicket32,
        LevelZeroProgramTicket64
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
    ) -> Result<LevelZeroPendingProgram<'a>> {
        let fns = &self.runtime.fns;
        let queue = self.take_queue()?;
        let command_list = self.create_command_list()?;
        let mut pending_luts = Vec::new();
        let mut device_allocations = Vec::with_capacity(prepared.allocations.len());
        let mut host_allocations = Vec::with_capacity(prepared.allocations.len());

        for (index, (allocation_plan, prepared_allocation)) in prepared
            .memory_plan
            .allocations
            .iter()
            .zip(&prepared.allocations)
            .enumerate()
        {
            let byte_len = prepared_allocation.byte_len;
            if byte_len == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "Level Zero program contains a zero-byte physical allocation",
                ));
            }
            let host_bytes = prepared_allocation.host_bytes.as_deref();
            if allocation_plan.kind == ProgramAllocationKind::LookupTable {
                let bytes = host_bytes.ok_or(VkFftError::InvalidKernelIr(
                    "Level Zero lookup-table allocation is missing initialization bytes",
                ))?;
                let cached = self.lut_cache_lock()?.get(bytes).copied();
                if let Some(ptr) = cached {
                    device_allocations.push(LevelZeroAllocation {
                        fns,
                        context: self.context,
                        ptr,
                        bytes: byte_len,
                        owned: false,
                    });
                    host_allocations.push(None);
                    continue;
                }
            }

            let device = self.allocate_device(byte_len)?;
            let host = if let Some(bytes) = host_bytes {
                let host = self.allocate_host(byte_len)?;
                unsafe {
                    ptr::copy_nonoverlapping(bytes.as_ptr(), host.ptr.cast::<u8>(), bytes.len());
                }
                check_level_zero(
                    unsafe {
                        (fns.command_list_append_memory_copy)(
                            command_list.handle,
                            device.ptr,
                            host.ptr.cast_const(),
                            bytes.len(),
                            ptr::null_mut(),
                            0,
                            ptr::null(),
                        )
                    },
                    "zeCommandListAppendMemoryCopy(HtoD)",
                )?;
                Some(host)
            } else {
                None
            };
            device_allocations.push(device);
            host_allocations.push(host);
            if allocation_plan.kind == ProgramAllocationKind::LookupTable {
                pending_luts.push((
                    index,
                    host_bytes
                        .expect("validated Level Zero LUT host bytes above")
                        .to_vec(),
                ));
            }
        }

        check_level_zero(
            unsafe {
                (fns.command_list_append_barrier)(
                    command_list.handle,
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                )
            },
            "zeCommandListAppendBarrier(HtoD -> kernels)",
        )?;

        let mut kernels = Vec::with_capacity(source.shaders.len());
        for (pass, shader) in source.program.passes.iter().zip(&source.shaders) {
            shader.validate()?;
            let compiled = self.compiler.compile(&shader.source)?;
            let kernel = self.create_kernel(
                compiled.format,
                &compiled.bytes,
                &shader.source,
                shader.entry_point,
                shader.workgroup_size,
            )?;
            let allocation_ids = level_zero_ordered_allocation_ids(pass, &prepared.memory_plan)?;
            for (argument_index, allocation_id) in allocation_ids.into_iter().enumerate() {
                let allocation =
                    device_allocations
                        .get(allocation_id.0)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "Level Zero pass references a missing device allocation",
                        ))?;
                let value = allocation.ptr;
                check_level_zero(
                    unsafe {
                        (fns.kernel_set_argument_value)(
                            kernel.kernel,
                            u32::try_from(argument_index).map_err(|_| {
                                VkFftError::ArithmeticOverflow {
                                    operation: "Level Zero kernel argument index",
                                }
                            })?,
                            std::mem::size_of::<*mut c_void>(),
                            (&value as *const *mut c_void).cast(),
                        )
                    },
                    "zeKernelSetArgumentValue",
                )?;
            }
            let groups = ZeGroupCount {
                group_count_x: shader.dispatch.x,
                group_count_y: shader.dispatch.y,
                group_count_z: shader.dispatch.z,
            };
            check_level_zero(
                unsafe {
                    (fns.command_list_append_launch_kernel)(
                        command_list.handle,
                        kernel.kernel,
                        &groups,
                        ptr::null_mut(),
                        0,
                        ptr::null(),
                    )
                },
                "zeCommandListAppendLaunchKernel",
            )?;
            check_level_zero(
                unsafe {
                    (fns.command_list_append_barrier)(
                        command_list.handle,
                        ptr::null_mut(),
                        0,
                        ptr::null(),
                    )
                },
                "zeCommandListAppendBarrier(kernel -> next command)",
            )?;
            kernels.push(kernel);
        }

        let output_index = prepared.output_allocation.0;
        let output_device =
            device_allocations
                .get(output_index)
                .ok_or(VkFftError::InvalidKernelIr(
                    "Level Zero program is missing its output device allocation",
                ))?;
        let output_byte_len = prepared
            .allocations
            .get(output_index)
            .ok_or(VkFftError::InvalidKernelIr(
                "Level Zero host program is missing its output allocation",
            ))?
            .byte_len;
        let output_host_slot =
            host_allocations
                .get_mut(output_index)
                .ok_or(VkFftError::InvalidKernelIr(
                    "Level Zero program is missing its output host-allocation slot",
                ))?;
        if output_host_slot.is_none() {
            *output_host_slot = Some(self.allocate_host(output_byte_len)?);
        }
        let output_host = output_host_slot
            .as_ref()
            .expect("Level Zero output host allocation initialized above");
        check_level_zero(
            unsafe {
                (fns.command_list_append_memory_copy)(
                    command_list.handle,
                    output_host.ptr,
                    output_device.ptr.cast_const(),
                    output_byte_len,
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                )
            },
            "zeCommandListAppendMemoryCopy(DtoH)",
        )?;
        check_level_zero(
            unsafe { (fns.command_list_close)(command_list.handle) },
            "zeCommandListClose",
        )?;
        let lists = [command_list.handle];
        check_level_zero(
            unsafe {
                (fns.command_queue_execute_command_lists)(
                    queue.handle,
                    1,
                    lists.as_ptr(),
                    ptr::null_mut(),
                )
            },
            "zeCommandQueueExecuteCommandLists",
        )?;
        self.active_submissions.fetch_add(1, Ordering::AcqRel);
        Ok(LevelZeroPendingProgram {
            queue,
            context: self,
            prepared,
            device_allocations,
            host_allocations,
            pending_luts,
            _command_list: command_list,
            _kernels: kernels,
            completed: false,
        })
    }

    fn recycle_command_list(&self, command_list: &mut LevelZeroCommandListGuard<'_>) {
        if command_list.handle.is_null() {
            return;
        }
        let reset = unsafe { (self.runtime.fns.command_list_reset)(command_list.handle) };
        if reset != ZE_RESULT_SUCCESS {
            return;
        }
        let Ok(mut pool) = self.command_list_pool.lock() else {
            return;
        };
        if pool.len() >= LEVEL_ZERO_COMMAND_LIST_POOL_MAX_INSTANCES {
            return;
        }
        pool.push(command_list.relinquish());
    }

    fn recycle_kernel(&self, kernel: &mut LevelZeroLoadedKernel<'_>) -> Result<()> {
        let Some(key) = kernel.pool_key.clone() else {
            return Ok(());
        };
        let mut pool = self.kernel_pool.lock().map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero kernel pool lock is poisoned",
            )
        })?;
        if pool.len() >= LEVEL_ZERO_KERNEL_POOL_MAX_INSTANCES {
            return Ok(());
        }
        let handle = kernel.relinquish_kernel();
        debug_assert!(pool.insert_if_capacity(key, handle));
        Ok(())
    }

    fn take_queue(&self) -> Result<LevelZeroOwnedQueue<'_>> {
        if let Some(handle) = self
            .queue_pool
            .lock()
            .map_err(|_| {
                runtime_error(Backend::LevelZero, "Level Zero queue pool lock is poisoned")
            })?
            .pop()
        {
            return Ok(LevelZeroOwnedQueue {
                fns: &self.runtime.fns,
                handle,
                owned: true,
            });
        }
        let desc = ZeCommandQueueDesc::compute(self.compute_queue_group_ordinal);
        let mut handle = ptr::null_mut();
        let result = unsafe {
            (self.runtime.fns.command_queue_create)(
                self.context,
                self.selection.device,
                &desc,
                &mut handle,
            )
        };
        if result != ZE_RESULT_SUCCESS || handle.is_null() {
            if !handle.is_null() {
                unsafe {
                    (self.runtime.fns.command_queue_destroy)(handle);
                }
            }
            if result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeCommandQueueCreate succeeded with a null queue handle",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeCommandQueueCreate returned ze_result_t 0x{:08x}",
                    result as u32
                ),
            ));
        }
        Ok(LevelZeroOwnedQueue {
            fns: &self.runtime.fns,
            handle,
            owned: true,
        })
    }

    fn allocate_device(&self, bytes: usize) -> Result<LevelZeroAllocation<'_>> {
        if let Some(ptr) = self
            .device_buffer_pool
            .lock()
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero device buffer pool lock is poisoned",
                )
            })?
            .get_mut(&bytes)
            .and_then(Vec::pop)
        {
            return Ok(LevelZeroAllocation {
                fns: &self.runtime.fns,
                context: self.context,
                ptr,
                bytes,
                owned: true,
            });
        }
        let desc = ZeDeviceMemAllocDesc::default_runtime();
        let mut allocation = ptr::null_mut();
        let allocation_result = unsafe {
            (self.runtime.fns.mem_alloc_device)(
                self.context,
                &desc,
                bytes,
                64,
                self.selection.device,
                &mut allocation,
            )
        };
        if allocation_result != ZE_RESULT_SUCCESS || allocation.is_null() {
            if !allocation.is_null() {
                unsafe {
                    (self.runtime.fns.mem_free)(self.context, allocation);
                }
            }
            if allocation_result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeMemAllocDevice succeeded with a null pointer",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeMemAllocDevice returned ze_result_t 0x{:08x}",
                    allocation_result as u32
                ),
            ));
        }
        Ok(LevelZeroAllocation {
            fns: &self.runtime.fns,
            context: self.context,
            ptr: allocation,
            bytes,
            owned: true,
        })
    }

    fn allocate_host(&self, bytes: usize) -> Result<LevelZeroAllocation<'_>> {
        if let Some(ptr) = self
            .host_buffer_pool
            .lock()
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero host buffer pool lock is poisoned",
                )
            })?
            .get_mut(&bytes)
            .and_then(Vec::pop)
        {
            return Ok(LevelZeroAllocation {
                fns: &self.runtime.fns,
                context: self.context,
                ptr,
                bytes,
                owned: true,
            });
        }
        let desc = ZeHostMemAllocDesc::default_runtime();
        let mut allocation = ptr::null_mut();
        let allocation_result = unsafe {
            (self.runtime.fns.mem_alloc_host)(self.context, &desc, bytes, 64, &mut allocation)
        };
        if allocation_result != ZE_RESULT_SUCCESS || allocation.is_null() {
            if !allocation.is_null() {
                unsafe {
                    (self.runtime.fns.mem_free)(self.context, allocation);
                }
            }
            if allocation_result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeMemAllocHost succeeded with a null pointer",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeMemAllocHost returned ze_result_t 0x{:08x}",
                    allocation_result as u32
                ),
            ));
        }
        Ok(LevelZeroAllocation {
            fns: &self.runtime.fns,
            context: self.context,
            ptr: allocation,
            bytes,
            owned: true,
        })
    }

    pub fn clear_runtime_caches(&self) -> Result<()> {
        let active = self.active_submissions.load(Ordering::Acquire);
        if active != 0 {
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "cannot clear Level Zero runtime caches while {active} submission(s) are in flight"
                ),
            ));
        }
        let mut device_buffers = self.device_buffer_pool.lock().map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero device buffer pool lock is poisoned",
            )
        })?;
        for ptr in device_buffers.drain().flat_map(|(_, buffers)| buffers) {
            check_level_zero(
                unsafe { (self.runtime.fns.mem_free)(self.context, ptr) },
                "zeMemFree(device pool)",
            )?;
        }
        drop(device_buffers);
        let mut host_buffers = self.host_buffer_pool.lock().map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero host buffer pool lock is poisoned",
            )
        })?;
        for ptr in host_buffers.drain().flat_map(|(_, buffers)| buffers) {
            check_level_zero(
                unsafe { (self.runtime.fns.mem_free)(self.context, ptr) },
                "zeMemFree(host pool)",
            )?;
        }
        drop(host_buffers);
        let mut luts = self.lut_cache_lock()?;
        for ptr in luts.drain().map(|(_, ptr)| ptr) {
            check_level_zero(
                unsafe { (self.runtime.fns.mem_free)(self.context, ptr) },
                "zeMemFree(LUT cache)",
            )?;
        }
        drop(luts);
        let mut queues = self.queue_pool.lock().map_err(|_| {
            runtime_error(Backend::LevelZero, "Level Zero queue pool lock is poisoned")
        })?;
        for queue in queues.drain(..) {
            check_level_zero(
                unsafe { (self.runtime.fns.command_queue_destroy)(queue) },
                "zeCommandQueueDestroy(pool)",
            )?;
        }
        drop(queues);
        let mut command_lists = self.command_list_pool.lock().map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero command-list pool lock is poisoned",
            )
        })?;
        for command_list in command_lists.drain(..) {
            check_level_zero(
                unsafe { (self.runtime.fns.command_list_destroy)(command_list) },
                "zeCommandListDestroy(pool)",
            )?;
        }
        drop(command_lists);
        let mut kernels = self.kernel_pool.lock().map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero kernel pool lock is poisoned",
            )
        })?;
        for kernel in kernels.drain_kernels() {
            check_level_zero(
                unsafe { (self.runtime.fns.kernel_destroy)(kernel) },
                "zeKernelDestroy(pool)",
            )?;
        }
        drop(kernels);
        let mut modules = self.module_cache.lock().map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "Level Zero module cache lock is poisoned",
            )
        })?;
        let mut first_module_error = None;
        for module in modules.drain_modules() {
            let result = unsafe { (self.runtime.fns.module_destroy)(module) };
            if result != ZE_RESULT_SUCCESS && first_module_error.is_none() {
                first_module_error = Some(result);
            }
        }
        if let Some(result) = first_module_error {
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeModuleDestroy(cache) returned ze_result_t 0x{:08x}",
                    result as u32
                ),
            ));
        }
        Ok(())
    }

    fn create_command_list(&self) -> Result<LevelZeroCommandListGuard<'_>> {
        if let Some(handle) = self
            .command_list_pool
            .lock()
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero command-list pool lock is poisoned",
                )
            })?
            .pop()
        {
            return Ok(LevelZeroCommandListGuard {
                fns: &self.runtime.fns,
                handle,
                owned: true,
            });
        }
        let desc = ZeCommandListDesc::compute(self.compute_queue_group_ordinal);
        let mut handle = ptr::null_mut();
        let list_result = unsafe {
            (self.runtime.fns.command_list_create)(
                self.context,
                self.selection.device,
                &desc,
                &mut handle,
            )
        };
        if list_result != ZE_RESULT_SUCCESS || handle.is_null() {
            if !handle.is_null() {
                unsafe {
                    (self.runtime.fns.command_list_destroy)(handle);
                }
            }
            if list_result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeCommandListCreate succeeded with a null handle",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeCommandListCreate returned ze_result_t 0x{:08x}",
                    list_result as u32
                ),
            ));
        }
        Ok(LevelZeroCommandListGuard {
            fns: &self.runtime.fns,
            handle,
            owned: true,
        })
    }

    fn create_kernel(
        &self,
        module_format: c_uint,
        module_bytes: &[u8],
        source_key: &str,
        entry_point: &str,
        workgroup: crate::kernel_ir::WorkgroupSize,
    ) -> Result<LevelZeroLoadedKernel<'_>> {
        if module_format == ZE_MODULE_FORMAT_IL_SPIRV
            && (module_bytes.len() < 20 || !module_bytes.len().is_multiple_of(4))
        {
            return Err(VkFftError::ShaderCompilation(
                "Level Zero module received malformed SPIR-V".to_owned(),
            ));
        }
        if module_bytes.is_empty() {
            return Err(VkFftError::ShaderCompilation(
                "Level Zero module received empty compiler output".to_owned(),
            ));
        }
        let mut module_key = Vec::with_capacity(4 + module_bytes.len());
        module_key.extend_from_slice(&module_format.to_le_bytes());
        module_key.extend_from_slice(module_bytes);
        let entry = CString::new(entry_point).map_err(|_| {
            VkFftError::ShaderCompilation(
                "Level Zero kernel entry point contains an interior NUL byte".to_owned(),
            )
        })?;

        let cached_module = self
            .module_cache
            .lock()
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero module cache lock is poisoned",
                )
            })?
            .get(&module_key);
        let (module, owns_module) = if let Some(module) = cached_module {
            (module, false)
        } else {
            let module = self.create_module(module_format, module_bytes)?;
            let mut cache = self.module_cache.lock().map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero module cache lock is poisoned",
                )
            })?;
            if cache.insert_if_capacity(&module_key, module) {
                (module, false)
            } else if let Some(existing) = cache.get(&module_key) {
                check_level_zero(
                    unsafe { (self.runtime.fns.module_destroy)(module) },
                    "zeModuleDestroy(duplicate cache module)",
                )?;
                (existing, false)
            } else {
                (module, true)
            }
        };

        let pool_key = (!owns_module).then(|| LevelZeroKernelPoolKey {
            source: source_key.to_owned(),
            entry_point: entry_point.to_owned(),
        });
        if let Some(key) = pool_key.as_ref() {
            let pooled = self
                .kernel_pool
                .lock()
                .map_err(|_| {
                    runtime_error(
                        Backend::LevelZero,
                        "Level Zero kernel pool lock is poisoned",
                    )
                })?
                .take(key);
            if let Some(kernel) = pooled {
                let loaded = LevelZeroLoadedKernel {
                    fns: &self.runtime.fns,
                    module,
                    kernel,
                    owns_module: false,
                    owns_kernel: true,
                    pool_key,
                };
                check_level_zero(
                    unsafe {
                        (self.runtime.fns.kernel_set_group_size)(
                            loaded.kernel,
                            workgroup.x,
                            workgroup.y,
                            workgroup.z,
                        )
                    },
                    "zeKernelSetGroupSize(pooled)",
                )?;
                return Ok(loaded);
            }
        }

        self.create_kernel_from_module(module, owns_module, pool_key, &entry, workgroup)
    }

    fn compiled_kernel_resource_metrics(
        &self,
        kernel: ZeKernelHandle,
    ) -> Result<Option<NativeCompiledResourceMetrics>> {
        let Some(kernel_get_properties) = self.runtime.fns.kernel_get_properties else {
            return Ok(None);
        };
        let mut properties = ZeKernelProperties::query();
        check_level_zero(
            unsafe { kernel_get_properties(kernel, &mut properties) },
            "zeKernelGetProperties",
        )?;
        Ok(Some(NativeCompiledResourceMetrics::LevelZero {
            local_memory_bytes_per_workgroup: properties.local_mem_size as usize,
            private_memory_bytes_per_thread: properties.private_mem_size as usize,
            spill_memory_bytes: properties.spill_mem_size as usize,
            required_group_size: [
                properties.required_group_size_x as usize,
                properties.required_group_size_y as usize,
                properties.required_group_size_z as usize,
            ],
            required_num_subgroups: properties.required_num_subgroups as usize,
            required_subgroup_size: properties.required_subgroup_size as usize,
            max_subgroup_size: properties.max_subgroup_size as usize,
            max_num_subgroups: properties.max_num_subgroups as usize,
        }))
    }

    fn create_module(&self, module_format: c_uint, module_bytes: &[u8]) -> Result<ZeModuleHandle> {
        let build_flags = CString::new("").expect("empty CString");
        let module_desc = ZeModuleDesc {
            stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
            p_next: ptr::null(),
            format: module_format,
            input_size: module_bytes.len(),
            p_input_module: module_bytes.as_ptr(),
            p_build_flags: build_flags.as_ptr(),
            p_constants: ptr::null(),
        };
        let mut module = ptr::null_mut();
        let mut build_log = ptr::null_mut();
        let create_result = unsafe {
            (self.runtime.fns.module_create)(
                self.context,
                self.selection.device,
                &module_desc,
                &mut module,
                &mut build_log,
            )
        };
        if create_result != ZE_RESULT_SUCCESS || module.is_null() {
            let log = level_zero_module_build_log(&self.runtime.fns, build_log);
            if !build_log.is_null() {
                unsafe {
                    (self.runtime.fns.module_build_log_destroy)(build_log);
                }
            }
            if !module.is_null() {
                unsafe {
                    (self.runtime.fns.module_destroy)(module);
                }
            }
            if create_result == ZE_RESULT_SUCCESS {
                return Err(VkFftError::ShaderCompilation(
                    "zeModuleCreate succeeded with a null module handle".to_owned(),
                ));
            }
            return Err(VkFftError::ShaderCompilation(format!(
                "zeModuleCreate returned 0x{:08x}: {log}",
                create_result as u32
            )));
        }
        if !build_log.is_null() {
            unsafe {
                (self.runtime.fns.module_build_log_destroy)(build_log);
            }
        }
        Ok(module)
    }

    fn create_kernel_from_module(
        &self,
        module: ZeModuleHandle,
        owns_module: bool,
        pool_key: Option<LevelZeroKernelPoolKey>,
        entry: &CString,
        workgroup: crate::kernel_ir::WorkgroupSize,
    ) -> Result<LevelZeroLoadedKernel<'_>> {
        let kernel_desc = ZeKernelDesc {
            stype: ZE_STRUCTURE_TYPE_KERNEL_DESC,
            p_next: ptr::null(),
            flags: 0,
            p_kernel_name: entry.as_ptr(),
        };
        let mut kernel = ptr::null_mut();
        let kernel_result =
            unsafe { (self.runtime.fns.kernel_create)(module, &kernel_desc, &mut kernel) };
        if kernel_result != ZE_RESULT_SUCCESS || kernel.is_null() {
            unsafe {
                if !kernel.is_null() {
                    (self.runtime.fns.kernel_destroy)(kernel);
                }
                if owns_module {
                    (self.runtime.fns.module_destroy)(module);
                }
            }
            if kernel_result == ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeKernelCreate succeeded with a null kernel handle",
                ));
            }
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "zeKernelCreate returned ze_result_t 0x{:08x}",
                    kernel_result as u32
                ),
            ));
        }
        let loaded = LevelZeroLoadedKernel {
            fns: &self.runtime.fns,
            module,
            kernel,
            owns_module,
            owns_kernel: true,
            pool_key,
        };
        check_level_zero(
            unsafe {
                (self.runtime.fns.kernel_set_group_size)(
                    loaded.kernel,
                    workgroup.x,
                    workgroup.y,
                    workgroup.z,
                )
            },
            "zeKernelSetGroupSize",
        )?;
        Ok(loaded)
    }
}

impl Drop for LevelZeroExecutionContext {
    fn drop(&mut self) {
        let device_buffers = match self.device_buffer_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for ptr in device_buffers.drain().flat_map(|(_, buffers)| buffers) {
            unsafe {
                (self.runtime.fns.mem_free)(self.context, ptr);
            }
        }
        let host_buffers = match self.host_buffer_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for ptr in host_buffers.drain().flat_map(|(_, buffers)| buffers) {
            unsafe {
                (self.runtime.fns.mem_free)(self.context, ptr);
            }
        }
        let luts = match self.lut_cache.get_mut() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        for ptr in luts.drain().map(|(_, ptr)| ptr) {
            unsafe {
                (self.runtime.fns.mem_free)(self.context, ptr);
            }
        }
        let command_lists = match self.command_list_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for command_list in command_lists.drain(..) {
            unsafe {
                (self.runtime.fns.command_list_destroy)(command_list);
            }
        }
        let kernels = match self.kernel_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for kernel in kernels.drain_kernels() {
            unsafe {
                (self.runtime.fns.kernel_destroy)(kernel);
            }
        }
        let modules = match self.module_cache.get_mut() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        for module in modules.drain_modules() {
            unsafe {
                (self.runtime.fns.module_destroy)(module);
            }
        }
        let queues = match self.queue_pool.get_mut() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        for queue in queues.drain(..) {
            unsafe {
                (self.runtime.fns.command_queue_destroy)(queue);
            }
        }
        unsafe {
            if !self.context.is_null() {
                (self.runtime.fns.context_destroy)(self.context);
                self.context = ptr::null_mut();
            }
        }
    }
}

impl crate::backend::native_runtime::NativeProgramTicket32 for LevelZeroProgramTicket32<'_> {
    fn wait(self) -> Result<Vec<Complex32>> {
        LevelZeroProgramTicket32::wait(self)
    }
}

impl crate::backend::native_runtime::NativeProgramTicket64 for LevelZeroProgramTicket64<'_> {
    fn wait(self) -> Result<Vec<Complex64>> {
        LevelZeroProgramTicket64::wait(self)
    }
}

impl crate::backend::native_runtime::NativeAsyncRuntime for LevelZeroExecutionContext {
    type Ticket32<'a> = LevelZeroProgramTicket32<'a>;
    type Ticket64<'a> = LevelZeroProgramTicket64<'a>;

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

impl crate::backend::native_runtime::NativeRuntime for LevelZeroExecutionContext {
    fn backend(&self) -> Backend {
        Backend::LevelZero
    }

    fn device_profile(&self) -> DeviceProfile {
        LevelZeroExecutionContext::device_profile(self)
    }

    fn device_name(&self) -> &str {
        LevelZeroExecutionContext::device_name(self)
    }

    fn compiled_pass_resource_reports(
        &self,
        source: &NativeProgramSource,
    ) -> Result<Vec<NativeCompiledPassResourceReport>> {
        source.validate()?;
        if source.backend != Backend::LevelZero {
            return Err(VkFftError::InvalidKernelIr(
                "Level Zero compiled-resource reporting requires a Level Zero native program",
            ));
        }
        if self.runtime.fns.kernel_get_properties.is_none() {
            return Ok(Vec::new());
        }
        let mut reports = Vec::with_capacity(source.shaders.len());
        for (pass, shader) in source.program.passes.iter().zip(&source.shaders) {
            shader.validate()?;
            let compiled = self.compiler.compile(&shader.source)?;
            let kernel = self.create_kernel(
                compiled.format,
                &compiled.bytes,
                &shader.source,
                shader.entry_point,
                shader.workgroup_size,
            )?;
            let metrics = self
                .compiled_kernel_resource_metrics(kernel.kernel)?
                .ok_or(VkFftError::InvalidKernelIr(
                    "Level Zero kernel properties disappeared after capability detection",
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
        LevelZeroExecutionContext::execute_program_complex32(self, source, input)
    }

    fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        LevelZeroExecutionContext::execute_program_complex64(self, source, input)
    }
}

fn level_zero_ordered_allocation_ids(
    pass: &ProgramPass,
    memory_plan: &ProgramMemoryPlan,
) -> Result<Vec<ProgramAllocationId>> {
    let mut ordered = pass.bindings.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|binding| binding.binding);
    ordered
        .into_iter()
        .map(|binding| memory_plan.allocation_for(binding.resource))
        .collect()
}

fn level_zero_module_build_log(
    fns: &LevelZeroExecutionFns,
    build_log: ZeModuleBuildLogHandle,
) -> String {
    if build_log.is_null() {
        return "no Level Zero module build log available".to_owned();
    }
    let mut size = 0usize;
    if unsafe { (fns.module_build_log_get_string)(build_log, &mut size, ptr::null_mut()) }
        != ZE_RESULT_SUCCESS
        || size == 0
    {
        return "failed to query Level Zero module build log size".to_owned();
    }
    if size > LEVEL_ZERO_BUILD_LOG_LIMIT {
        return format!(
            "Level Zero module build log omitted because reported size {size} exceeds {LEVEL_ZERO_BUILD_LOG_LIMIT} bytes"
        );
    }
    let mut bytes = vec![0u8; size];
    if unsafe { (fns.module_build_log_get_string)(build_log, &mut size, bytes.as_mut_ptr().cast()) }
        != ZE_RESULT_SUCCESS
    {
        return "failed to read Level Zero module build log".to_owned();
    }
    if let Some(nul) = bytes.iter().position(|byte| *byte == 0) {
        bytes.truncate(nul);
    }
    String::from_utf8_lossy(&bytes).trim().to_owned()
}

#[derive(Debug, Clone)]
pub struct LevelZeroRuntimeAdapter {
    availability: NativeRuntimeAvailability,
    profile: DeviceProfile,
    compute_queue_group_ordinal: u32,
    compiler: LevelZeroOclocCompiler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OclocCompilerIdentity {
    pub executable_fingerprint: [u8; OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES],
    pub executable_len: u64,
}

/// Versioned persistent form of the OCLOC source-to-SPIR-V cache. The archive is
/// tied to the exact offline-compiler executable contents plus the fixed upstream
/// VkFFT porting baseline; Level Zero modules/kernels remain context-local.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OclocSpirvCacheArchive {
    pub identity: OclocCompilerIdentity,
    entries: Vec<(String, Vec<u8>)>,
}

impl OclocSpirvCacheArchive {
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn encode(&self) -> Vec<u8> {
        let commit = crate::UPSTREAM_VKFFT_COMMIT.as_bytes();
        debug_assert_eq!(commit.len(), OCLOC_SPIRV_ARCHIVE_COMMIT_BYTES);
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(OCLOC_SPIRV_ARCHIVE_MAGIC);
        bytes.extend_from_slice(&OCLOC_SPIRV_ARCHIVE_VERSION.to_le_bytes());
        bytes.extend_from_slice(commit);
        bytes.extend_from_slice(&self.identity.executable_fingerprint);
        bytes.extend_from_slice(&self.identity.executable_len.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (source, spirv) in entries {
            bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(spirv.len() as u64).to_le_bytes());
            bytes.extend_from_slice(source.as_bytes());
            bytes.extend_from_slice(spirv);
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let malformed = |message: &str| runtime_error(Backend::LevelZero, message);
        if bytes.len() < OCLOC_SPIRV_ARCHIVE_HEADER_BYTES {
            return Err(malformed("OCLOC SPIR-V archive is truncated"));
        }
        if &bytes[..8] != OCLOC_SPIRV_ARCHIVE_MAGIC {
            return Err(malformed("OCLOC SPIR-V archive magic does not match"));
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| malformed("OCLOC SPIR-V archive version is malformed"))?,
        );
        if version != OCLOC_SPIRV_ARCHIVE_VERSION {
            return Err(malformed(
                "OCLOC SPIR-V archive schema version is unsupported",
            ));
        }
        let commit_start = 12;
        let commit_end = commit_start + OCLOC_SPIRV_ARCHIVE_COMMIT_BYTES;
        if &bytes[commit_start..commit_end] != crate::UPSTREAM_VKFFT_COMMIT.as_bytes() {
            return Err(malformed(
                "OCLOC SPIR-V archive upstream commit does not match",
            ));
        }
        let fingerprint_end = commit_end + OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES;
        let executable_fingerprint = bytes[commit_end..fingerprint_end]
            .try_into()
            .map_err(|_| malformed("OCLOC SPIR-V archive compiler fingerprint is malformed"))?;
        let mut offset = fingerprint_end;
        let executable_len = ocloc_archive_u64(bytes, &mut offset, "compiler length")?;
        let entry_count = ocloc_archive_usize(bytes, &mut offset, "entry count")?;
        if entry_count > OCLOC_CACHE_MAX_ENTRIES {
            return Err(malformed(
                "OCLOC SPIR-V archive exceeds the cache entry limit",
            ));
        }
        let mut decoded = HashMap::<String, Vec<u8>>::with_capacity(entry_count);
        let mut total_bytes = 0usize;
        for _ in 0..entry_count {
            let source_len = ocloc_archive_usize(bytes, &mut offset, "source length")?;
            let spirv_len = ocloc_archive_usize(bytes, &mut offset, "SPIR-V length")?;
            total_bytes = total_bytes
                .checked_add(source_len)
                .and_then(|value| value.checked_add(spirv_len))
                .ok_or_else(|| malformed("OCLOC SPIR-V archive payload size overflows"))?;
            if total_bytes > OCLOC_CACHE_MAX_BYTES {
                return Err(malformed(
                    "OCLOC SPIR-V archive exceeds the cache byte limit",
                ));
            }
            let source = ocloc_archive_string(bytes, &mut offset, source_len, "source")?;
            if source.trim().is_empty() {
                return Err(malformed("OCLOC SPIR-V archive contains an empty source"));
            }
            let spirv = ocloc_archive_take(bytes, &mut offset, spirv_len, "SPIR-V")?.to_vec();
            validate_spirv_bytes(&spirv)?;
            if decoded.insert(source, spirv).is_some() {
                return Err(malformed(
                    "OCLOC SPIR-V archive contains duplicate source entries",
                ));
            }
        }
        if offset != bytes.len() {
            return Err(malformed("OCLOC SPIR-V archive has trailing bytes"));
        }
        let mut entries = decoded.into_iter().collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        Ok(Self {
            identity: OclocCompilerIdentity {
                executable_fingerprint,
                executable_len,
            },
            entries,
        })
    }
}

fn ocloc_archive_take<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    len: usize,
    field: &str,
) -> Result<&'a [u8]> {
    let end = offset.checked_add(len).ok_or_else(|| {
        runtime_error(
            Backend::LevelZero,
            format!("OCLOC SPIR-V archive {field} length overflows"),
        )
    })?;
    if end > bytes.len() {
        return Err(runtime_error(
            Backend::LevelZero,
            format!("OCLOC SPIR-V archive {field} is truncated"),
        ));
    }
    let value = &bytes[*offset..end];
    *offset = end;
    Ok(value)
}

fn ocloc_archive_u64(bytes: &[u8], offset: &mut usize, field: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(
        ocloc_archive_take(bytes, offset, 8, field)?
            .try_into()
            .map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    format!("OCLOC SPIR-V archive {field} is malformed"),
                )
            })?,
    ))
}

fn ocloc_archive_usize(bytes: &[u8], offset: &mut usize, field: &str) -> Result<usize> {
    usize::try_from(ocloc_archive_u64(bytes, offset, field)?).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            format!("OCLOC SPIR-V archive {field} does not fit this platform"),
        )
    })
}

fn ocloc_archive_string(
    bytes: &[u8],
    offset: &mut usize,
    len: usize,
    field: &str,
) -> Result<String> {
    String::from_utf8(ocloc_archive_take(bytes, offset, len, field)?.to_vec()).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            format!("OCLOC SPIR-V archive {field} is not valid UTF-8"),
        )
    })
}

fn ocloc_compiler_fingerprint(bytes: &[u8]) -> [u8; OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES] {
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut forward = 0xcbf29ce484222325u64;
    let mut reverse = 0x84222325cbf29ce4u64;
    for &byte in bytes {
        forward ^= u64::from(byte);
        forward = forward.wrapping_mul(FNV_PRIME);
    }
    for &byte in bytes.iter().rev() {
        reverse ^= u64::from(byte);
        reverse = reverse.wrapping_mul(FNV_PRIME);
    }
    let mut fingerprint = [0u8; OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES];
    fingerprint[..8].copy_from_slice(&forward.to_le_bytes());
    fingerprint[8..].copy_from_slice(&reverse.to_le_bytes());
    fingerprint
}

#[derive(Debug, Clone)]
struct LevelZeroOclocCompiler {
    executable: PathBuf,
    cache: Arc<Mutex<OclocSpirvCache>>,
}

#[derive(Debug)]
struct OclocSpirvCache {
    entries: HashMap<String, Vec<u8>>,
    insertion_order: VecDeque<String>,
    total_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl OclocSpirvCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            total_bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    fn get(&self, source: &str) -> Option<Vec<u8>> {
        self.entries.get(source).cloned()
    }

    fn insert(&mut self, source: &str, spirv: &[u8]) {
        let entry_bytes = source.len().saturating_add(spirv.len());
        if self.max_entries == 0 || entry_bytes > self.max_bytes {
            return;
        }
        if let Some(previous) = self.entries.remove(source) {
            self.total_bytes = self
                .total_bytes
                .saturating_sub(source.len().saturating_add(previous.len()));
            self.insertion_order.retain(|key| key != source);
        }
        while !self.insertion_order.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.total_bytes.saturating_add(entry_bytes) > self.max_bytes)
        {
            let oldest = self
                .insertion_order
                .pop_front()
                .expect("non-empty OCLOC cache order");
            if let Some(previous) = self.entries.remove(&oldest) {
                self.total_bytes = self
                    .total_bytes
                    .saturating_sub(oldest.len().saturating_add(previous.len()));
            }
        }
        if self.entries.len() < self.max_entries
            && self.total_bytes.saturating_add(entry_bytes) <= self.max_bytes
        {
            self.total_bytes = self.total_bytes.saturating_add(entry_bytes);
            self.insertion_order.push_back(source.to_owned());
            self.entries.insert(source.to_owned(), spirv.to_vec());
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OclocInvocationStyle {
    ModernCompileSubcommand,
    LegacyTopLevel,
}

#[derive(Debug)]
struct OclocWorkspace {
    path: PathBuf,
}

impl LevelZeroRuntimeAdapter {
    pub fn probe() -> NativeRuntimeAvailability {
        let compiler_available = find_executable_on_path("ocloc").is_some();
        let runtime = load_first_library(Backend::LevelZero, LEVEL_ZERO_LOADER_CANDIDATES);
        let Ok((library, library_name)) = runtime else {
            return NativeRuntimeAvailability {
                backend: Backend::LevelZero,
                loader_available: false,
                compiler_available,
                device_count: 0,
                detail: format!(
                    "Level Zero loader was not found; ocloc available={compiler_available}"
                ),
            };
        };
        match level_zero_device_count(&library) {
            Ok(device_count) => NativeRuntimeAvailability {
                backend: Backend::LevelZero,
                loader_available: true,
                compiler_available,
                device_count,
                detail: format!(
                    "{library_name} reports {device_count} Level Zero GPU device(s); ocloc available={compiler_available}"
                ),
            },
            Err(error) => NativeRuntimeAvailability {
                backend: Backend::LevelZero,
                loader_available: true,
                compiler_available,
                device_count: 0,
                detail: error.to_string(),
            },
        }
    }

    pub fn new() -> Result<Self> {
        let availability = Self::probe();
        if !availability.available() {
            return Err(unavailable(Backend::LevelZero, availability.detail.clone()));
        }
        let (library, _) = load_first_library(Backend::LevelZero, LEVEL_ZERO_LOADER_CANDIDATES)?;
        let selection = level_zero_device_selection(&library, 0)?;
        let profile = level_zero_device_profile_for_handle(&library, selection.device)?;
        let compute_queue_group_ordinal =
            level_zero_compute_queue_group_ordinal_for_handle(&library, selection.device)?;
        let compiler = LevelZeroOclocCompiler::new()?;
        Ok(Self {
            availability,
            profile,
            compute_queue_group_ordinal,
            compiler,
        })
    }

    pub const fn availability(&self) -> &NativeRuntimeAvailability {
        &self.availability
    }

    pub const fn device_profile(&self) -> DeviceProfile {
        self.profile
    }

    pub const fn compute_queue_group_ordinal(&self) -> u32 {
        self.compute_queue_group_ordinal
    }

    /// Compile the OpenCL-C-like source emitted by `LevelZeroSourceBackend` into
    /// generic SPIR-V suitable for a future `ZE_MODULE_FORMAT_IL_SPIRV` module.
    pub fn compile_opencl_c_to_spirv(&self, source: &str) -> Result<Vec<u8>> {
        self.compiler.compile_opencl_c_to_spirv(source)
    }

    pub fn ocloc_spirv_cache_archive_data(&self) -> Result<Vec<u8>> {
        self.compiler.cache_archive_data()
    }

    pub fn restore_ocloc_spirv_cache_archive(&self, encoded: &[u8]) -> Result<()> {
        self.compiler.restore_cache_archive(encoded)
    }
}

fn level_zero_device_count(library: &Library) -> Result<usize> {
    Ok(level_zero_device_selections(library)?.len())
}

fn level_zero_device_selections(library: &Library) -> Result<Vec<LevelZeroDeviceSelection>> {
    // SAFETY: signatures follow the Level Zero core API and the library remains loaded
    // for the duration of all symbol calls. Count/list races are bounded by the queried
    // capacities and null handles are rejected before any device call.
    unsafe {
        let init = library
            .get::<ZeInit>(b"zeInit\0")
            .map_err(|error| unavailable(Backend::LevelZero, format!("missing zeInit: {error}")))?;
        let driver_get = library
            .get::<ZeDriverGet>(b"zeDriverGet\0")
            .map_err(|error| {
                unavailable(Backend::LevelZero, format!("missing zeDriverGet: {error}"))
            })?;
        let device_get = library
            .get::<ZeDeviceGet>(b"zeDeviceGet\0")
            .map_err(|error| {
                unavailable(Backend::LevelZero, format!("missing zeDeviceGet: {error}"))
            })?;
        let device_get_properties = library
            .get::<ZeDeviceGetProperties>(b"zeDeviceGetProperties\0")
            .map_err(|error| {
                unavailable(
                    Backend::LevelZero,
                    format!("missing zeDeviceGetProperties: {error}"),
                )
            })?;
        let init_result = init(0);
        if init_result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeInit returned {init_result}"),
            ));
        }
        let mut driver_count = 0;
        let result = driver_get(&mut driver_count, core::ptr::null_mut());
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDriverGet(count) returned {result}"),
            ));
        }
        if driver_count == 0 {
            return Ok(Vec::new());
        }
        let mut drivers = vec![core::ptr::null_mut(); driver_count as usize];
        let result = driver_get(&mut driver_count, drivers.as_mut_ptr());
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDriverGet(list) returned {result}"),
            ));
        }
        if driver_count as usize > drivers.len() {
            return Err(runtime_error(
                Backend::LevelZero,
                "zeDriverGet returned more drivers than the requested capacity",
            ));
        }
        drivers.truncate(driver_count as usize);

        let mut selections = Vec::new();
        for driver in drivers {
            if driver.is_null() {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeDriverGet returned a null driver handle",
                ));
            }
            let mut count = 0;
            let result = device_get(driver, &mut count, core::ptr::null_mut());
            if result != ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    format!("zeDeviceGet(count) returned {result}"),
                ));
            }
            if count == 0 {
                continue;
            }
            let mut driver_devices = vec![core::ptr::null_mut(); count as usize];
            let result = device_get(driver, &mut count, driver_devices.as_mut_ptr());
            if result != ZE_RESULT_SUCCESS {
                return Err(runtime_error(
                    Backend::LevelZero,
                    format!("zeDeviceGet(list) returned {result}"),
                ));
            }
            if count as usize > driver_devices.len() {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeDeviceGet returned more devices than the requested capacity",
                ));
            }
            driver_devices.truncate(count as usize);
            if driver_devices.iter().any(|device| device.is_null()) {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "zeDeviceGet returned a null device handle",
                ));
            }
            for device in driver_devices {
                let mut properties = ZeDeviceProperties::query();
                let result = device_get_properties(device, &mut properties);
                if result != ZE_RESULT_SUCCESS {
                    return Err(runtime_error(
                        Backend::LevelZero,
                        format!("zeDeviceGetProperties returned {result}"),
                    ));
                }
                if properties.device_type == ZE_DEVICE_TYPE_GPU {
                    selections.push(LevelZeroDeviceSelection { driver, device });
                }
            }
        }
        Ok(selections)
    }
}

fn level_zero_device_selection(
    library: &Library,
    device_index: usize,
) -> Result<LevelZeroDeviceSelection> {
    level_zero_device_selections(library)?
        .get(device_index)
        .copied()
        .ok_or_else(|| {
            unavailable(
                Backend::LevelZero,
                format!("Level Zero device index {device_index} is unavailable"),
            )
        })
}

fn level_zero_device_name_for_handle(library: &Library, device: ZeDeviceHandle) -> Result<String> {
    // SAFETY: the device handle came from this live loader and the output structure
    // carries the required stype/pNext prefix.
    unsafe {
        let get_properties = library
            .get::<ZeDeviceGetProperties>(b"zeDeviceGetProperties\0")
            .map_err(|error| {
                unavailable(
                    Backend::LevelZero,
                    format!("missing zeDeviceGetProperties: {error}"),
                )
            })?;
        let mut properties = ZeDeviceProperties::query();
        let result = get_properties(device, &mut properties);
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDeviceGetProperties returned {result}"),
            ));
        }
        let end = properties
            .name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(properties.name.len());
        let name = String::from_utf8_lossy(&properties.name[..end])
            .trim()
            .to_owned();
        if name.is_empty() {
            Ok(format!("Level Zero GPU 0x{:04x}", properties.device_id))
        } else {
            Ok(name)
        }
    }
}

fn level_zero_device_profile_for_handle(
    library: &Library,
    device: ZeDeviceHandle,
) -> Result<DeviceProfile> {
    // SAFETY: the device handle was returned by this live loader instance and every
    // output structure has the ABI stype/pNext prefix required by Level Zero.
    unsafe {
        let get_properties = library
            .get::<ZeDeviceGetProperties>(b"zeDeviceGetProperties\0")
            .map_err(|error| {
                unavailable(
                    Backend::LevelZero,
                    format!("missing zeDeviceGetProperties: {error}"),
                )
            })?;
        let get_compute_properties = library
            .get::<ZeDeviceGetComputeProperties>(b"zeDeviceGetComputeProperties\0")
            .map_err(|error| {
                unavailable(
                    Backend::LevelZero,
                    format!("missing zeDeviceGetComputeProperties: {error}"),
                )
            })?;
        let get_module_properties = library
            .get::<ZeDeviceGetModuleProperties>(b"zeDeviceGetModuleProperties\0")
            .map_err(|error| {
                unavailable(
                    Backend::LevelZero,
                    format!("missing zeDeviceGetModuleProperties: {error}"),
                )
            })?;

        let mut properties = ZeDeviceProperties::query();
        let result = get_properties(device, &mut properties);
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDeviceGetProperties returned {result}"),
            ));
        }
        let mut compute = ZeDeviceComputeProperties::query();
        let result = get_compute_properties(device, &mut compute);
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDeviceGetComputeProperties returned {result}"),
            ));
        }
        let mut module = ZeDeviceModuleProperties::query();
        let result = get_module_properties(device, &mut module);
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDeviceGetModuleProperties returned {result}"),
            ));
        }
        level_zero_profile_from_properties(&properties, &compute, &module)
    }
}

fn level_zero_compute_queue_group_ordinal_for_handle(
    library: &Library,
    device: ZeDeviceHandle,
) -> Result<u32> {
    // SAFETY: the device handle was returned by this live loader. Query elements are
    // initialized with the required stype/pNext prefix before the driver writes them.
    unsafe {
        let get_queue_groups = library
            .get::<ZeDeviceGetCommandQueueGroupProperties>(
                b"zeDeviceGetCommandQueueGroupProperties\0",
            )
            .map_err(|error| {
                unavailable(
                    Backend::LevelZero,
                    format!("missing zeDeviceGetCommandQueueGroupProperties: {error}"),
                )
            })?;
        let mut count = 0;
        let result = get_queue_groups(device, &mut count, core::ptr::null_mut());
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDeviceGetCommandQueueGroupProperties(count) returned {result}"),
            ));
        }
        if count == 0 {
            return Err(unavailable(
                Backend::LevelZero,
                "Level Zero device reports no command queue groups",
            ));
        }
        let mut groups = vec![ZeCommandQueueGroupProperties::query(); count as usize];
        let result = get_queue_groups(device, &mut count, groups.as_mut_ptr());
        if result != ZE_RESULT_SUCCESS {
            return Err(runtime_error(
                Backend::LevelZero,
                format!("zeDeviceGetCommandQueueGroupProperties(list) returned {result}"),
            ));
        }
        if count as usize > groups.len() {
            return Err(runtime_error(
                Backend::LevelZero,
                "Level Zero returned more command queue groups than requested capacity",
            ));
        }
        groups.truncate(count as usize);
        select_level_zero_compute_queue_group(&groups)
    }
}

fn select_level_zero_compute_queue_group(groups: &[ZeCommandQueueGroupProperties]) -> Result<u32> {
    let ordinal = groups
        .iter()
        .position(|group| {
            group.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE != 0 && group.num_queues > 0
        })
        .ok_or_else(|| {
            unavailable(
                Backend::LevelZero,
                "Level Zero device reports no usable compute command queue group",
            )
        })?;
    u32::try_from(ordinal).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            "Level Zero command queue group ordinal does not fit u32",
        )
    })
}

fn level_zero_profile_from_properties(
    properties: &ZeDeviceProperties,
    compute: &ZeDeviceComputeProperties,
    module: &ZeDeviceModuleProperties,
) -> Result<DeviceProfile> {
    if properties.device_type != ZE_DEVICE_TYPE_GPU {
        return Err(unavailable(
            Backend::LevelZero,
            format!(
                "Level Zero device type {} is not a GPU",
                properties.device_type
            ),
        ));
    }
    let max_threads_per_block = usize::try_from(compute.max_total_group_size).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            "Level Zero maxTotalGroupSize does not fit this platform",
        )
    })?;
    let max_workgroup_size = [
        usize::try_from(compute.max_group_size_x).unwrap_or(usize::MAX),
        usize::try_from(compute.max_group_size_y).unwrap_or(usize::MAX),
        usize::try_from(compute.max_group_size_z).unwrap_or(usize::MAX),
    ];
    let shared_memory_bytes = usize::try_from(compute.max_shared_local_memory).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            "Level Zero maxSharedLocalMemory does not fit this platform",
        )
    })?;
    if max_threads_per_block == 0 || max_workgroup_size.contains(&0) || shared_memory_bytes == 0 {
        return Err(runtime_error(
            Backend::LevelZero,
            "Level Zero reported zero compute/workgroup/shared-memory limits",
        ));
    }

    let subgroup_size = usize::try_from(properties.physical_eu_simd_width).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            "Level Zero physicalEUSimdWidth does not fit this platform",
        )
    })?;
    if subgroup_size == 0 || !subgroup_size.is_power_of_two() {
        return Err(runtime_error(
            Backend::LevelZero,
            "Level Zero physicalEUSimdWidth must be a non-zero power of two",
        ));
    }
    let subgroup_count = usize::try_from(compute.num_sub_group_sizes).map_err(|_| {
        runtime_error(
            Backend::LevelZero,
            "Level Zero numSubGroupSizes does not fit this platform",
        )
    })?;
    if subgroup_count > ZE_SUBGROUPSIZE_COUNT {
        return Err(runtime_error(
            Backend::LevelZero,
            "Level Zero numSubGroupSizes exceeds the ABI array capacity",
        ));
    }
    let mut subgroup_min = subgroup_size;
    let mut subgroup_max = subgroup_size;
    if subgroup_count > 0 {
        subgroup_min = usize::MAX;
        subgroup_max = 0;
        for &raw in &compute.sub_group_sizes[..subgroup_count] {
            let size = usize::try_from(raw).map_err(|_| {
                runtime_error(
                    Backend::LevelZero,
                    "Level Zero subgroup size does not fit this platform",
                )
            })?;
            if size == 0 || !size.is_power_of_two() {
                return Err(runtime_error(
                    Backend::LevelZero,
                    "Level Zero subgroup sizes must be non-zero powers of two",
                ));
            }
            subgroup_min = subgroup_min.min(size);
            subgroup_max = subgroup_max.max(size);
        }
    }
    if module.spirv_version_supported == 0 {
        return Err(unavailable(
            Backend::LevelZero,
            "Level Zero device does not report SPIR-V module support",
        ));
    }

    let vendor = match properties.vendor_id {
        0x10de => GpuVendor::Nvidia,
        0x1002 | 0x1022 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        other => GpuVendor::Other(other),
    };
    Ok(DeviceProfile {
        backend: Backend::LevelZero,
        vendor,
        shared_memory_bytes,
        shared_memory_pow2_bytes: floor_power_of_two(shared_memory_bytes),
        max_threads_per_block,
        max_workgroup_size,
        coalesced_memory_bytes: 64,
        shared_banks: 32,
        supports_f64: module.flags & ZE_DEVICE_MODULE_FLAG_FP64 != 0,
        // These fields describe the exact physical scheduling width. The Level Zero
        // OpenCL-C source backend can lower a proof-gated Intel subgroup dialect, but
        // topology alone does not prove that this compiler/device path accepts the
        // required subgroup + shuffle extensions. Keep every executable gate disabled
        // until OCLOC/module/runtime capability probing proves that contract.
        subgroup: SubgroupProfile {
            size: subgroup_size,
            min_size: subgroup_min,
            max_size: subgroup_max,
            required_size_compute_supported: false,
            compute_supported: false,
            basic_supported: false,
            shuffle_supported: false,
            shuffle_relative_supported: false,
            compute_full_subgroups: false,
        },
    })
}

fn level_zero_subgroup_probe_candidates(physical: SubgroupProfile) -> Vec<usize> {
    if physical.size < 2
        || !physical.size.is_power_of_two()
        || physical.min_size < 2
        || !physical.min_size.is_power_of_two()
        || physical.max_size < physical.min_size
        || !physical.max_size.is_power_of_two()
        || physical.size < physical.min_size
        || physical.size > physical.max_size
    {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let mut subgroup_size = physical.max_size;
    loop {
        candidates.push(subgroup_size);
        if subgroup_size <= physical.min_size {
            break;
        }
        subgroup_size >>= 1;
    }
    if !candidates.contains(&physical.size) {
        candidates.push(physical.size);
        candidates.sort_unstable_by(|left, right| right.cmp(left));
        candidates.dedup();
    }
    candidates
}

fn proven_level_zero_subgroup_profile(
    physical: SubgroupProfile,
    subgroup_size: usize,
) -> SubgroupProfile {
    debug_assert!(subgroup_size.is_power_of_two());
    debug_assert!(subgroup_size >= physical.min_size && subgroup_size <= physical.max_size);
    SubgroupProfile {
        size: subgroup_size,
        min_size: physical.min_size,
        max_size: physical.max_size,
        required_size_compute_supported: true,
        compute_supported: true,
        basic_supported: true,
        shuffle_supported: true,
        // The production Level Zero lowering uses indexed `intel_sub_group_shuffle`;
        // no relative shuffle builtin is required by the current register exchange.
        shuffle_relative_supported: false,
        compute_full_subgroups: true,
    }
}

fn validate_level_zero_subgroup_probe_output(
    bytes: &[u8],
    subgroup_size: usize,
    local_size: usize,
) -> bool {
    if subgroup_size == 0
        || local_size == 0
        || !local_size.is_multiple_of(subgroup_size)
        || bytes.len() != local_size * 4 * core::mem::size_of::<u32>()
    {
        return false;
    }
    let subgroup_count = local_size / subgroup_size;
    let mut seen = vec![false; local_size];
    for record in bytes.chunks_exact(4 * core::mem::size_of::<u32>()) {
        let value = |offset: usize| {
            u32::from_ne_bytes(record[offset..offset + 4].try_into().expect("four bytes")) as usize
        };
        let lane = value(0);
        let subgroup = value(4);
        let width = value(8);
        let shuffled = value(12);
        if width != subgroup_size
            || lane >= subgroup_size
            || subgroup >= subgroup_count
            || shuffled != subgroup_size - 1 - lane
        {
            return false;
        }
        let index = subgroup * subgroup_size + lane;
        if seen[index] {
            return false;
        }
        seen[index] = true;
    }
    seen.into_iter().all(|value| value)
}

fn floor_power_of_two(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

impl LevelZeroOclocCompiler {
    fn compiler_identity(&self) -> Result<OclocCompilerIdentity> {
        let executable = fs::read(&self.executable).map_err(|error| {
            runtime_error(
                Backend::LevelZero,
                format!("failed to read OCLOC executable for cache identity: {error}"),
            )
        })?;
        let executable_len = u64::try_from(executable.len()).map_err(|_| {
            runtime_error(
                Backend::LevelZero,
                "OCLOC executable length does not fit the archive identity",
            )
        })?;
        Ok(OclocCompilerIdentity {
            executable_fingerprint: ocloc_compiler_fingerprint(&executable),
            executable_len,
        })
    }

    fn cache_archive_data(&self) -> Result<Vec<u8>> {
        let identity = self.compiler_identity()?;
        let cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entries = cache
            .entries
            .iter()
            .map(|(source, spirv)| (source.clone(), spirv.clone()))
            .collect::<Vec<_>>();
        Ok(OclocSpirvCacheArchive { identity, entries }.encode())
    }

    fn restore_cache_archive(&self, encoded: &[u8]) -> Result<()> {
        let archive = OclocSpirvCacheArchive::decode(encoded)?;
        let expected = self.compiler_identity()?;
        if archive.identity != expected {
            return Err(runtime_error(
                Backend::LevelZero,
                format!(
                    "OCLOC SPIR-V archive targets compiler {:?}, but the current executable reports {:?}",
                    archive.identity, expected
                ),
            ));
        }
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (source, spirv) in archive.entries {
            if let Some(existing) = cache.entries.get(&source) {
                if existing != &spirv {
                    return Err(runtime_error(
                        Backend::LevelZero,
                        "OCLOC SPIR-V archive conflicts with an existing source entry",
                    ));
                }
                continue;
            }
            cache.insert(&source, &spirv);
        }
        Ok(())
    }

    fn new() -> Result<Self> {
        let executable = find_executable_on_path("ocloc").ok_or_else(|| {
            unavailable(
                Backend::LevelZero,
                "OpenCL offline compiler `ocloc` was not found on PATH",
            )
        })?;
        Ok(Self::with_cache_limits(
            executable,
            OCLOC_CACHE_MAX_ENTRIES,
            OCLOC_CACHE_MAX_BYTES,
        ))
    }

    fn with_cache_limits(executable: PathBuf, max_entries: usize, max_bytes: usize) -> Self {
        Self {
            executable,
            cache: Arc::new(Mutex::new(OclocSpirvCache::new(max_entries, max_bytes))),
        }
    }

    #[cfg(all(test, unix))]
    fn from_executable(executable: PathBuf) -> Self {
        Self::with_cache_limits(executable, OCLOC_CACHE_MAX_ENTRIES, OCLOC_CACHE_MAX_BYTES)
    }

    #[cfg(all(test, unix))]
    fn from_executable_with_cache_limits(
        executable: PathBuf,
        max_entries: usize,
        max_bytes: usize,
    ) -> Self {
        Self::with_cache_limits(executable, max_entries, max_bytes)
    }

    fn compile_opencl_c_to_spirv(&self, source: &str) -> Result<Vec<u8>> {
        if source.trim().is_empty() {
            return Err(VkFftError::ShaderCompilation(
                "Level Zero OCLOC received empty OpenCL-C source".to_owned(),
            ));
        }
        if let Some(cached) = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(source)
        {
            return Ok(cached);
        }
        let spirv = self.compile_opencl_c_to_spirv_uncached(source)?;
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(source, &spirv);
        Ok(spirv)
    }

    fn compile_opencl_c_to_spirv_uncached(&self, source: &str) -> Result<Vec<u8>> {
        let workspace = OclocWorkspace::create()?;
        let source_path = workspace.path.join("vkfft_kernel.cl");
        let output_base = workspace.path.join("vkfft_kernel");
        let spirv_path = output_base.with_extension("spv");
        fs::write(&source_path, source).map_err(|error| {
            runtime_error(
                Backend::LevelZero,
                format!("failed to stage OCLOC source: {error}"),
            )
        })?;

        let modern = self.invoke(
            OclocInvocationStyle::ModernCompileSubcommand,
            &source_path,
            &output_base,
        )?;
        if modern.status.success() && spirv_path.is_file() {
            return read_valid_spirv(&spirv_path);
        }

        // Compute Runtime 22.x used the same options at the top level, while current
        // OCLOC documents the `compile` subcommand. Retry the legacy form only after
        // the modern form failed, keeping both diagnostics if compilation still fails.
        let _ = fs::remove_file(&spirv_path);
        let legacy = self.invoke(
            OclocInvocationStyle::LegacyTopLevel,
            &source_path,
            &output_base,
        )?;
        if legacy.status.success() && spirv_path.is_file() {
            return read_valid_spirv(&spirv_path);
        }

        Err(VkFftError::ShaderCompilation(format!(
            "Level Zero OCLOC failed; modern={} [{}]; legacy={} [{}]",
            modern.status,
            bounded_process_log(&modern),
            legacy.status,
            bounded_process_log(&legacy),
        )))
    }

    fn invoke(
        &self,
        style: OclocInvocationStyle,
        source_path: &Path,
        output_base: &Path,
    ) -> Result<Output> {
        let mut command = Command::new(&self.executable);
        if matches!(style, OclocInvocationStyle::ModernCompileSubcommand) {
            command.arg("compile");
        }
        command
            .arg("-file")
            .arg(source_path)
            .arg("-spv_only")
            .arg("-output_no_suffix")
            .arg("-output")
            .arg(output_base);
        command.output().map_err(|error| {
            unavailable(
                Backend::LevelZero,
                format!("failed to invoke {}: {error}", self.executable.display()),
            )
        })
    }
}

impl OclocWorkspace {
    fn create() -> Result<Self> {
        let root = env::temp_dir();
        for _ in 0..64 {
            let sequence = OCLOC_WORKSPACE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("vkfft-rs-ocloc-{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(runtime_error(
                        Backend::LevelZero,
                        format!("failed to create isolated OCLOC workspace: {error}"),
                    ));
                }
            }
        }
        Err(runtime_error(
            Backend::LevelZero,
            "failed to allocate a unique OCLOC workspace after 64 attempts",
        ))
    }
}

impl Drop for OclocWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn validate_spirv_bytes(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return Err(VkFftError::ShaderCompilation(format!(
            "Level Zero OCLOC produced malformed SPIR-V length {}",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("four-byte SPIR-V magic"));
    if magic != SPIRV_MAGIC {
        return Err(VkFftError::ShaderCompilation(format!(
            "Level Zero OCLOC produced invalid SPIR-V magic 0x{magic:08x}"
        )));
    }
    Ok(())
}

fn read_valid_spirv(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|error| {
        runtime_error(
            Backend::LevelZero,
            format!("failed to read OCLOC SPIR-V output: {error}"),
        )
    })?;
    validate_spirv_bytes(&bytes)?;
    Ok(bytes)
}

fn bounded_process_log(output: &Output) -> String {
    let mut combined = Vec::with_capacity(output.stdout.len() + output.stderr.len() + 1);
    combined.extend_from_slice(&output.stdout);
    if !output.stdout.is_empty() && !output.stderr.is_empty() {
        combined.push(b'\n');
    }
    combined.extend_from_slice(&output.stderr);
    let text = String::from_utf8_lossy(&combined);
    let mut bounded: String = text.chars().take(OCLOC_LOG_LIMIT).collect();
    if text.chars().count() > OCLOC_LOG_LIMIT {
        bounded.push_str("...<truncated>");
    }
    bounded.trim().to_owned()
}

fn find_executable_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join(name))
            .find(|candidate| executable_file(candidate))
    })
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NativeSourceBackend;

    #[test]
    fn level_zero_probe_reports_loader_and_compiler_independently() {
        let availability = LevelZeroRuntimeAdapter::probe();
        assert_eq!(availability.backend, Backend::LevelZero);
        if let Ok(adapter) = LevelZeroRuntimeAdapter::new() {
            assert!(adapter.availability().available());
            assert_eq!(adapter.device_profile().backend, Backend::LevelZero);
        }
    }

    #[test]
    fn level_zero_execution_context_implements_native_async_runtime_contract() {
        fn assert_async_runtime<T: crate::backend::native_runtime::NativeAsyncRuntime>() {}
        fn assert_ticket32<T: crate::backend::native_runtime::NativeProgramTicket32>() {}
        fn assert_ticket64<T: crate::backend::native_runtime::NativeProgramTicket64>() {}

        assert_async_runtime::<LevelZeroExecutionContext>();
        assert_ticket32::<LevelZeroProgramTicket32<'static>>();
        assert_ticket64::<LevelZeroProgramTicket64<'static>>();
        let _ = LevelZeroExecutionContext::submit_transform_complex32;
        let _ = LevelZeroExecutionContext::execute_transform_complex32;
        let _ = LevelZeroExecutionContext::submit_transform_complex64;
        let _ = LevelZeroExecutionContext::execute_transform_complex64;
        let _ = LevelZeroExecutionContext::submit_transform_f32;
        let _ = LevelZeroExecutionContext::submit_transform_f64;
    }

    #[test]
    fn level_zero_compiled_resource_report_matches_real_kernel_or_skip() {
        if !LevelZeroExecutionContext::probe().available() {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
        let transform = TransformIr::build(
            crate::FftConfig::new(vec![64]).with_batch_count(2),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::LevelZero)
            .lower_transform(&transform)
            .unwrap();
        let reports =
            crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                &context, &source,
            )
            .unwrap();
        if context.runtime.fns.kernel_get_properties.is_none() {
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
            let NativeCompiledResourceMetrics::LevelZero {
                local_memory_bytes_per_workgroup,
                private_memory_bytes_per_thread: _,
                spill_memory_bytes: _,
                required_group_size,
                required_num_subgroups: _,
                required_subgroup_size: _,
                max_subgroup_size,
                max_num_subgroups,
            } = report.metrics
            else {
                panic!("Level Zero compiled-resource report changed backend metric kind");
            };
            assert!(
                local_memory_bytes_per_workgroup >= shader.required_shared_memory_bytes,
                "compiled Level Zero local memory {local_memory_bytes_per_workgroup} is below typed requirement {}",
                shader.required_shared_memory_bytes
            );
            assert!(
                local_memory_bytes_per_workgroup <= context.device_profile().shared_memory_bytes
            );
            let workgroup = [
                shader.workgroup_size.x as usize,
                shader.workgroup_size.y as usize,
                shader.workgroup_size.z as usize,
            ];
            assert!(required_group_size == [0, 0, 0] || required_group_size == workgroup);
            assert!(max_subgroup_size > 0);
            assert!(max_num_subgroups > 0);
        }
    }

    #[test]
    fn level_zero_double_double_typed_facade_matches_reference_or_skip() {
        if !LevelZeroExecutionContext::probe().available() {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
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
            "Level Zero full-DD N15 mismatch on {}: {full_error:e}",
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
            "Level Zero DD/F64 N15 mismatch on {}: {f64_error:e}",
            context.device_name()
        );
    }

    #[test]
    fn level_zero_module_cache_is_bounded_and_deduplicates_spirv() {
        let mut cache = LevelZeroModuleCache::new(2);
        let spirv_a = [1u8, 2, 3, 4];
        let spirv_b = [5u8, 6, 7, 8];
        let spirv_c = [9u8, 10, 11, 12];
        let module_a = 1usize as ZeModuleHandle;
        let module_b = 2usize as ZeModuleHandle;
        let module_c = 3usize as ZeModuleHandle;

        assert!(cache.insert_if_capacity(&spirv_a, module_a));
        assert_eq!(cache.get(&spirv_a), Some(module_a));
        assert!(!cache.insert_if_capacity(&spirv_a, module_c));
        assert!(cache.insert_if_capacity(&spirv_b, module_b));
        assert!(!cache.insert_if_capacity(&spirv_c, module_c));
        assert_eq!(cache.len(), 2);

        let mut modules = cache.drain_modules().collect::<Vec<_>>();
        modules.sort_by_key(|module| *module as usize);
        assert_eq!(modules, vec![module_a, module_b]);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn level_zero_kernel_pool_is_bounded_and_reuses_matching_key() {
        let mut pool = LevelZeroKernelPool::new(2);
        let key_a = LevelZeroKernelPoolKey {
            source: "source-a".to_owned(),
            entry_point: "kernel-a".to_owned(),
        };
        let key_b = LevelZeroKernelPoolKey {
            source: "source-b".to_owned(),
            entry_point: "kernel-b".to_owned(),
        };
        let key_c = LevelZeroKernelPoolKey {
            source: "source-c".to_owned(),
            entry_point: "kernel-c".to_owned(),
        };
        let kernel_a = 1usize as ZeKernelHandle;
        let kernel_b = 2usize as ZeKernelHandle;
        let kernel_c = 3usize as ZeKernelHandle;

        assert!(pool.insert_if_capacity(key_a.clone(), kernel_a));
        assert!(pool.insert_if_capacity(key_b.clone(), kernel_b));
        assert!(!pool.insert_if_capacity(key_c, kernel_c));
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.take(&key_a), Some(kernel_a));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.take(&key_a), None);
        assert_eq!(pool.take(&key_b), Some(kernel_b));
        assert_eq!(pool.len(), 0);

        assert!(pool.insert_if_capacity(key_a, kernel_a));
        assert!(pool.insert_if_capacity(key_b, kernel_b));
        let mut drained = pool
            .drain_kernels()
            .into_iter()
            .map(|kernel| kernel as usize)
            .collect::<Vec<_>>();
        drained.sort_unstable();
        assert_eq!(drained, vec![1, 2]);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn level_zero_module_cache_reuses_repeat_transform_or_skip() {
        if !LevelZeroExecutionContext::probe().available() {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
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
                Complex32::new((0.13 * x).sin(), (0.07 * x).cos())
            })
            .collect::<Vec<_>>();

        context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        let first = context.cached_module_count().unwrap();
        assert!(first > 0);
        context
            .execute_transform_f32(&ir, NativeTransformInput32::Complex(&input))
            .unwrap();
        assert_eq!(context.cached_module_count().unwrap(), first);
        context.clear_runtime_caches().unwrap();
        assert_eq!(context.cached_module_count().unwrap(), 0);
    }

    #[test]
    fn level_zero_runtime_caches_modules_luts_and_transients_or_skip() {
        if !LevelZeroExecutionContext::probe().available() {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
        assert_eq!(context.cached_module_count().unwrap(), 0);
        assert_eq!(context.pooled_kernel_instance_count().unwrap(), 0);
        assert_eq!(context.pooled_command_list_count().unwrap(), 0);
        assert_eq!(context.cached_lut_count().unwrap(), 0);
        assert_eq!(context.cached_device_buffer_count().unwrap(), 0);
        assert_eq!(context.cached_host_buffer_count().unwrap(), 0);

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
        let source = NativeSourceBackend::new(Backend::LevelZero)
            .lower_transform(&ir)
            .unwrap();

        let first = context.execute_program_complex32(&source, &input).unwrap();
        let first_counts = (
            context.cached_module_count().unwrap(),
            context.pooled_kernel_instance_count().unwrap(),
            context.pooled_command_list_count().unwrap(),
            context.cached_lut_count().unwrap(),
            context.cached_device_buffer_count().unwrap(),
            context.cached_host_buffer_count().unwrap(),
        );
        assert!(first_counts.0 > 0);
        assert!(first_counts.1 > 0);
        assert!(first_counts.2 > 0);
        assert!(first_counts.3 > 0);
        assert!(first_counts.4 > 0);
        assert!(first_counts.5 > 0);

        let second = context.execute_program_complex32(&source, &input).unwrap();
        let second_counts = (
            context.cached_module_count().unwrap(),
            context.pooled_kernel_instance_count().unwrap(),
            context.pooled_command_list_count().unwrap(),
            context.cached_lut_count().unwrap(),
            context.cached_device_buffer_count().unwrap(),
            context.cached_host_buffer_count().unwrap(),
        );
        assert_eq!(second, first);
        assert_eq!(second_counts, first_counts);

        context.clear_runtime_caches().unwrap();
        assert_eq!(context.cached_module_count().unwrap(), 0);
        assert_eq!(context.pooled_kernel_instance_count().unwrap(), 0);
        assert_eq!(context.pooled_command_list_count().unwrap(), 0);
        assert_eq!(context.cached_lut_count().unwrap(), 0);
        assert_eq!(context.cached_device_buffer_count().unwrap(), 0);
        assert_eq!(context.cached_host_buffer_count().unwrap(), 0);
        assert_eq!(context.pooled_queue_count().unwrap(), 0);
    }

    #[test]
    fn level_zero_two_high_level_tickets_can_be_in_flight_before_wait_or_skip() {
        if !LevelZeroExecutionContext::probe().available() {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
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
                panic!("async Level Zero C2C returned real output");
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
                "Level Zero async error {error:e}"
            );
        }
        assert!(context.pooled_queue_count().unwrap() >= 2);
        assert!(context.cached_device_buffer_count().unwrap() > 0);
        assert!(context.cached_host_buffer_count().unwrap() > 0);
    }

    #[cfg(feature = "opencl-runtime")]
    #[test]
    fn level_zero_intel_opencl_native_fallback_stockham_matches_cpu_or_skip() {
        if LevelZeroOclocCompiler::new().is_ok() {
            return;
        }
        let availability = LevelZeroExecutionContext::probe();
        if !availability.available()
            || !availability
                .detail
                .contains("Intel OpenCL native compiler fallback")
        {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
        assert_eq!(context.device_profile().vendor, GpuVendor::Intel);

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
            panic!("Level Zero native-fallback C2C transform returned a real output");
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
            .fold(0.0f64, f64::max);
        assert!(
            error < 2.0e-3 * length as f64,
            "Level Zero Intel OpenCL native-fallback F32 error {error:e}"
        );
    }

    #[test]
    fn level_zero_intel_required_width_executes_shuffle_fft_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_SUBGROUP_RUNTIME").is_some();
        let availability = LevelZeroExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Level Zero subgroup gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
        let profile = context.device_profile();
        if profile.vendor != GpuVendor::Intel
            || !profile.subgroup.supports_full_subgroup_shuffle_compute()
        {
            assert!(
                !require,
                "strict Intel Level Zero subgroup runtime proof failed: profile={:?}",
                profile.subgroup
            );
            return;
        }
        let required = profile
            .subgroup
            .required_compute_subgroup_size()
            .expect("proven Level Zero subgroup profile must request its exact width");

        let required_usize =
            usize::try_from(required).expect("Level Zero subgroup width fits usize");
        let length = if required == 32 {
            152
        } else {
            required_usize
                .checked_mul(8)
                .expect("Level Zero subgroup witness length")
        };
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            profile,
        )
        .unwrap();
        let source = NativeSourceBackend::new(Backend::LevelZero)
            .lower_transform(&ir)
            .unwrap();
        let subgroup_shaders = source
            .shaders
            .iter()
            .filter(|shader| shader.source.contains("intel_sub_group_shuffle("))
            .collect::<Vec<_>>();
        assert!(!subgroup_shaders.is_empty());
        for shader in subgroup_shaders {
            assert!(
                shader
                    .source
                    .contains("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable")
            );
            assert!(shader.source.contains(&format!(
                "__attribute__((intel_reqd_sub_group_size({required}))) __kernel"
            )));
        }

        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.071 * x).sin() + 0.001 * x, (0.037 * x).cos())
            })
            .collect::<Vec<_>>();
        let input64 = input
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let expected = ir.execute_complex_reference(&input64).unwrap();
        let actual = context.execute_transform_complex32(&ir, &input).unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
            })
            .fold(0.0f64, f64::max);
        assert!(
            max_error <= 2.0e-3,
            "Intel Level Zero required-subgroup N{length} error {max_error:e} on {}",
            context.device_name()
        );
    }

    #[test]
    fn level_zero_stockham_matches_cpu_when_device_available() {
        if !LevelZeroExecutionContext::probe().available() {
            return;
        }
        let context = LevelZeroExecutionContext::new(0).unwrap();
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
            panic!("Level Zero C2C transform returned a real output");
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
        assert!(
            error < 2.0e-3 * length as f64,
            "Level Zero F32 error {error}"
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn level_zero_property_and_execution_ffi_layout_matches_v1_18_core_abi() {
        use std::mem::{offset_of, size_of};

        assert_eq!(size_of::<ZeDeviceProperties>(), 368);
        assert_eq!(offset_of!(ZeDeviceProperties, p_next), 8);
        assert_eq!(offset_of!(ZeDeviceProperties, vendor_id), 20);
        assert_eq!(offset_of!(ZeDeviceProperties, max_mem_alloc_size), 40);
        assert_eq!(offset_of!(ZeDeviceProperties, physical_eu_simd_width), 60);
        assert_eq!(offset_of!(ZeDeviceProperties, timer_resolution), 80);
        assert_eq!(offset_of!(ZeDeviceProperties, uuid), 96);
        assert_eq!(offset_of!(ZeDeviceProperties, name), 112);

        assert_eq!(size_of::<ZeDeviceComputeProperties>(), 88);
        assert_eq!(offset_of!(ZeDeviceComputeProperties, p_next), 8);
        assert_eq!(
            offset_of!(ZeDeviceComputeProperties, max_total_group_size),
            16
        );
        assert_eq!(
            offset_of!(ZeDeviceComputeProperties, max_shared_local_memory),
            44
        );
        assert_eq!(
            offset_of!(ZeDeviceComputeProperties, num_sub_group_sizes),
            48
        );
        assert_eq!(offset_of!(ZeDeviceComputeProperties, sub_group_sizes), 52);

        assert_eq!(size_of::<ZeDeviceModuleProperties>(), 64);
        assert_eq!(offset_of!(ZeDeviceModuleProperties, p_next), 8);
        assert_eq!(
            offset_of!(ZeDeviceModuleProperties, spirv_version_supported),
            16
        );
        assert_eq!(offset_of!(ZeDeviceModuleProperties, flags), 20);
        assert_eq!(offset_of!(ZeDeviceModuleProperties, fp64flags), 32);
        assert_eq!(
            offset_of!(ZeDeviceModuleProperties, native_kernel_supported),
            44
        );

        assert_eq!(size_of::<ZeCommandQueueGroupProperties>(), 40);
        assert_eq!(offset_of!(ZeCommandQueueGroupProperties, p_next), 8);
        assert_eq!(offset_of!(ZeCommandQueueGroupProperties, flags), 16);
        assert_eq!(
            offset_of!(ZeCommandQueueGroupProperties, max_memory_fill_pattern_size),
            24
        );
        assert_eq!(offset_of!(ZeCommandQueueGroupProperties, num_queues), 32);

        assert_eq!(size_of::<ZeContextDesc>(), 24);
        assert_eq!(offset_of!(ZeContextDesc, p_next), 8);
        assert_eq!(offset_of!(ZeContextDesc, flags), 16);
        assert_eq!(size_of::<ZeDeviceMemAllocDesc>(), 24);
        assert_eq!(offset_of!(ZeDeviceMemAllocDesc, ordinal), 20);
        assert_eq!(size_of::<ZeHostMemAllocDesc>(), 24);
        assert_eq!(size_of::<ZeModuleDesc>(), 56);
        assert_eq!(offset_of!(ZeModuleDesc, format), 16);
        assert_eq!(offset_of!(ZeModuleDesc, input_size), 24);
        assert_eq!(offset_of!(ZeModuleDesc, p_input_module), 32);
        assert_eq!(offset_of!(ZeModuleDesc, p_constants), 48);
        assert_eq!(size_of::<ZeKernelDesc>(), 32);
        assert_eq!(offset_of!(ZeKernelDesc, p_kernel_name), 24);
        assert_eq!(size_of::<ZeKernelUuid>(), 32);
        assert_eq!(size_of::<ZeKernelProperties>(), 96);
        assert_eq!(offset_of!(ZeKernelProperties, p_next), 8);
        assert_eq!(offset_of!(ZeKernelProperties, num_kernel_args), 16);
        assert_eq!(offset_of!(ZeKernelProperties, required_group_size_x), 20);
        assert_eq!(offset_of!(ZeKernelProperties, required_num_subgroups), 32);
        assert_eq!(offset_of!(ZeKernelProperties, max_subgroup_size), 40);
        assert_eq!(offset_of!(ZeKernelProperties, local_mem_size), 48);
        assert_eq!(offset_of!(ZeKernelProperties, private_mem_size), 52);
        assert_eq!(offset_of!(ZeKernelProperties, spill_mem_size), 56);
        assert_eq!(offset_of!(ZeKernelProperties, uuid), 60);
        assert_eq!(size_of::<ZeCommandQueueDesc>(), 40);
        assert_eq!(offset_of!(ZeCommandQueueDesc, ordinal), 16);
        assert_eq!(offset_of!(ZeCommandQueueDesc, priority), 32);
        assert_eq!(size_of::<ZeCommandListDesc>(), 24);
        assert_eq!(
            offset_of!(ZeCommandListDesc, command_queue_group_ordinal),
            16
        );
        assert_eq!(size_of::<ZeGroupCount>(), 12);
        assert_eq!(ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0x0d);
        assert_eq!(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, 0x0e);
        assert_eq!(ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, 0x0f);
        assert_eq!(ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, 0x15);
        assert_eq!(ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, 0x16);
        assert_eq!(ZE_STRUCTURE_TYPE_MODULE_DESC, 0x1b);
        assert_eq!(ZE_STRUCTURE_TYPE_KERNEL_DESC, 0x1d);
        assert_eq!(ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, 0x1e);
        assert!(LEVEL_ZERO_LOADER_CANDIDATES.contains(&"ze_loader.dll"));
        assert!(LEVEL_ZERO_LOADER_CANDIDATES.contains(&"libze_loader.so.1"));
        assert_eq!(ZE_MODULE_FORMAT_IL_SPIRV, 0);
    }

    #[test]
    fn level_zero_exact_properties_build_scheduler_profile_without_enabling_subgroups() {
        let mut properties = ZeDeviceProperties::query();
        properties.device_type = ZE_DEVICE_TYPE_GPU;
        properties.vendor_id = 0x8086;
        properties.physical_eu_simd_width = 16;
        let mut compute = ZeDeviceComputeProperties::query();
        compute.max_total_group_size = 1024;
        compute.max_group_size_x = 1024;
        compute.max_group_size_y = 1024;
        compute.max_group_size_z = 64;
        compute.max_shared_local_memory = 65_536;
        compute.num_sub_group_sizes = 3;
        compute.sub_group_sizes[..3].copy_from_slice(&[8, 16, 32]);
        let mut module = ZeDeviceModuleProperties::query();
        module.spirv_version_supported = 0x0001_0600;
        module.flags = ZE_DEVICE_MODULE_FLAG_FP64;
        module.fp64flags = 1;

        let profile = level_zero_profile_from_properties(&properties, &compute, &module).unwrap();
        assert_eq!(profile.backend, Backend::LevelZero);
        assert_eq!(profile.vendor, GpuVendor::Intel);
        assert_eq!(profile.shared_memory_bytes, 65_536);
        assert_eq!(profile.shared_memory_pow2_bytes, 65_536);
        assert_eq!(profile.max_threads_per_block, 1024);
        assert_eq!(profile.max_workgroup_size, [1024, 1024, 64]);
        assert_eq!(profile.coalesced_memory_bytes, 64);
        assert!(profile.supports_f64);
        assert_eq!(profile.subgroup.size, 16);
        assert_eq!(profile.subgroup.min_size, 8);
        assert_eq!(profile.subgroup.max_size, 32);
        assert!(!profile.subgroup.compute_supported);
        assert!(!profile.subgroup.shuffle_supported);
        assert!(!profile.subgroup.supports_full_subgroup_shuffle_compute());
        let policy = crate::scheduler::plan_gpu_scheduler_policy(crate::Precision::F32, profile)
            .expect("Intel Level Zero policy");
        assert_eq!(policy.subgroup_width, 16);
        assert_eq!(policy.register_boost, 1);
    }

    #[test]
    fn level_zero_subgroup_proof_promotes_only_executable_capabilities() {
        let physical = SubgroupProfile {
            size: 16,
            min_size: 8,
            max_size: 32,
            required_size_compute_supported: false,
            compute_supported: false,
            basic_supported: false,
            shuffle_supported: false,
            shuffle_relative_supported: false,
            compute_full_subgroups: false,
        };
        assert_eq!(
            level_zero_subgroup_probe_candidates(physical),
            vec![32, 16, 8]
        );
        let proven = proven_level_zero_subgroup_profile(physical, 32);
        assert_eq!(proven.size, 32);
        assert_eq!(proven.min_size, 8);
        assert_eq!(proven.max_size, 32);
        assert!(proven.required_size_compute_supported);
        assert!(proven.compute_supported);
        assert!(proven.basic_supported);
        assert!(proven.shuffle_supported);
        assert!(!proven.shuffle_relative_supported);
        assert!(proven.compute_full_subgroups);
        assert!(proven.supports_full_subgroup_shuffle_compute());
        assert_eq!(proven.required_compute_subgroup_size(), Some(32));

        let mut profile = DeviceProfile::generic(Backend::LevelZero, GpuVendor::Intel);
        profile.subgroup = proven;
        let policy = crate::scheduler::plan_gpu_scheduler_policy(crate::Precision::F32, profile)
            .expect("proven Intel Level Zero scheduler policy");
        assert_eq!(policy.subgroup_width, 32);
    }

    #[test]
    fn level_zero_subgroup_probe_output_validation_rejects_partial_or_wrong_shuffle() {
        let encode = |records: &[[u32; 4]]| {
            records
                .iter()
                .flat_map(|record| record.iter().flat_map(|value| value.to_ne_bytes()))
                .collect::<Vec<_>>()
        };
        let good = encode(&[[0, 0, 2, 1], [1, 0, 2, 0], [0, 1, 2, 1], [1, 1, 2, 0]]);
        assert!(validate_level_zero_subgroup_probe_output(&good, 2, 4));

        let duplicate_lane = encode(&[[0, 0, 2, 1], [0, 0, 2, 1], [0, 1, 2, 1], [1, 1, 2, 0]]);
        assert!(!validate_level_zero_subgroup_probe_output(
            &duplicate_lane,
            2,
            4
        ));
        let wrong_shuffle = encode(&[[0, 0, 2, 0], [1, 0, 2, 0], [0, 1, 2, 1], [1, 1, 2, 0]]);
        assert!(!validate_level_zero_subgroup_probe_output(
            &wrong_shuffle,
            2,
            4
        ));
        assert!(!validate_level_zero_subgroup_probe_output(
            &good[..48],
            2,
            4
        ));
    }

    #[test]
    fn level_zero_profile_tracks_fp64_flag_and_non_power_of_two_shared_capacity() {
        let mut properties = ZeDeviceProperties::query();
        properties.device_type = ZE_DEVICE_TYPE_GPU;
        properties.vendor_id = 0x8086;
        properties.physical_eu_simd_width = 32;
        let mut compute = ZeDeviceComputeProperties::query();
        compute.max_total_group_size = 512;
        compute.max_group_size_x = 512;
        compute.max_group_size_y = 256;
        compute.max_group_size_z = 64;
        compute.max_shared_local_memory = 48 * 1024;
        let mut module = ZeDeviceModuleProperties::query();
        module.spirv_version_supported = 0x0001_0300;

        let profile = level_zero_profile_from_properties(&properties, &compute, &module).unwrap();
        assert_eq!(profile.shared_memory_bytes, 48 * 1024);
        assert_eq!(profile.shared_memory_pow2_bytes, 32 * 1024);
        assert!(!profile.supports_f64);
        assert_eq!(profile.subgroup.size, 32);
        assert_eq!(profile.subgroup.min_size, 32);
        assert_eq!(profile.subgroup.max_size, 32);
    }

    #[test]
    fn level_zero_profile_rejects_invalid_runtime_limits() {
        let mut properties = ZeDeviceProperties::query();
        properties.device_type = ZE_DEVICE_TYPE_GPU;
        properties.vendor_id = 0x8086;
        properties.physical_eu_simd_width = 32;
        let mut compute = ZeDeviceComputeProperties::query();
        compute.max_total_group_size = 256;
        compute.max_group_size_x = 256;
        compute.max_group_size_y = 256;
        compute.max_group_size_z = 64;
        compute.max_shared_local_memory = 32 * 1024;
        let mut module = ZeDeviceModuleProperties::query();
        module.spirv_version_supported = 0x0001_0300;

        let mut invalid_properties = properties;
        invalid_properties.device_type = 2;
        assert!(matches!(
            level_zero_profile_from_properties(&invalid_properties, &compute, &module),
            Err(VkFftError::NativeUnavailable { .. })
        ));

        let mut invalid_compute = compute;
        invalid_compute.max_total_group_size = 0;
        assert!(
            level_zero_profile_from_properties(&properties, &invalid_compute, &module).is_err()
        );

        invalid_properties = properties;
        invalid_properties.physical_eu_simd_width = 24;
        assert!(
            level_zero_profile_from_properties(&invalid_properties, &compute, &module).is_err()
        );

        invalid_compute = compute;
        invalid_compute.num_sub_group_sizes = (ZE_SUBGROUPSIZE_COUNT + 1) as u32;
        assert!(
            level_zero_profile_from_properties(&properties, &invalid_compute, &module).is_err()
        );

        invalid_compute = compute;
        invalid_compute.num_sub_group_sizes = 1;
        invalid_compute.sub_group_sizes[0] = 3;
        assert!(
            level_zero_profile_from_properties(&properties, &invalid_compute, &module).is_err()
        );

        let unsupported_module = ZeDeviceModuleProperties::query();
        assert!(matches!(
            level_zero_profile_from_properties(&properties, &compute, &unsupported_module),
            Err(VkFftError::NativeUnavailable { .. })
        ));
    }

    #[test]
    fn level_zero_compute_queue_group_selection_uses_reported_ordinal() {
        let mut copy_only = ZeCommandQueueGroupProperties::query();
        copy_only.flags = 1 << 1;
        copy_only.num_queues = 1;
        let mut compute_without_engines = ZeCommandQueueGroupProperties::query();
        compute_without_engines.flags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        compute_without_engines.num_queues = 0;
        let mut compute = ZeCommandQueueGroupProperties::query();
        compute.flags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE | (1 << 1);
        compute.num_queues = 2;

        assert_eq!(
            select_level_zero_compute_queue_group(&[copy_only, compute_without_engines, compute,])
                .unwrap(),
            2
        );
        assert!(select_level_zero_compute_queue_group(&[copy_only]).is_err());
        assert!(select_level_zero_compute_queue_group(&[]).is_err());
    }

    #[test]
    fn level_zero_sparse_bindings_map_to_positional_kernel_arguments() {
        use crate::kernel_ir::{BufferAccess, BufferRole, DispatchGeometry};
        use crate::program_ir::{ProgramPassBinding, ProgramResourceId};

        let pass = ProgramPass {
            name: "sparse_bindings".to_owned(),
            dispatch: DispatchGeometry { x: 1, y: 1, z: 1 },
            bindings: vec![
                ProgramPassBinding {
                    binding: 7,
                    resource: ProgramResourceId(0),
                    role: BufferRole::Input,
                    access: BufferAccess::ReadOnly,
                },
                ProgramPassBinding {
                    binding: 1,
                    resource: ProgramResourceId(2),
                    role: BufferRole::Auxiliary,
                    access: BufferAccess::ReadOnly,
                },
                ProgramPassBinding {
                    binding: 4,
                    resource: ProgramResourceId(1),
                    role: BufferRole::Output,
                    access: BufferAccess::ReadWrite,
                },
            ],
        };
        let memory_plan = ProgramMemoryPlan {
            allocations: Vec::new(),
            resource_allocations: vec![
                ProgramAllocationId(2),
                ProgramAllocationId(0),
                ProgramAllocationId(1),
            ],
        };
        assert_eq!(
            level_zero_ordered_allocation_ids(&pass, &memory_plan).unwrap(),
            vec![
                ProgramAllocationId(1),
                ProgramAllocationId(0),
                ProgramAllocationId(2),
            ]
        );
    }

    #[test]
    fn ocloc_spirv_archive_round_trips_and_rejects_incompatible_headers() {
        let valid_spirv = || {
            let mut bytes = vec![0u8; 20];
            bytes[..4].copy_from_slice(&SPIRV_MAGIC.to_le_bytes());
            bytes
        };
        let archive = OclocSpirvCacheArchive {
            identity: OclocCompilerIdentity {
                executable_fingerprint: [0x5a; OCLOC_SPIRV_ARCHIVE_FINGERPRINT_BYTES],
                executable_len: 12345,
            },
            entries: vec![
                ("source-b".to_owned(), valid_spirv()),
                ("source-a".to_owned(), valid_spirv()),
            ],
        };
        let encoded = archive.encode();
        let decoded = OclocSpirvCacheArchive::decode(&encoded).unwrap();
        assert_eq!(decoded.identity, archive.identity);
        assert_eq!(decoded.entry_count(), 2);
        assert_eq!(decoded.entries[0].0, "source-a");
        assert_eq!(decoded.entries[1].0, "source-b");
        assert_eq!(decoded.encode(), encoded);

        let mut bad_version = encoded.clone();
        bad_version[8..12].copy_from_slice(&(OCLOC_SPIRV_ARCHIVE_VERSION + 1).to_le_bytes());
        assert!(OclocSpirvCacheArchive::decode(&bad_version).is_err());

        let mut bad_commit = encoded.clone();
        bad_commit[12] ^= 0x01;
        assert!(OclocSpirvCacheArchive::decode(&bad_commit).is_err());

        let mut trailing = encoded;
        trailing.push(0);
        assert!(OclocSpirvCacheArchive::decode(&trailing).is_err());
    }

    #[test]
    fn ocloc_cache_enforces_total_byte_budget_and_skips_oversized_entries() {
        let mut cache = OclocSpirvCache::new(4, 12);
        cache.insert("aaaa", &[1, 2, 3, 4]);
        assert_eq!(cache.get("aaaa"), Some(vec![1, 2, 3, 4]));
        assert_eq!(cache.total_bytes, 8);

        // This 8-byte entry would push the cache to 16 bytes, so the oldest entry is evicted.
        cache.insert("bbbb", &[5, 6, 7, 8]);
        assert!(cache.get("aaaa").is_none());
        assert_eq!(cache.get("bbbb"), Some(vec![5, 6, 7, 8]));
        assert_eq!(cache.total_bytes, 8);

        // Source + artifact exceeds the entire budget and must never enter the cache.
        cache.insert("oversized", &[0; 8]);
        assert!(cache.get("oversized").is_none());
        assert_eq!(cache.get("bbbb"), Some(vec![5, 6, 7, 8]));
        assert_eq!(cache.total_bytes, 8);
    }

    #[test]
    fn level_zero_spirv_validation_rejects_bad_header() {
        let workspace = OclocWorkspace::create().expect("test workspace");
        let path = workspace.path.join("bad.spv");
        fs::write(&path, [0u8; 20]).expect("write malformed SPIR-V");
        assert!(matches!(
            read_valid_spirv(&path),
            Err(VkFftError::ShaderCompilation(message)) if message.contains("magic")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_discovery_requires_an_executable_file() {
        use std::os::unix::fs::PermissionsExt;

        let workspace = OclocWorkspace::create().expect("test workspace");
        let path = workspace.path.join("ocloc");
        fs::write(&path, "#!/bin/sh\nexit 0\n").expect("write test executable");
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(&path, permissions.clone()).unwrap();
        assert!(!executable_file(&path));
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).unwrap();
        assert!(executable_file(&path));
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_adapter_compiles_opencl_c_with_modern_cli_contract() {
        let script_workspace = OclocWorkspace::create().expect("script workspace");
        let executable = write_fake_ocloc(&script_workspace.path, false, false);
        let compiler = LevelZeroOclocCompiler::from_executable(executable);
        let spirv = compiler
            .compile_opencl_c_to_spirv(
                "__kernel void vkfft_test(__global float *x) { x[0] = 1.0f; }",
            )
            .expect("fake OCLOC compile");
        assert_eq!(
            u32::from_le_bytes(spirv[0..4].try_into().unwrap()),
            SPIRV_MAGIC
        );
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_adapter_falls_back_to_legacy_cli_contract() {
        let script_workspace = OclocWorkspace::create().expect("script workspace");
        let executable = write_fake_ocloc(&script_workspace.path, true, false);
        let compiler = LevelZeroOclocCompiler::from_executable(executable);
        let spirv = compiler
            .compile_opencl_c_to_spirv(
                "__kernel void vkfft_test(__global float *x) { x[0] = 2.0f; }",
            )
            .expect("legacy fake OCLOC compile");
        assert_eq!(spirv.len(), 20);
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_cache_reuses_successful_sources_and_evicts_by_entry_limit() {
        let script_workspace = OclocWorkspace::create().expect("script workspace");
        let (executable, counter) = write_counting_fake_ocloc(&script_workspace.path, false);
        let compiler = LevelZeroOclocCompiler::from_executable_with_cache_limits(
            executable,
            1,
            OCLOC_CACHE_MAX_BYTES,
        );
        let source_a = "__kernel void vkfft_a(__global float *x) { x[0] = 1.0f; }";
        let source_b = "__kernel void vkfft_b(__global float *x) { x[0] = 2.0f; }";
        compiler.compile_opencl_c_to_spirv(source_a).unwrap();
        compiler.compile_opencl_c_to_spirv(source_a).unwrap();
        assert_eq!(read_fake_ocloc_count(&counter), 1);
        compiler.compile_opencl_c_to_spirv(source_b).unwrap();
        assert_eq!(read_fake_ocloc_count(&counter), 2);
        compiler.compile_opencl_c_to_spirv(source_a).unwrap();
        assert_eq!(read_fake_ocloc_count(&counter), 3);
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_persistent_cache_reuses_same_compiler_and_rejects_changed_binary() {
        let script_workspace = OclocWorkspace::create().expect("script workspace");
        let (executable, counter) = write_counting_fake_ocloc(&script_workspace.path, false);
        let source = "__kernel void vkfft_persist(__global float *x) { x[0] = 7.0f; }";

        let compiler = LevelZeroOclocCompiler::from_executable(executable.clone());
        let first = compiler.compile_opencl_c_to_spirv(source).unwrap();
        assert_eq!(read_fake_ocloc_count(&counter), 1);
        let archive_data = compiler.cache_archive_data().unwrap();
        let archive = OclocSpirvCacheArchive::decode(&archive_data).unwrap();
        assert_eq!(archive.identity, compiler.compiler_identity().unwrap());
        assert_eq!(archive.entry_count(), 1);

        let restored = LevelZeroOclocCompiler::from_executable(executable.clone());
        restored.restore_cache_archive(&archive_data).unwrap();
        let second = restored.compile_opencl_c_to_spirv(source).unwrap();
        assert_eq!(second, first);
        assert_eq!(read_fake_ocloc_count(&counter), 1);

        let mut script = fs::read_to_string(&executable).unwrap();
        script.push_str("\n# compiler identity changed\n");
        write_fake_executable_script(&executable, &script);
        let changed = LevelZeroOclocCompiler::from_executable(executable);
        assert!(changed.restore_cache_archive(&archive_data).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_cache_does_not_store_failed_compilations() {
        let script_workspace = OclocWorkspace::create().expect("script workspace");
        let (executable, counter) = write_counting_fake_ocloc(&script_workspace.path, true);
        let compiler = LevelZeroOclocCompiler::from_executable(executable);
        let source = "__kernel void vkfft_fail() {}";
        assert!(compiler.compile_opencl_c_to_spirv(source).is_err());
        assert!(compiler.compile_opencl_c_to_spirv(source).is_err());
        // Each failed compilation tries modern and legacy CLI forms; neither is cached.
        assert_eq!(read_fake_ocloc_count(&counter), 4);
    }

    #[cfg(unix)]
    #[test]
    fn ocloc_adapter_preserves_bounded_compiler_diagnostics() {
        let script_workspace = OclocWorkspace::create().expect("script workspace");
        let executable = write_fake_ocloc(&script_workspace.path, false, true);
        let compiler = LevelZeroOclocCompiler::from_executable(executable);
        let error = compiler
            .compile_opencl_c_to_spirv("__kernel void vkfft_test() {}")
            .expect_err("fake compiler failure must propagate");
        assert!(matches!(
            error,
            VkFftError::ShaderCompilation(message)
                if message.contains("fake OCLOC failure")
                    && message.contains("modern=")
                    && message.contains("legacy=")
        ));
    }

    #[cfg(unix)]
    fn write_counting_fake_ocloc(directory: &Path, always_fail: bool) -> (PathBuf, PathBuf) {
        let executable = directory.join("ocloc-counting");
        let counter = directory.join("ocloc-count");
        let body = if always_fail {
            format!(
                "count=0\n[ -f '{0}' ] && count=$(cat '{0}')\ncount=$((count + 1))\nprintf '%s' \"$count\" > '{0}'\necho 'fake OCLOC failure' >&2\nexit 7\n",
                counter.display()
            )
        } else {
            format!(
                "count=0\n[ -f '{0}' ] && count=$(cat '{0}')\ncount=$((count + 1))\nprintf '%s' \"$count\" > '{0}'\nif [ \"$1\" = \"compile\" ]; then shift; fi\ninput=''\nout=''\nwhile [ \"$#\" -gt 0 ]; do\n  case \"$1\" in\n    -file) shift; input=\"$1\" ;;\n    -output) shift; out=\"$1\" ;;\n  esac\n  shift\ndone\n[ -n \"$input\" ] && [ -f \"$input\" ] || exit 91\ngrep -q '__kernel' \"$input\" || exit 92\n[ -n \"$out\" ] || exit 93\nprintf '\\003\\002\\043\\007\\000\\000\\001\\000\\000\\000\\000\\000\\001\\000\\000\\000\\000\\000\\000\\000' > \"${{out}}.spv\"\n",
                counter.display()
            )
        };
        write_fake_executable_script(&executable, &format!("#!/bin/sh\n{body}"));
        (executable, counter)
    }

    #[cfg(unix)]
    fn write_fake_executable_script(path: &Path, script: &str) {
        use std::os::unix::fs::PermissionsExt;

        let staged = path.with_extension("staged");
        fs::write(&staged, script).expect("stage fake OCLOC executable");
        let mut permissions = fs::metadata(&staged).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&staged, permissions).unwrap();
        fs::rename(&staged, path).expect("publish fake OCLOC executable atomically");
    }

    #[cfg(unix)]
    fn read_fake_ocloc_count(path: &Path) -> usize {
        fs::read_to_string(path)
            .expect("read fake OCLOC counter")
            .parse()
            .expect("parse fake OCLOC counter")
    }

    #[cfg(unix)]
    fn write_fake_ocloc(directory: &Path, reject_modern: bool, always_fail: bool) -> PathBuf {
        let executable = directory.join("ocloc");
        let modern_guard = if reject_modern {
            "if [ \"$1\" = \"compile\" ]; then echo 'modern unsupported' >&2; exit 2; fi\n"
        } else {
            "if [ \"$1\" = \"compile\" ]; then shift; fi\n"
        };
        let body = if always_fail {
            "echo 'fake OCLOC failure' >&2\nexit 7\n".to_owned()
        } else {
            format!(
                "{modern_guard}input=''\nout=''\nwhile [ \"$#\" -gt 0 ]; do\n  case \"$1\" in\n    -file) shift; input=\"$1\" ;;\n    -output) shift; out=\"$1\" ;;\n  esac\n  shift\ndone\n[ -n \"$input\" ] && [ -f \"$input\" ] || exit 91\ngrep -q '__kernel' \"$input\" || exit 92\n[ -n \"$out\" ] || exit 93\nprintf '\\003\\002\\043\\007\\000\\000\\001\\000\\000\\000\\000\\000\\001\\000\\000\\000\\000\\000\\000\\000' > \"${{out}}.spv\"\n"
            )
        };
        write_fake_executable_script(&executable, &format!("#!/bin/sh\n{body}"));
        executable
    }
}
