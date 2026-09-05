//! Correctness-first Metal runtime.
//!
//! Metal source lowering already shares VkFFT's typed native-program contract with CUDA,
//! HIP, OpenCL and Level Zero. This module owns the remaining macOS execution layer:
//! `MTLDevice` enumeration, device-owned runtime MSL compilation, shared buffers, compute
//! pipelines, command submission and readback. F32 compute (including F16 caller storage)
//! is supported first; Metal F64/DD compute remains deliberately unsupported.

#[cfg(target_os = "macos")]
use core::ffi::{c_char, c_void};
#[cfg(any(test, target_os = "macos"))]
use std::collections::{HashMap, VecDeque};
#[cfg(target_os = "macos")]
use std::ffi::{CStr, CString};
#[cfg(target_os = "macos")]
use std::mem::MaybeUninit;
#[cfg(target_os = "macos")]
use std::ptr;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

use crate::application::TransformIr;
use crate::backend::native::NativeProgramSource;
#[cfg(target_os = "macos")]
use crate::backend::native::NativeShaderSource;
use crate::backend::native_runtime::{
    NativeCompiledPassResourceReport, NativeRuntime, NativeRuntimeAvailability,
    NativeTransformInput32, NativeTransformOutput32, unavailable,
};
#[cfg(target_os = "macos")]
use crate::backend::native_runtime::{
    NativeCompiledResourceMetrics, PreparedProgramStorage, prepare_program_complex32, runtime_error,
};
use crate::complex::{Complex32, Complex64};
#[cfg(any(test, target_os = "macos"))]
use crate::config::GpuVendor;
#[cfg(target_os = "macos")]
use crate::config::SubgroupProfile;
use crate::config::{Backend, DeviceProfile};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::ScalarType;
#[cfg(target_os = "macos")]
use crate::program_ir::ProgramAllocationKind;

#[cfg(target_os = "macos")]
type ObjcId = *mut c_void;
#[cfg(target_os = "macos")]
type ObjcSel = *mut c_void;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct MtlSize {
    width: usize,
    height: usize,
    depth: usize,
}

#[cfg(target_os = "macos")]
const METAL_PIPELINE_CACHE_MAX_ENTRIES: usize = 64;
#[cfg(target_os = "macos")]
const METAL_BUFFER_POOL_MAX_BUFFERS: usize = 64;
#[cfg(target_os = "macos")]
const METAL_LUT_CACHE_MAX_ENTRIES: usize = 64;
#[cfg(target_os = "macos")]
const METAL_LUT_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

#[cfg(any(test, target_os = "macos"))]
struct MetalPipelineCache<V> {
    entries: HashMap<String, V>,
    insertion_order: VecDeque<String>,
    max_entries: usize,
}

#[cfg(any(test, target_os = "macos"))]
impl<V> MetalPipelineCache<V> {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            max_entries,
        }
    }

    fn get(&self, source: &str) -> Option<&V> {
        self.entries.get(source)
    }

    fn insert(&mut self, source: String, value: V) {
        if self.max_entries == 0 {
            return;
        }
        if self.entries.remove(&source).is_some() {
            self.insertion_order.retain(|key| key != &source);
        }
        while self.entries.len() >= self.max_entries {
            let oldest = self
                .insertion_order
                .pop_front()
                .expect("non-empty Metal pipeline cache order");
            self.entries.remove(&oldest);
        }
        self.insertion_order.push_back(source.clone());
        self.entries.insert(source, value);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(any(test, target_os = "macos"))]
struct MetalBufferPool<V> {
    buffers: VecDeque<(usize, V)>,
    max_buffers: usize,
}

#[cfg(any(test, target_os = "macos"))]
impl<V> MetalBufferPool<V> {
    fn new(max_buffers: usize) -> Self {
        Self {
            buffers: VecDeque::new(),
            max_buffers,
        }
    }

    fn take(&mut self, allocation_bytes: usize) -> Option<V> {
        let position = self
            .buffers
            .iter()
            .position(|(bytes, _)| *bytes == allocation_bytes)?;
        self.buffers.remove(position).map(|(_, value)| value)
    }

    fn insert(&mut self, allocation_bytes: usize, value: V) {
        if self.max_buffers == 0 {
            return;
        }
        while self.buffers.len() >= self.max_buffers {
            self.buffers.pop_front();
        }
        self.buffers.push_back((allocation_bytes, value));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.buffers.len()
    }

    #[cfg(test)]
    fn contains_size(&self, allocation_bytes: usize) -> bool {
        self.buffers
            .iter()
            .any(|(bytes, _)| *bytes == allocation_bytes)
    }
}

#[cfg(any(test, target_os = "macos"))]
struct MetalLookupBufferCache<V> {
    entries: HashMap<Vec<u8>, V>,
    insertion_order: VecDeque<Vec<u8>>,
    retained_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

#[cfg(any(test, target_os = "macos"))]
impl<V> MetalLookupBufferCache<V> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            retained_bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    fn entry_retained_bytes(key_len: usize) -> Option<usize> {
        // Count the GPU buffer plus the HashMap key and insertion-order key copies.
        key_len.checked_mul(3)
    }

    fn get(&self, key: &[u8]) -> Option<&V> {
        self.entries.get(key)
    }

    fn insert(&mut self, key: Vec<u8>, value: V) {
        let Some(entry_bytes) = Self::entry_retained_bytes(key.len()) else {
            return;
        };
        if self.max_entries == 0 || self.max_bytes == 0 || entry_bytes > self.max_bytes {
            return;
        }
        if self.entries.remove(key.as_slice()).is_some() {
            self.insertion_order.retain(|existing| existing != &key);
            self.retained_bytes = self.retained_bytes.saturating_sub(entry_bytes);
        }
        while self.entries.len() >= self.max_entries
            || self.retained_bytes.saturating_add(entry_bytes) > self.max_bytes
        {
            let Some(oldest) = self.insertion_order.pop_front() else {
                break;
            };
            if self.entries.remove(oldest.as_slice()).is_some()
                && let Some(bytes) = Self::entry_retained_bytes(oldest.len())
            {
                self.retained_bytes = self.retained_bytes.saturating_sub(bytes);
            }
        }
        if self.entries.len() >= self.max_entries
            || self.retained_bytes.saturating_add(entry_bytes) > self.max_bytes
        {
            return;
        }
        self.retained_bytes = self.retained_bytes.saturating_add(entry_bytes);
        self.insertion_order.push_back(key.clone());
        self.entries.insert(key, value);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

#[cfg(target_os = "macos")]
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
#[cfg(target_os = "macos")]
const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

#[cfg(target_os = "macos")]
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCopyAllDevices() -> ObjcId;
}

#[cfg(target_os = "macos")]
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        allocator: *const c_void,
        c_string: *const c_char,
        encoding: u32,
    ) -> *mut c_void;
    fn CFRelease(value: *const c_void);
}

// Objective-C message dispatch is intentionally redeclared with the exact ABI of each
// selector family used below. The runtime symbol is variadic at the language level, while
// Rust requires a concrete foreign-function type at every call site.
#[cfg_attr(target_os = "macos", allow(clashing_extern_declarations))]
#[cfg(target_os = "macos")]
#[link(name = "objc")]
unsafe extern "C" {
    fn sel_registerName(name: *const c_char) -> ObjcSel;
    fn objc_retain(value: ObjcId) -> ObjcId;
    fn objc_release(value: ObjcId);
    fn objc_autoreleasePoolPush() -> ObjcId;
    fn objc_autoreleasePoolPop(pool: ObjcId);

    #[link_name = "objc_msgSend"]
    fn objc_msg_send_id(receiver: ObjcId, selector: ObjcSel) -> ObjcId;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_id_usize(receiver: ObjcId, selector: ObjcSel, value: usize) -> ObjcId;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_id_usize_usize(
        receiver: ObjcId,
        selector: ObjcSel,
        first: usize,
        second: usize,
    ) -> ObjcId;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_id_id(receiver: ObjcId, selector: ObjcSel, value: ObjcId) -> ObjcId;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_id_id_error(
        receiver: ObjcId,
        selector: ObjcSel,
        value: ObjcId,
        error: *mut ObjcId,
    ) -> ObjcId;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_id_id_id_error(
        receiver: ObjcId,
        selector: ObjcSel,
        first: ObjcId,
        second: ObjcId,
        error: *mut ObjcId,
    ) -> ObjcId;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_usize(receiver: ObjcId, selector: ObjcSel) -> usize;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_ptr(receiver: ObjcId, selector: ObjcSel) -> *mut c_void;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_c_char_ptr(receiver: ObjcId, selector: ObjcSel) -> *const c_char;
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_void(receiver: ObjcId, selector: ObjcSel);
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_void_id(receiver: ObjcId, selector: ObjcSel, value: ObjcId);
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_void_id_usize_usize(
        receiver: ObjcId,
        selector: ObjcSel,
        value: ObjcId,
        offset: usize,
        index: usize,
    );
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_void_mtl_size_mtl_size(
        receiver: ObjcId,
        selector: ObjcSel,
        first: MtlSize,
        second: MtlSize,
    );

    #[cfg(target_arch = "aarch64")]
    #[link_name = "objc_msgSend"]
    fn objc_msg_send_mtl_size(receiver: ObjcId, selector: ObjcSel) -> MtlSize;
    #[cfg(target_arch = "x86_64")]
    #[link_name = "objc_msgSend_stret"]
    fn objc_msg_send_mtl_size_stret(result: *mut MtlSize, receiver: ObjcId, selector: ObjcSel);
}

#[cfg(target_os = "macos")]
struct AutoreleasePool(ObjcId);

#[cfg(target_os = "macos")]
impl AutoreleasePool {
    fn new() -> Self {
        // SAFETY: the Objective-C runtime accepts one balanced push/pop pair on this thread.
        Self(unsafe { objc_autoreleasePoolPush() })
    }
}

#[cfg(target_os = "macos")]
impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        // SAFETY: this token came from the matching push in `new`.
        unsafe { objc_autoreleasePoolPop(self.0) };
    }
}

#[cfg(target_os = "macos")]
struct ObjcOwned(ObjcId);

#[cfg(target_os = "macos")]
impl ObjcOwned {
    unsafe fn from_owned(value: ObjcId, operation: &'static str) -> Result<Self> {
        if value.is_null() {
            Err(runtime_error(
                Backend::Metal,
                format!("{operation} returned nil"),
            ))
        } else {
            Ok(Self(value))
        }
    }

    unsafe fn retain_borrowed(value: ObjcId, operation: &'static str) -> Result<Self> {
        if value.is_null() {
            return Err(runtime_error(
                Backend::Metal,
                format!("{operation} returned nil"),
            ));
        }
        // SAFETY: `value` is a live Objective-C object borrowed from a retained container.
        let retained = unsafe { objc_retain(value) };
        if retained.is_null() {
            Err(runtime_error(
                Backend::Metal,
                format!("{operation} could not retain the returned object"),
            ))
        } else {
            Ok(Self(retained))
        }
    }

    const fn as_ptr(&self) -> ObjcId {
        self.0
    }
}

#[cfg(target_os = "macos")]
impl Drop for ObjcOwned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `ObjcOwned` contains one +1 Objective-C reference.
            unsafe { objc_release(self.0) };
        }
    }
}

#[cfg(target_os = "macos")]
struct CfString(*mut c_void);

#[cfg(target_os = "macos")]
impl CfString {
    fn new(value: &str, operation: &'static str) -> Result<Self> {
        let encoded = CString::new(value.as_bytes()).map_err(|_| {
            VkFftError::ShaderCompilation(format!("{operation} contains an interior NUL byte"))
        })?;
        // SAFETY: `encoded` is a valid NUL-terminated UTF-8 string for the duration of the call.
        let string = unsafe {
            CFStringCreateWithCString(ptr::null(), encoded.as_ptr(), CF_STRING_ENCODING_UTF8)
        };
        if string.is_null() {
            Err(runtime_error(
                Backend::Metal,
                format!("{operation} could not allocate a CFString"),
            ))
        } else {
            Ok(Self(string))
        }
    }

    const fn as_objc(&self) -> ObjcId {
        self.0.cast()
    }
}

#[cfg(target_os = "macos")]
impl Drop for CfString {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the string was returned from a CoreFoundation Create function.
            unsafe { CFRelease(self.0.cast_const()) };
        }
    }
}

#[cfg(target_os = "macos")]
unsafe fn selector(name: &'static [u8]) -> ObjcSel {
    debug_assert_eq!(name.last(), Some(&0));
    // SAFETY: every caller supplies a static NUL-terminated selector spelling.
    unsafe { sel_registerName(name.as_ptr().cast()) }
}

#[cfg(target_os = "macos")]
unsafe fn message_mtl_size(receiver: ObjcId, selector_value: ObjcSel) -> MtlSize {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: this selector returns `MTLSize` by value on arm64.
        unsafe { objc_msg_send_mtl_size(receiver, selector_value) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mut result = MaybeUninit::<MtlSize>::uninit();
        // SAFETY: x86_64 Objective-C uses objc_msgSend_stret for this `MTLSize` return.
        unsafe { objc_msg_send_mtl_size_stret(result.as_mut_ptr(), receiver, selector_value) };
        // SAFETY: objc_msgSend_stret initializes the complete result on return.
        unsafe { result.assume_init() }
    }
}

#[cfg(target_os = "macos")]
unsafe fn objc_string(value: ObjcId) -> String {
    if value.is_null() {
        return "<nil>".to_owned();
    }
    // SAFETY: NSString responds to UTF8String and returns a borrowed C string.
    let bytes = unsafe { objc_msg_send_c_char_ptr(value, selector(b"UTF8String\0")) };
    if bytes.is_null() {
        "<unavailable string>".to_owned()
    } else {
        // SAFETY: UTF8String is NUL-terminated while the NSString remains alive.
        unsafe { CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(target_os = "macos")]
unsafe fn objc_error_string(error: ObjcId) -> String {
    if error.is_null() {
        return "no NSError detail".to_owned();
    }
    // SAFETY: NSError responds to localizedDescription.
    let description = unsafe { objc_msg_send_id(error, selector(b"localizedDescription\0")) };
    // SAFETY: the returned object is an NSString valid inside the active autorelease pool.
    unsafe { objc_string(description) }
}

#[cfg(target_os = "macos")]
fn enumerate_metal_devices() -> Result<Vec<ObjcOwned>> {
    let _pool = AutoreleasePool::new();
    // SAFETY: MTLCopyAllDevices is a macOS Metal framework function returning a retained NSArray.
    let array = unsafe { ObjcOwned::from_owned(MTLCopyAllDevices(), "MTLCopyAllDevices")? };
    // SAFETY: NSArray responds to count.
    let count = unsafe { objc_msg_send_usize(array.as_ptr(), selector(b"count\0")) };
    let mut devices = Vec::with_capacity(count);
    for index in 0..count {
        // SAFETY: index is bounded by NSArray::count.
        let device =
            unsafe { objc_msg_send_id_usize(array.as_ptr(), selector(b"objectAtIndex:\0"), index) };
        // SAFETY: retain the borrowed array element before the array is released.
        devices.push(unsafe {
            ObjcOwned::retain_borrowed(device, "MTLCopyAllDevices objectAtIndex")?
        });
    }
    Ok(devices)
}

#[cfg(target_os = "macos")]
fn metal_device_name(device: ObjcId) -> String {
    let _pool = AutoreleasePool::new();
    // SAFETY: every MTLDevice responds to `name` and returns NSString.
    let name = unsafe { objc_msg_send_id(device, selector(b"name\0")) };
    // SAFETY: the NSString remains valid for the active autorelease pool.
    unsafe { objc_string(name) }
}

#[cfg(target_os = "macos")]
fn floor_power_of_two(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

#[cfg(target_os = "macos")]
fn metal_device_profile(device: ObjcId) -> Result<DeviceProfile> {
    // SAFETY: these selectors are part of the MTLDevice protocol on supported macOS releases.
    let shared_memory_bytes =
        unsafe { objc_msg_send_usize(device, selector(b"maxThreadgroupMemoryLength\0")) };
    // SAFETY: `maxThreadsPerThreadgroup` returns MTLSize.
    let max_size = unsafe { message_mtl_size(device, selector(b"maxThreadsPerThreadgroup\0")) };
    if shared_memory_bytes == 0
        || max_size.width == 0
        || max_size.height == 0
        || max_size.depth == 0
    {
        return Err(runtime_error(
            Backend::Metal,
            format!(
                "MTLDevice reported invalid limits: threadgroup_memory={shared_memory_bytes}, max_threads={max_size:?}"
            ),
        ));
    }
    Ok(DeviceProfile {
        backend: Backend::Metal,
        vendor: GpuVendor::Apple,
        shared_memory_bytes,
        shared_memory_pow2_bytes: floor_power_of_two(shared_memory_bytes),
        // Metal's x dimension is the useful total-thread ceiling for VkFFT's 1D/2D
        // workgroups; retain the complete per-dimension MTLSize separately below.
        max_threads_per_block: max_size.width,
        max_workgroup_size: [max_size.width, max_size.height, max_size.depth],
        coalesced_memory_bytes: 64,
        shared_banks: 32,
        supports_f64: false,
        // The source backend deliberately uses the shared/threadgroup fallback until
        // SIMD-group shuffle semantics are proven on real Apple hardware.
        subgroup: SubgroupProfile::unavailable(),
    })
}

#[derive(Debug, Clone)]
pub struct MetalRuntimeAdapter {
    availability: NativeRuntimeAvailability,
}

impl MetalRuntimeAdapter {
    pub fn probe() -> NativeRuntimeAvailability {
        MetalExecutionContext::probe()
    }

    pub fn new() -> Result<Self> {
        let availability = Self::probe();
        if !availability.available() {
            return Err(unavailable(Backend::Metal, availability.detail.clone()));
        }
        Ok(Self { availability })
    }

    pub const fn availability(&self) -> &NativeRuntimeAvailability {
        &self.availability
    }

    /// Legacy capability hook retained for API compatibility. Execution is considered
    /// validated only when the runtime has a real MTLDevice and device-owned MSL compiler.
    pub fn execution_not_yet_validated(&self) -> Result<()> {
        if self.availability.available() {
            Ok(())
        } else {
            Err(unavailable(
                Backend::Metal,
                self.availability.detail.clone(),
            ))
        }
    }
}

pub struct MetalExecutionContext {
    #[cfg(target_os = "macos")]
    device: ObjcOwned,
    #[cfg(target_os = "macos")]
    queue: ObjcOwned,
    #[cfg(target_os = "macos")]
    pipeline_cache: Mutex<MetalPipelineCache<ObjcOwned>>,
    #[cfg(target_os = "macos")]
    buffer_pool: Mutex<MetalBufferPool<ObjcOwned>>,
    #[cfg(target_os = "macos")]
    lut_cache: Mutex<MetalLookupBufferCache<ObjcOwned>>,
    profile: DeviceProfile,
    device_name: String,
}

#[cfg(target_os = "macos")]
struct MetalPendingProgram<'a> {
    context: &'a MetalExecutionContext,
    command_buffer: ObjcOwned,
    prepared: PreparedProgramStorage,
    buffers: Vec<ObjcOwned>,
    _pipelines: Vec<ObjcOwned>,
    completed: bool,
}

pub struct MetalProgramTicket32<'a> {
    #[cfg(target_os = "macos")]
    pending: MetalPendingProgram<'a>,
    #[cfg(not(target_os = "macos"))]
    _context: core::marker::PhantomData<&'a MetalExecutionContext>,
}

pub struct MetalProgramTicket64<'a> {
    _context: core::marker::PhantomData<&'a MetalExecutionContext>,
}

impl MetalExecutionContext {
    pub fn probe() -> NativeRuntimeAvailability {
        #[cfg(target_os = "macos")]
        {
            // Do not probe the framework by pathname. Modern macOS may satisfy Metal from
            // the dyld shared cache while `/System/Library/Frameworks/Metal.framework/Metal`
            // is a dangling-on-disk symlink. Reaching this linked macOS module already proves
            // that the framework loader resolved; device enumeration is the authoritative
            // runtime capability boundary.
            match enumerate_metal_devices() {
                Ok(devices) => {
                    let names = devices
                        .iter()
                        .map(|device| metal_device_name(device.as_ptr()))
                        .collect::<Vec<_>>();
                    let device_count = devices.len();
                    NativeRuntimeAvailability {
                        backend: Backend::Metal,
                        loader_available: true,
                        // MSL compilation is device-owned through newLibraryWithSource; no
                        // external xcrun/metal compiler is required for runtime execution.
                        compiler_available: device_count > 0,
                        device_count,
                        detail: if device_count == 0 {
                            "Metal.framework loaded, but MTLCopyAllDevices returned no MTLDevice"
                                .to_owned()
                        } else {
                            format!(
                                "Metal.framework reports {device_count} MTLDevice(s): {}; runtime MSL compiler=newLibraryWithSource",
                                names.join(", ")
                            )
                        },
                    }
                }
                Err(error) => NativeRuntimeAvailability {
                    backend: Backend::Metal,
                    loader_available: true,
                    compiler_available: false,
                    device_count: 0,
                    detail: error.to_string(),
                },
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            NativeRuntimeAvailability {
                backend: Backend::Metal,
                loader_available: false,
                compiler_available: false,
                device_count: 0,
                detail: "Metal is available only on macOS".to_owned(),
            }
        }
    }

    pub fn new(device_index: usize) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let mut devices = enumerate_metal_devices()?;
            let device_count = devices.len();
            if device_index >= device_count {
                return Err(unavailable(
                    Backend::Metal,
                    format!(
                        "requested device {device_index}, but only {device_count} Metal device(s) are available"
                    ),
                ));
            }
            let device = devices.swap_remove(device_index);
            let device_name = metal_device_name(device.as_ptr());
            let profile = metal_device_profile(device.as_ptr())?;
            // SAFETY: MTLDevice::newCommandQueue returns a retained command queue.
            let queue = unsafe {
                ObjcOwned::from_owned(
                    objc_msg_send_id(device.as_ptr(), selector(b"newCommandQueue\0")),
                    "MTLDevice newCommandQueue",
                )?
            };
            Ok(Self {
                device,
                queue,
                pipeline_cache: Mutex::new(MetalPipelineCache::new(
                    METAL_PIPELINE_CACHE_MAX_ENTRIES,
                )),
                buffer_pool: Mutex::new(MetalBufferPool::new(METAL_BUFFER_POOL_MAX_BUFFERS)),
                lut_cache: Mutex::new(MetalLookupBufferCache::new(
                    METAL_LUT_CACHE_MAX_ENTRIES,
                    METAL_LUT_CACHE_MAX_BYTES,
                )),
                profile,
                device_name,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = device_index;
            Err(unavailable(
                Backend::Metal,
                "Metal is available only on macOS",
            ))
        }
    }

    pub const fn device_profile(&self) -> DeviceProfile {
        self.profile
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn submit_program_complex32<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<MetalProgramTicket32<'a>> {
        source.validate()?;
        if source.backend != Backend::Metal || source.program.scalar != ScalarType::F32 {
            return Err(VkFftError::InvalidKernelIr(
                "Metal F32 execution requires a Metal/F32 native program",
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let prepared = prepare_program_complex32(Backend::Metal, &source.program, input)?;
            Ok(MetalProgramTicket32 {
                pending: self.submit_prepared(source, prepared)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = input;
            Err(unavailable(
                Backend::Metal,
                "Metal is available only on macOS",
            ))
        }
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
        _input: &[Complex64],
    ) -> Result<MetalProgramTicket64<'a>> {
        source.validate()?;
        Err(VkFftError::UnsupportedPrecision {
            backend: "Metal runtime",
            precision: "f64 compute",
        })
    }

    pub fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        self.submit_program_complex64(source, input)?.wait()
    }

    crate::backend::native_runtime::impl_native_transform_convenience_facade!(
        Backend::Metal,
        MetalProgramTicket32,
        MetalProgramTicket64
    );

    pub fn execute_transform_f32(
        &self,
        ir: &TransformIr,
        input: NativeTransformInput32<'_>,
    ) -> Result<NativeTransformOutput32> {
        self.submit_transform_f32(ir, input)?.wait()
    }

    #[cfg(target_os = "macos")]
    fn compile_pipeline(&self, shader: &NativeShaderSource) -> Result<ObjcOwned> {
        shader.validate()?;
        if shader.backend != Backend::Metal || shader.scalar != ScalarType::F32 {
            return Err(VkFftError::InvalidKernelIr(
                "Metal pipeline compilation requires a Metal/F32 shader",
            ));
        }
        let cached = {
            let cache = self.pipeline_cache.lock().map_err(|_| {
                runtime_error(Backend::Metal, "Metal pipeline cache lock is poisoned")
            })?;
            match cache.get(&shader.source) {
                Some(pipeline) => Some(unsafe {
                    ObjcOwned::retain_borrowed(pipeline.as_ptr(), "cached Metal compute pipeline")?
                }),
                None => None,
            }
        };
        if let Some(pipeline) = cached {
            self.validate_pipeline_workgroup(shader, &pipeline)?;
            return Ok(pipeline);
        }

        let _pool = AutoreleasePool::new();
        let source = CfString::new(&shader.source, "Metal shader source")?;
        let mut error = ptr::null_mut();
        // SAFETY: selector and argument ABI follow MTLDevice::newLibraryWithSource:options:error:.
        let library_ptr = unsafe {
            objc_msg_send_id_id_id_error(
                self.device.as_ptr(),
                selector(b"newLibraryWithSource:options:error:\0"),
                source.as_objc(),
                ptr::null_mut(),
                &mut error,
            )
        };
        if library_ptr.is_null() {
            // SAFETY: NSError is valid inside the active autorelease pool.
            let detail = unsafe { objc_error_string(error) };
            return Err(VkFftError::ShaderCompilation(format!(
                "Metal newLibraryWithSource failed: {detail}"
            )));
        }
        // SAFETY: `newLibraryWithSource` returns a +1 object.
        let library = unsafe { ObjcOwned::from_owned(library_ptr, "Metal library")? };
        let entry = CfString::new(shader.entry_point, "Metal entry point")?;
        // SAFETY: MTLLibrary::newFunctionWithName returns a retained function object.
        let function_ptr = unsafe {
            objc_msg_send_id_id(
                library.as_ptr(),
                selector(b"newFunctionWithName:\0"),
                entry.as_objc(),
            )
        };
        // SAFETY: `newFunctionWithName` follows the Cocoa create rule.
        let function = unsafe { ObjcOwned::from_owned(function_ptr, "Metal function")? };
        error = ptr::null_mut();
        // SAFETY: selector ABI follows MTLDevice::newComputePipelineStateWithFunction:error:.
        let pipeline_ptr = unsafe {
            objc_msg_send_id_id_error(
                self.device.as_ptr(),
                selector(b"newComputePipelineStateWithFunction:error:\0"),
                function.as_ptr(),
                &mut error,
            )
        };
        if pipeline_ptr.is_null() {
            // SAFETY: NSError is valid inside the active autorelease pool.
            let detail = unsafe { objc_error_string(error) };
            return Err(VkFftError::ShaderCompilation(format!(
                "Metal compute-pipeline compilation failed: {detail}"
            )));
        }
        // SAFETY: `newComputePipelineStateWithFunction` returns a +1 object.
        let pipeline = unsafe { ObjcOwned::from_owned(pipeline_ptr, "Metal compute pipeline")? };
        self.validate_pipeline_workgroup(shader, &pipeline)?;
        // Cache owns one retained reference; the returned pipeline keeps the original +1
        // reference so in-flight tickets remain independent of later cache eviction.
        let cached_pipeline = unsafe {
            ObjcOwned::retain_borrowed(pipeline.as_ptr(), "Metal pipeline cache insertion")?
        };
        self.pipeline_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Metal, "Metal pipeline cache lock is poisoned"))?
            .insert(shader.source.clone(), cached_pipeline);
        Ok(pipeline)
    }

    #[cfg(target_os = "macos")]
    fn validate_pipeline_workgroup(
        &self,
        shader: &NativeShaderSource,
        pipeline: &ObjcOwned,
    ) -> Result<()> {
        let requested_threads = (shader.workgroup_size.x as usize)
            .checked_mul(shader.workgroup_size.y as usize)
            .and_then(|value| value.checked_mul(shader.workgroup_size.z as usize))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Metal workgroup thread product",
            })?;
        // SAFETY: MTLComputePipelineState exposes maxTotalThreadsPerThreadgroup as NSUInteger.
        let max_threads = unsafe {
            objc_msg_send_usize(
                pipeline.as_ptr(),
                selector(b"maxTotalThreadsPerThreadgroup\0"),
            )
        };
        if requested_threads == 0 || requested_threads > max_threads {
            return Err(runtime_error(
                Backend::Metal,
                format!(
                    "Metal shader requests {requested_threads} threads/workgroup, pipeline allows {max_threads}"
                ),
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn compiled_pipeline_resource_metrics(
        &self,
        pipeline: &ObjcOwned,
    ) -> Result<NativeCompiledResourceMetrics> {
        // SAFETY: these are read-only MTLComputePipelineState NSUInteger properties.
        let static_threadgroup_memory_bytes = unsafe {
            objc_msg_send_usize(
                pipeline.as_ptr(),
                selector(b"staticThreadgroupMemoryLength\0"),
            )
        };
        let max_threads_per_threadgroup = unsafe {
            objc_msg_send_usize(
                pipeline.as_ptr(),
                selector(b"maxTotalThreadsPerThreadgroup\0"),
            )
        };
        let thread_execution_width =
            unsafe { objc_msg_send_usize(pipeline.as_ptr(), selector(b"threadExecutionWidth\0")) };
        if max_threads_per_threadgroup == 0 || thread_execution_width == 0 {
            return Err(runtime_error(
                Backend::Metal,
                format!(
                    "Metal pipeline reported invalid execution limits: max_threads={max_threads_per_threadgroup}, thread_execution_width={thread_execution_width}"
                ),
            ));
        }
        Ok(NativeCompiledResourceMetrics::Metal {
            static_threadgroup_memory_bytes,
            max_threads_per_threadgroup,
            thread_execution_width,
        })
    }

    #[cfg(target_os = "macos")]
    fn allocate_shared_buffer(&self, allocation_bytes: usize) -> Result<ObjcOwned> {
        // MTLResourceStorageModeShared is zero in MTLResourceOptions.
        // SAFETY: selector ABI follows MTLDevice::newBufferWithLength:options:.
        let buffer_ptr = unsafe {
            objc_msg_send_id_usize_usize(
                self.device.as_ptr(),
                selector(b"newBufferWithLength:options:\0"),
                allocation_bytes,
                0,
            )
        };
        // SAFETY: `newBufferWithLength` returns a +1 object.
        unsafe { ObjcOwned::from_owned(buffer_ptr, "Metal buffer") }
    }

    #[cfg(target_os = "macos")]
    fn upload_shared_buffer(&self, buffer: &ObjcOwned, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: shared-storage MTLBuffer::contents returns CPU-visible memory.
        let contents = unsafe { objc_msg_send_ptr(buffer.as_ptr(), selector(b"contents\0")) };
        if contents.is_null() {
            return Err(runtime_error(
                Backend::Metal,
                "shared Metal buffer returned null contents",
            ));
        }
        // SAFETY: the pool matches exact allocation size, or a new allocation is at least
        // `bytes.len()` bytes; source and destination do not overlap.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast::<u8>(), bytes.len()) };
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn take_shared_buffer(&self, bytes: usize) -> Result<ObjcOwned> {
        let allocation_bytes = bytes.max(1);
        let pooled = self
            .buffer_pool
            .lock()
            .map_err(|_| runtime_error(Backend::Metal, "Metal buffer pool lock is poisoned"))?
            .take(allocation_bytes);
        match pooled {
            Some(buffer) => Ok(buffer),
            None => self.allocate_shared_buffer(allocation_bytes),
        }
    }

    #[cfg(target_os = "macos")]
    fn take_shared_buffer_and_upload(&self, bytes: &[u8]) -> Result<ObjcOwned> {
        let buffer = self.take_shared_buffer(bytes.len())?;
        self.upload_shared_buffer(&buffer, bytes)?;
        Ok(buffer)
    }

    #[cfg(target_os = "macos")]
    fn take_lookup_buffer(&self, bytes: &[u8]) -> Result<ObjcOwned> {
        let cached = {
            let cache = self
                .lut_cache
                .lock()
                .map_err(|_| runtime_error(Backend::Metal, "Metal LUT cache lock is poisoned"))?;
            match cache.get(bytes) {
                Some(buffer) => Some(unsafe {
                    ObjcOwned::retain_borrowed(buffer.as_ptr(), "cached Metal lookup buffer")?
                }),
                None => None,
            }
        };
        if let Some(buffer) = cached {
            return Ok(buffer);
        }

        let buffer = self.take_shared_buffer_and_upload(bytes)?;
        let cached_buffer =
            unsafe { ObjcOwned::retain_borrowed(buffer.as_ptr(), "Metal LUT cache insertion")? };
        self.lut_cache
            .lock()
            .map_err(|_| runtime_error(Backend::Metal, "Metal LUT cache lock is poisoned"))?
            .insert(bytes.to_vec(), cached_buffer);
        Ok(buffer)
    }

    #[cfg(target_os = "macos")]
    fn submit_prepared<'a>(
        &'a self,
        source: &NativeProgramSource,
        prepared: PreparedProgramStorage,
    ) -> Result<MetalPendingProgram<'a>> {
        source.validate()?;
        let _pool = AutoreleasePool::new();
        let buffers = prepared
            .memory_plan
            .allocations
            .iter()
            .zip(&prepared.allocations)
            .map(|(allocation, prepared_allocation)| {
                let host_bytes = prepared_allocation.host_bytes.as_deref();
                if allocation.kind == ProgramAllocationKind::LookupTable {
                    let bytes = host_bytes.ok_or(VkFftError::InvalidKernelIr(
                        "Metal lookup-table allocation is missing initialization bytes",
                    ))?;
                    self.take_lookup_buffer(bytes)
                } else if let Some(bytes) = host_bytes {
                    self.take_shared_buffer_and_upload(bytes)
                } else {
                    self.take_shared_buffer(prepared_allocation.byte_len)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        // commandBuffer/encoders are autoreleased. Retain the command buffer explicitly so
        // the returned ticket remains valid after this submission-local pool drains.
        // SAFETY: MTLCommandQueue responds to commandBuffer and returns a borrowed object.
        let command_buffer = unsafe {
            ObjcOwned::retain_borrowed(
                objc_msg_send_id(self.queue.as_ptr(), selector(b"commandBuffer\0")),
                "MTLCommandQueue commandBuffer",
            )?
        };
        let mut pipelines = Vec::with_capacity(source.shaders.len());
        for (pass, shader) in source.program.passes.iter().zip(&source.shaders) {
            let pipeline = self.compile_pipeline(shader)?;
            // SAFETY: MTLCommandBuffer responds to computeCommandEncoder.
            let encoder = unsafe {
                objc_msg_send_id(
                    command_buffer.as_ptr(),
                    selector(b"computeCommandEncoder\0"),
                )
            };
            if encoder.is_null() {
                return Err(runtime_error(
                    Backend::Metal,
                    format!(
                        "Metal pass `{}` could not create a compute encoder",
                        pass.name
                    ),
                ));
            }
            // SAFETY: selector ABI follows setComputePipelineState:.
            unsafe {
                objc_msg_send_void_id(
                    encoder,
                    selector(b"setComputePipelineState:\0"),
                    pipeline.as_ptr(),
                )
            };
            for binding in &pass.bindings {
                let allocation = prepared.memory_plan.allocation_for(binding.resource)?;
                let buffer = buffers
                    .get(allocation.0)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "Metal pass references a missing program buffer",
                    ))?;
                // Native MSL uses the typed binding number directly in [[buffer(n)]].
                // SAFETY: selector ABI follows setBuffer:offset:atIndex:.
                unsafe {
                    objc_msg_send_void_id_usize_usize(
                        encoder,
                        selector(b"setBuffer:offset:atIndex:\0"),
                        buffer.as_ptr(),
                        0,
                        binding.binding as usize,
                    )
                };
            }
            let groups = MtlSize {
                width: shader.dispatch.x as usize,
                height: shader.dispatch.y as usize,
                depth: shader.dispatch.z as usize,
            };
            let threads = MtlSize {
                width: shader.workgroup_size.x as usize,
                height: shader.workgroup_size.y as usize,
                depth: shader.workgroup_size.z as usize,
            };
            // SAFETY: selector ABI follows dispatchThreadgroups:threadsPerThreadgroup:.
            unsafe {
                objc_msg_send_void_mtl_size_mtl_size(
                    encoder,
                    selector(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                    groups,
                    threads,
                );
                objc_msg_send_void(encoder, selector(b"endEncoding\0"));
            }
            pipelines.push(pipeline);
        }
        // Commit without waiting. The ticket retains the command buffer and every explicitly
        // referenced resource until `wait` or drop.
        // SAFETY: MTLCommandBuffer::commit begins execution of the fully encoded buffer.
        unsafe { objc_msg_send_void(command_buffer.as_ptr(), selector(b"commit\0")) };
        Ok(MetalPendingProgram {
            context: self,
            command_buffer,
            prepared,
            buffers,
            _pipelines: pipelines,
            completed: false,
        })
    }
}

#[cfg(target_os = "macos")]
impl MetalPendingProgram<'_> {
    fn recycle_buffers(&mut self) {
        let allocation_metadata = self
            .prepared
            .memory_plan
            .allocations
            .iter()
            .zip(&self.prepared.allocations)
            .map(|(allocation, prepared_allocation)| {
                (allocation.kind, prepared_allocation.byte_len.max(1))
            })
            .collect::<Vec<_>>();
        let Ok(mut pool) = self.context.buffer_pool.lock() else {
            return;
        };
        for ((kind, allocation_bytes), buffer) in
            allocation_metadata.into_iter().zip(self.buffers.drain(..))
        {
            if kind != ProgramAllocationKind::LookupTable {
                pool.insert(allocation_bytes, buffer);
            }
        }
    }

    fn finish(&mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        let _pool = AutoreleasePool::new();
        // SAFETY: the retained command buffer remains valid until this pending program drops.
        unsafe {
            objc_msg_send_void(
                self.command_buffer.as_ptr(),
                selector(b"waitUntilCompleted\0"),
            )
        };
        // SAFETY: MTLCommandBuffer::status returns MTLCommandBufferStatus as NSUInteger.
        let status =
            unsafe { objc_msg_send_usize(self.command_buffer.as_ptr(), selector(b"status\0")) };
        if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
            // SAFETY: MTLCommandBuffer::error returns NSError or nil.
            let error =
                unsafe { objc_msg_send_id(self.command_buffer.as_ptr(), selector(b"error\0")) };
            // SAFETY: error object is valid inside the active autorelease pool.
            let detail = unsafe { objc_error_string(error) };
            return Err(runtime_error(
                Backend::Metal,
                format!("Metal command buffer finished with status {status}: {detail}"),
            ));
        }

        let output_index = self.prepared.output_allocation.0;
        let output_buffer = self
            .buffers
            .get(output_index)
            .ok_or(VkFftError::InvalidKernelIr(
                "Metal program is missing its output buffer",
            ))?;
        let output_bytes = self.prepared.output_bytes_mut()?;
        if !output_bytes.is_empty() {
            // SAFETY: shared-storage MTLBuffer::contents is CPU-visible after completion.
            let contents =
                unsafe { objc_msg_send_ptr(output_buffer.as_ptr(), selector(b"contents\0")) };
            if contents.is_null() {
                return Err(runtime_error(
                    Backend::Metal,
                    "Metal output buffer returned null contents",
                ));
            }
            // SAFETY: both slices cover at least output_bytes.len() bytes and do not overlap.
            unsafe {
                ptr::copy_nonoverlapping(
                    contents.cast::<u8>(),
                    output_bytes.as_mut_ptr(),
                    output_bytes.len(),
                )
            };
        }
        self.recycle_buffers();
        self.completed = true;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl Drop for MetalPendingProgram<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _pool = AutoreleasePool::new();
            // A dropped ticket must not release buffers/pipelines while the GPU may still
            // reference them. Errors cannot be returned from Drop; explicit `wait` remains
            // the status-reporting path.
            unsafe {
                objc_msg_send_void(
                    self.command_buffer.as_ptr(),
                    selector(b"waitUntilCompleted\0"),
                )
            };
            self.recycle_buffers();
            self.completed = true;
        }
    }
}

impl MetalProgramTicket32<'_> {
    pub fn wait(self) -> Result<Vec<Complex32>> {
        #[cfg(target_os = "macos")]
        {
            let mut pending = self.pending;
            pending.finish()?;
            pending.prepared.output_complex32()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = self;
            Err(unavailable(
                Backend::Metal,
                "Metal is available only on macOS",
            ))
        }
    }
}

impl MetalProgramTicket64<'_> {
    pub fn wait(self) -> Result<Vec<Complex64>> {
        let _ = self;
        Err(VkFftError::UnsupportedPrecision {
            backend: "Metal runtime",
            precision: "f64 compute",
        })
    }
}

impl crate::backend::native_runtime::NativeProgramTicket32 for MetalProgramTicket32<'_> {
    fn wait(self) -> Result<Vec<Complex32>> {
        MetalProgramTicket32::wait(self)
    }
}

impl crate::backend::native_runtime::NativeProgramTicket64 for MetalProgramTicket64<'_> {
    fn wait(self) -> Result<Vec<Complex64>> {
        MetalProgramTicket64::wait(self)
    }
}

impl crate::backend::native_runtime::NativeAsyncRuntime for MetalExecutionContext {
    type Ticket32<'a> = MetalProgramTicket32<'a>;
    type Ticket64<'a> = MetalProgramTicket64<'a>;

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

impl NativeRuntime for MetalExecutionContext {
    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn device_profile(&self) -> DeviceProfile {
        self.profile
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn compiled_pass_resource_reports(
        &self,
        source: &NativeProgramSource,
    ) -> Result<Vec<NativeCompiledPassResourceReport>> {
        source.validate()?;
        if source.backend != Backend::Metal {
            return Err(VkFftError::InvalidKernelIr(
                "Metal compiled-resource reporting requires a Metal native program",
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let mut reports = Vec::with_capacity(source.shaders.len());
            for (pass, shader) in source.program.passes.iter().zip(&source.shaders) {
                let pipeline = self.compile_pipeline(shader)?;
                reports.push(NativeCompiledPassResourceReport {
                    pass_name: pass.name.clone(),
                    metrics: self.compiled_pipeline_resource_metrics(&pipeline)?,
                });
            }
            Ok(reports)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(unavailable(
                Backend::Metal,
                "Metal is available only on macOS",
            ))
        }
    }

    fn execute_program_complex32(
        &self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        MetalExecutionContext::execute_program_complex32(self, source, input)
    }

    fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        MetalExecutionContext::execute_program_complex64(self, source, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_execution_context_implements_native_async_runtime() {
        fn assert_async<T: crate::backend::native_runtime::NativeAsyncRuntime>() {}
        assert_async::<MetalExecutionContext>();
    }

    #[test]
    fn metal_pipeline_cache_is_bounded_and_refreshes_reinsertions() {
        let mut cache = MetalPipelineCache::new(2);
        cache.insert("a".to_owned(), 1u32);
        cache.insert("b".to_owned(), 2u32);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("a"), Some(&1));
        cache.insert("a".to_owned(), 3u32);
        cache.insert("c".to_owned(), 4u32);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("a"), Some(&3));
        assert_eq!(cache.get("b"), None);
        assert_eq!(cache.get("c"), Some(&4));

        let mut disabled = MetalPipelineCache::new(0);
        disabled.insert("ignored".to_owned(), 5u32);
        assert_eq!(disabled.len(), 0);
    }

    #[test]
    fn metal_buffer_pool_is_bounded_and_matches_exact_sizes() {
        let mut pool = MetalBufferPool::new(2);
        pool.insert(8, 1u32);
        pool.insert(16, 2u32);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.take(8), Some(1));
        assert_eq!(pool.len(), 1);
        pool.insert(32, 3u32);
        pool.insert(64, 4u32);
        assert_eq!(pool.len(), 2);
        assert!(!pool.contains_size(16));
        assert!(pool.contains_size(32));
        assert!(pool.contains_size(64));
        assert_eq!(pool.take(32), Some(3));
        assert_eq!(pool.take(7), None);

        let mut disabled = MetalBufferPool::new(0);
        disabled.insert(8, 5u32);
        assert_eq!(disabled.len(), 0);
    }

    #[test]
    fn metal_lookup_buffer_cache_is_bounded_by_entries_and_bytes() {
        let mut cache = MetalLookupBufferCache::new(2, 14);
        cache.insert(vec![1, 2], 10u32);
        cache.insert(vec![3, 4], 20u32);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.retained_bytes(), 12);
        assert_eq!(cache.get(&[1, 2]), Some(&10));
        cache.insert(vec![1, 2], 11u32);
        cache.insert(vec![5, 6, 7, 8], 30u32);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&[1, 2]), None);
        assert_eq!(cache.get(&[5, 6, 7, 8]), Some(&30));
        assert_eq!(cache.retained_bytes(), 12);

        let mut oversized = MetalLookupBufferCache::new(4, 8);
        oversized.insert(vec![0; 3], 40u32);
        assert_eq!(oversized.len(), 0);
        assert_eq!(oversized.retained_bytes(), 0);

        let mut disabled = MetalLookupBufferCache::new(0, 64);
        disabled.insert(vec![9], 50u32);
        assert_eq!(disabled.len(), 0);
    }

    #[test]
    fn metal_p257_program_materializes_lookup_table_allocation() {
        let device = DeviceProfile::generic(Backend::Metal, GpuVendor::Apple);
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![257]),
            crate::Direction::Forward,
            device,
        )
        .unwrap();
        let source = crate::backend::native::NativeSourceBackend::new(Backend::Metal)
            .lower_transform(&ir)
            .unwrap();
        let memory = source.program.memory_plan().unwrap();
        let lookup_count = memory
            .allocations
            .iter()
            .filter(|allocation| {
                allocation.kind == crate::program_ir::ProgramAllocationKind::LookupTable
            })
            .count();
        assert!(
            lookup_count > 0,
            "Metal p257 must materialize a lookup allocation"
        );
    }

    #[test]
    fn metal_probe_is_platform_explicit_and_device_owned() {
        let availability = MetalRuntimeAdapter::probe();
        assert_eq!(availability.backend, Backend::Metal);
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!availability.available());
            assert_eq!(availability.device_count, 0);
        }
        #[cfg(target_os = "macos")]
        {
            assert!(availability.loader_available);
            assert_eq!(
                availability.compiler_available,
                availability.device_count > 0
            );
            assert!(!availability.detail.contains("xcrun"));
        }
    }

    #[test]
    fn metal_context_matches_probe_or_fails_soft() {
        let availability = MetalExecutionContext::probe();
        if availability.available() {
            let context = MetalExecutionContext::new(0).unwrap();
            assert_eq!(context.device_profile().backend, Backend::Metal);
            assert_eq!(context.device_profile().vendor, GpuVendor::Apple);
            assert!(!context.device_profile().supports_f64);
            assert!(!context.device_name().is_empty());
        } else {
            assert!(MetalExecutionContext::new(0).is_err());
        }
    }

    #[test]
    fn metal_stockham_matches_reference_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
        let availability = MetalExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Metal runtime gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = MetalExecutionContext::new(0).unwrap();
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
            panic!("Metal C2C transform returned a real output");
        };
        let oracle_input = input
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect::<Vec<_>>();
        let expected = crate::reference::dft(&oracle_input, crate::Direction::Forward, false);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error < 2.0e-3 * length as f64,
            "Metal F32 Stockham error {error:e} on {}",
            context.device_name()
        );
    }

    #[test]
    fn metal_compiled_resource_report_matches_pipeline_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
        let availability = MetalExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Metal resource-report gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = MetalExecutionContext::new(0).unwrap();
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = crate::backend::native::NativeSourceBackend::new(Backend::Metal)
            .lower_transform(&ir)
            .unwrap();
        let reports =
            crate::backend::native_runtime::NativeRuntime::compiled_pass_resource_reports(
                &context, &source,
            )
            .unwrap();
        assert_eq!(reports.len(), source.program.passes.len());
        for ((pass, shader), report) in source
            .program
            .passes
            .iter()
            .zip(&source.shaders)
            .zip(&reports)
        {
            assert_eq!(report.pass_name, pass.name);
            let crate::backend::native_runtime::NativeCompiledResourceMetrics::Metal {
                static_threadgroup_memory_bytes,
                max_threads_per_threadgroup,
                thread_execution_width,
            } = report.metrics
            else {
                panic!("Metal resource report returned the wrong metric variant");
            };
            assert!(
                static_threadgroup_memory_bytes >= shader.required_shared_memory_bytes,
                "compiled Metal static threadgroup memory {static_threadgroup_memory_bytes} is below typed requirement {}",
                shader.required_shared_memory_bytes
            );
            let requested_threads = shader.workgroup_size.x as usize
                * shader.workgroup_size.y as usize
                * shader.workgroup_size.z as usize;
            assert!(max_threads_per_threadgroup >= requested_threads);
            assert!(thread_execution_width > 0);
            assert!(thread_execution_width <= max_threads_per_threadgroup);
        }

        #[cfg(target_os = "macos")]
        let first_pipelines = {
            let cache = context.pipeline_cache.lock().unwrap();
            let mut pipelines = cache
                .entries
                .values()
                .map(|pipeline| pipeline.as_ptr() as usize)
                .collect::<Vec<_>>();
            pipelines.sort_unstable();
            assert!(!pipelines.is_empty());
            pipelines
        };
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.29 * x).sin(), (0.17 * x).cos())
            })
            .collect::<Vec<_>>();
        context.execute_program_complex32(&source, &input).unwrap();
        #[cfg(target_os = "macos")]
        {
            let cache = context.pipeline_cache.lock().unwrap();
            let mut pipelines = cache
                .entries
                .values()
                .map(|pipeline| pipeline.as_ptr() as usize)
                .collect::<Vec<_>>();
            pipelines.sort_unstable();
            assert_eq!(pipelines, first_pipelines);
        }
    }

    #[test]
    fn metal_pipeline_cache_reuses_stockham_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
        let availability = MetalExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Metal pipeline-cache gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = MetalExecutionContext::new(0).unwrap();
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = crate::backend::native::NativeSourceBackend::new(Backend::Metal)
            .lower_transform(&ir)
            .unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.19 * x).sin(), (0.05 * x).cos())
            })
            .collect::<Vec<_>>();
        context.execute_program_complex32(&source, &input).unwrap();
        #[cfg(target_os = "macos")]
        let first_entries = context.pipeline_cache.lock().unwrap().len();
        #[cfg(target_os = "macos")]
        {
            assert!(first_entries > 0);
            assert!(first_entries <= METAL_PIPELINE_CACHE_MAX_ENTRIES);
        }
        context.execute_program_complex32(&source, &input).unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(context.pipeline_cache.lock().unwrap().len(), first_entries);
    }

    #[test]
    fn metal_lut_cache_reuses_p257_buffers_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
        let availability = MetalExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Metal LUT-cache gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = MetalExecutionContext::new(0).unwrap();
        let length = 257usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = crate::backend::native::NativeSourceBackend::new(Backend::Metal)
            .lower_transform(&ir)
            .unwrap();
        let memory = source.program.memory_plan().unwrap();
        assert!(memory.allocations.iter().any(|allocation| {
            allocation.kind == crate::program_ir::ProgramAllocationKind::LookupTable
        }));
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.071 * x).sin(), (0.113 * x).cos())
            })
            .collect::<Vec<_>>();
        let oracle_input = input
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect::<Vec<_>>();
        let expected = crate::reference::dft(&oracle_input, crate::Direction::Forward, false);

        let first = context.execute_program_complex32(&source, &input).unwrap();
        let first_error = first
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
            })
            .fold(0.0f64, f64::max);
        assert!(
            first_error < 2.0e-3 * length as f64,
            "Metal p257 first-run error {first_error:e} on {}",
            context.device_name()
        );

        #[cfg(target_os = "macos")]
        let (first_entries, first_bytes, first_buffers) = {
            let cache = context.lut_cache.lock().unwrap();
            let mut buffers = cache
                .entries
                .values()
                .map(|buffer| buffer.as_ptr() as usize)
                .collect::<Vec<_>>();
            buffers.sort_unstable();
            (cache.len(), cache.retained_bytes(), buffers)
        };
        #[cfg(target_os = "macos")]
        {
            assert!(first_entries > 0);
            assert!(first_entries <= METAL_LUT_CACHE_MAX_ENTRIES);
            assert!(first_bytes > 0 && first_bytes <= METAL_LUT_CACHE_MAX_BYTES);
        }

        let second = context.execute_program_complex32(&source, &input).unwrap();
        let second_error = second
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
            })
            .fold(0.0f64, f64::max);
        assert!(
            second_error < 2.0e-3 * length as f64,
            "Metal p257 cached-run error {second_error:e} on {}",
            context.device_name()
        );
        #[cfg(target_os = "macos")]
        {
            let cache = context.lut_cache.lock().unwrap();
            let mut buffers = cache
                .entries
                .values()
                .map(|buffer| buffer.as_ptr() as usize)
                .collect::<Vec<_>>();
            buffers.sort_unstable();
            assert_eq!(cache.len(), first_entries);
            assert_eq!(cache.retained_bytes(), first_bytes);
            assert_eq!(buffers, first_buffers);
        }
    }

    #[test]
    fn metal_buffer_pool_reuses_stockham_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
        let availability = MetalExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Metal buffer-pool gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = MetalExecutionContext::new(0).unwrap();
        let length = 64usize;
        let ir = TransformIr::build(
            crate::FftConfig::new(vec![length]),
            crate::Direction::Forward,
            context.device_profile(),
        )
        .unwrap();
        let source = crate::backend::native::NativeSourceBackend::new(Backend::Metal)
            .lower_transform(&ir)
            .unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.23 * x).sin(), (0.03 * x).cos())
            })
            .collect::<Vec<_>>();
        context.execute_program_complex32(&source, &input).unwrap();
        #[cfg(target_os = "macos")]
        let first_buffers = context.buffer_pool.lock().unwrap().len();
        #[cfg(target_os = "macos")]
        {
            assert!(first_buffers > 0);
            assert!(first_buffers <= METAL_BUFFER_POOL_MAX_BUFFERS);
        }
        context.execute_program_complex32(&source, &input).unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(context.buffer_pool.lock().unwrap().len(), first_buffers);
    }

    #[test]
    fn metal_two_in_flight_stockham_matches_reference_or_skip() {
        let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
        let availability = MetalExecutionContext::probe();
        if !availability.available() {
            assert!(
                !require,
                "strict Metal async runtime gate is unavailable: {}",
                availability.detail
            );
            return;
        }
        let context = MetalExecutionContext::new(0).unwrap();
        let mut tickets = Vec::new();
        for length in [64usize, 96usize] {
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
            let oracle_input = input
                .iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect::<Vec<_>>();
            let expected = crate::reference::dft(&oracle_input, crate::Direction::Forward, false);
            let ticket = context
                .submit_transform_f32(&ir, NativeTransformInput32::Complex(&input))
                .unwrap();
            tickets.push((length, expected, ticket));
        }

        for (length, expected, ticket) in tickets.into_iter().rev() {
            let actual = ticket.wait().unwrap();
            let NativeTransformOutput32::Complex(actual) = actual else {
                panic!("Metal async C2C transform returned a real output");
            };
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| {
                    (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
                })
                .fold(0.0f64, f64::max);
            assert!(
                error < 2.0e-3 * length as f64,
                "Metal async F32 Stockham N{length} error {error:e} on {}",
                context.device_name()
            );
        }
    }
}
