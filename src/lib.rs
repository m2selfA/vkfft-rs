//! Pure-Rust reimplementation of VkFFT.
//!
//! `vkfft-rs` ports VkFFT's Application -> Plan -> Code architecture into typed Rust state:
//! configuration and device-aware planning, Stockham/Rader/Bluestein and real/R2R composition,
//! backend-neutral program/resource IR, generated GPU kernels, and correctness-first execution
//! runtimes. The fixed porting baseline is VkFFT 1.3.4 at upstream commit
//! `066a17c17068c0f11c9298d848c2976c71fad1c1`.
//!
//! # Default build
//!
//! The default feature set has no GPU runtime loader. It provides the planner, CPU reference
//! paths, typed IR, scheduler policies, LUT generation, and backend source/SPIR-V generation.
//! This makes planning and code generation usable without a CUDA, HIP, OpenCL, Level Zero,
//! Metal, or Vulkan runtime installed on the host.
//!
//! # Runtime features
//!
//! - `vulkan-runtime`: Vulkan 1.1 execution through `ash`, including explicit device selection,
//!   async tickets, pooled resources, resident program chaining, and versioned pipeline caches.
//! - `cuda-runtime`: CUDA Driver + NVRTC execution loaded dynamically at runtime.
//! - `hip-runtime`: HIP + HIPRTC/AMD compiler execution loaded dynamically at runtime.
//! - `opencl-runtime`: OpenCL execution loaded dynamically at runtime.
//! - `level-zero-runtime`: Intel Level Zero execution loaded dynamically at runtime. It can be
//!   combined with `opencl-runtime` to use the validated Intel native-binary compiler fallback
//!   when `ocloc` is unavailable.
//! - `metal-runtime`: correctness-first Metal F32/F16 execution on macOS; the runtime surface is
//!   implemented, while real Apple-GPU validation remains an explicit hardware gate.
//!
//! # Planning example
//!
//! ```rust
//! use vkfft_rs::{AlgorithmKind, FftConfig, FftPlan};
//!
//! let plan = FftPlan::build(FftConfig::new(vec![1024, 17, 65537]))?;
//! assert_eq!(plan.axes[0].algorithm.kind(), AlgorithmKind::Stockham);
//! assert_eq!(plan.axes[1].algorithm.kind(), AlgorithmKind::Rader);
//! assert_eq!(plan.axes[2].algorithm.kind(), AlgorithmKind::Bluestein);
//! # Ok::<(), vkfft_rs::VkFftError>(())
//! ```

pub mod application;
pub mod backend;
pub mod binary16;
pub mod bluestein_ir;
pub mod complex;
pub mod config;
pub mod convolution_ir;
pub mod double_double;
pub mod double_double_ir;
pub mod double_double_recursive_ir;
pub mod error;
pub mod kernel_ir;
pub mod lut;
pub mod mixed_ir;
pub mod nd_ir;
pub mod nd_real_ir;
pub mod one_dim_ir;
pub mod planner;
pub mod precision;
pub mod program_ir;
pub mod r2r_ir;
pub mod rader_ir;
pub mod real_ir;
pub mod recursive_ir;
pub mod reference;
pub mod scheduler;
pub mod scheduler_snapshot;
pub mod zero_pad_ir;

pub use application::TransformIr;
pub use binary16::{Binary16, Complex16, decode_complex16_native, encode_complex16_native};
pub use bluestein_ir::{
    BluesteinPassIr, BluesteinPassOperation, BluesteinPipelineIr, execute_bluestein_ir,
};
pub use complex::{Complex, Complex32, Complex64};
pub use config::{
    Backend, ConvolutionConjugation, DctType, DeviceProfile, Direction, DstType, FftConfig,
    GpuVendor, PLANNER_TUNING_PROFILE_VERSION, PlannerTuning, Precision, PrecisionCompute,
    PrecisionLayout, PrecisionStorage, SubgroupProfile, TransformKind, ZeroPaddingDomain,
    ZeroPaddingRange,
};
pub use convolution_ir::{
    ConvolutionDirectRaderMultiKernelStepIr, ConvolutionFftRaderMultiKernelStepIr,
    ConvolutionFftRaderStepIr, ConvolutionIr, ConvolutionMatrixLayout,
    ConvolutionMatrixStockhamStepIr, ConvolutionMultiKernelStockhamStepIr, ConvolutionMultiplyIr,
    ConvolutionMultiplyPolicy, ConvolutionThreeUploadMultiKernelStockhamIr,
    ConvolutionTwoUploadMultiKernelStockhamIr, NdConvolutionIr, NdRealConvolutionIr,
    execute_convolution_ir, execute_nd_convolution_ir, execute_nd_real_convolution_ir,
};
pub use double_double::{
    ComplexDoubleDouble, DoubleDouble, decode_complex_double_double, dft as double_double_dft,
    encode_complex_double_double, quick_two_sum, two_prod, two_sum,
    unit_root as double_double_unit_root,
};
pub use double_double_ir::{
    DoubleDoubleBluesteinConvolutionIr, DoubleDoubleBluesteinIr, DoubleDoubleDirectRaderIr,
    DoubleDoubleFftRaderIr, DoubleDoubleNdAxisIr, DoubleDoubleNdFftIr, DoubleDoubleNdR2rAxisIr,
    DoubleDoubleNdR2rIr, DoubleDoubleNdRealFftIr, DoubleDoubleOneDimIr, DoubleDoubleR2rAlgorithm,
    DoubleDoubleR2rIr, DoubleDoubleRealFftIr, DoubleDoubleStockhamIr,
    execute_double_double_bluestein_ir, execute_double_double_bluestein_ir_f64_storage,
    execute_double_double_c2r_ir, execute_double_double_c2r_ir_f64_storage,
    execute_double_double_direct_rader_ir, execute_double_double_direct_rader_ir_f64_storage,
    execute_double_double_fft_rader_ir, execute_double_double_fft_rader_ir_f64_storage,
    execute_double_double_nd_c2r_ir, execute_double_double_nd_c2r_ir_f64_storage,
    execute_double_double_nd_ir, execute_double_double_nd_ir_f64_storage,
    execute_double_double_nd_r2c_ir, execute_double_double_nd_r2c_ir_f64_storage,
    execute_double_double_nd_r2r_ir, execute_double_double_nd_r2r_ir_f64_storage,
    execute_double_double_one_dim_ir, execute_double_double_one_dim_ir_f64_storage,
    execute_double_double_r2c_ir, execute_double_double_r2c_ir_f64_storage,
    execute_double_double_r2r_ir, execute_double_double_r2r_ir_f64_storage,
    execute_double_double_stockham_ir, execute_double_double_stockham_ir_f64_storage,
};
pub use double_double_recursive_ir::{
    DoubleDoubleCooleyTukeyPassIr, DoubleDoubleCooleyTukeyPassOperation,
    DoubleDoubleRecursiveCooleyTukeyIr, DoubleDoubleRecursiveFftIr, DoubleDoubleRecursiveFftNodeIr,
    DoubleDoubleThreeUploadFourStepPlanIr, DoubleDoubleTwoUploadFourStepPlanIr,
    execute_double_double_recursive_ir, execute_double_double_recursive_ir_f64_storage,
};
pub use error::{Result, VkFftError};
pub use kernel_ir::{
    BufferAccess, BufferBinding, BufferRole, DispatchGeometry, FourStepMapping, KernelIr,
    KernelOperation, RaderGeneratorMapping, RaderScatterMapping, RealEvenInversePreprocessMapping,
    RealEvenPackMapping, RealEvenUnpackMapping, RegisterStageBoundary,
    RegisterStageBoundaryResidency, RegisterStockhamStage, RegisterSubgroupBoundary,
    RegisterSubgroupLaneModel, RegisterSubgroupSource, ScalarType, SharedBuffer, SharedMemoryPlan,
    StockhamExecutionLayout, StockhamInputModifier, StockhamIoMapping, StockhamOutputModifier,
    StockhamSharedMemoryLayout, StockhamStage, StockhamWorkgroupGrouping,
    ThreeUploadFourStepMapping, WorkgroupSize, execute_stockham_ir,
};
pub use lut::{
    BluesteinTable, DoubleDoubleBluesteinTable, DoubleDoubleRaderTable,
    DoubleDoubleStockhamTwiddleStage, DoubleDoubleStockhamTwiddleTable, RaderTable,
    StockhamTwiddleStage, StockhamTwiddleTable, stockham_root_table_double_double,
};
pub use mixed_ir::{
    MixedPassIr, MixedPassOperation, MixedPrimeRaderIr, MixedRaderStockhamIr,
    execute_mixed_rader_stockham_ir,
};
pub use nd_ir::{
    NdAxisIr, NdExternalTensorLayout, NdFftIr, NdFormattedCopyOperation, NdFormattedCopyPassIr,
    NdPassIr, NdPassOperation, execute_nd_fft_ir,
};
pub use nd_real_ir::{NdRealFftIr, execute_nd_c2r_ir, execute_nd_r2c_ir};
pub use one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
pub use planner::{
    AlgorithmKind, AxisAlgorithm, AxisPlan, FftPlan, RaderMode, RaderPrimePlan, RadixPlan,
};
pub use precision::{
    PRECISION_REPORT_SCHEMA_VERSION, PrecisionCaseReport, PrecisionMetrics,
    PrecisionTransformFamily, complex_precision_metrics, real_precision_metrics,
};
pub use program_ir::{
    ExternalBufferLayout, ProgramAllocation, ProgramAllocationId, ProgramAllocationKind, ProgramIr,
    ProgramMemoryPlan, ProgramPass, ProgramPassBinding, ProgramResource, ProgramResourceId,
    ProgramResourceInitialization, ProgramResourceKind,
};
pub use r2r_ir::{
    NdR2rAxisIr, NdR2rIr, R2rFftPassIr, R2rFftPassOperation, R2rFftReductionIr, R2rIr, R2rNdPassIr,
    R2rNdPassOperation, R2rTransform, execute_nd_r2r_ir, execute_r2r_ir,
};
pub use rader_ir::{
    RaderDirectIr, RaderFftInputStrategy, RaderFftPassIr, RaderFftPassOperation,
    RaderFftPipelineIr, execute_rader_direct_ir, execute_rader_fft_ir,
};
pub use real_ir::{
    RealFftAlgorithm, RealFftIr, RealFftKind, RealPassIr, RealPassOperation, execute_c2r_ir,
    execute_r2c_ir,
};
pub use recursive_ir::{
    CooleyTukeyPassIr, CooleyTukeyPassOperation, FourStepInputLayout, FourStepOutputLayout,
    FourStepPlanIr, FourStepTwiddlePlacement, FourStepUploadIr, FusedFftRaderStaticResourceReport,
    RecursiveCooleyTukeyIr, RecursiveFftIr, RecursiveFftNodeIr, execute_recursive_fft_ir,
};
pub use scheduler::{
    DoubleDoubleStockhamUploadSchedule, GpuSchedulerPolicy, NvidiaVulkanSchedulerTuning,
    RaderFftRegisterSchedule, RaderFftStageLaneLayout, RaderFftStageLaneSchedule,
    RaderFftTransposeSchedule, RadixRegisterSchedule, StockhamUploadSchedule,
    VKFFT_RADIX_TABLE_LEN, has_specialized_gpu_scheduler_policy,
    plan_gpu_double_double_stockham_uploads_for_batches, plan_gpu_force_rader_two_upload,
    plan_gpu_power_of_two_radix_registers, plan_gpu_power_of_two_stockham_uploads,
    plan_gpu_power_of_two_stockham_uploads_for_batches, plan_gpu_rader_fft_registers,
    plan_gpu_rader_fft_registers_for_containers, plan_gpu_scheduler_policy,
    plan_gpu_small_mixed_radix_registers, plan_gpu_smooth_stockham_uploads,
    plan_gpu_smooth_stockham_uploads_for_batches, plan_gpu_stockham_twiddle_source,
    plan_nvidia_vulkan_power_of_two_radix_registers,
    plan_nvidia_vulkan_power_of_two_stockham_uploads,
    plan_nvidia_vulkan_power_of_two_stockham_uploads_for_batches,
    plan_nvidia_vulkan_rader_fft_registers, plan_nvidia_vulkan_rader_fft_registers_for_containers,
    plan_nvidia_vulkan_small_mixed_radix_registers, plan_nvidia_vulkan_smooth_stockham_uploads,
    plan_nvidia_vulkan_smooth_stockham_uploads_for_batches,
    plan_nvidia_vulkan_two_three_radix_registers,
};
pub use scheduler_snapshot::{
    SCHEDULER_SNAPSHOT_SCHEMA_VERSION, SchedulerAxisClassification,
    SchedulerPhysicalAxisProbeContext, SchedulerRaderUploadProbe, SchedulerSnapshotMismatch,
    SchedulerSnapshotPayload, SchedulerSnapshotRecord, compare_upstream_scheduler_snapshot_jsonl,
    fixed_upstream_scheduler_snapshot_corpus, scheduler_axis_classification_probe,
    scheduler_rader_upload_axis_block_probe_records,
    scheduler_rader_upload_axis_block_probe_records_with_context,
    scheduler_rader_upload_probe_record,
};

pub use zero_pad_ir::{
    NdZeroPadPassIr, ZeroPadPassIr, ZeroPadPassOperation, execute_nd_zero_pad_pass,
    execute_zero_pad_pass,
};

/// VkFFT release encoded by the upstream header used as the initial porting baseline.
pub const UPSTREAM_VKFFT_VERSION: &str = "1.3.4";

/// Upstream VkFFT commit used as the initial porting baseline.
pub const UPSTREAM_VKFFT_COMMIT: &str = "066a17c17068c0f11c9298d848c2976c71fad1c1";
