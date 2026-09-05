//! Backend-neutral mixed Stockham × Rader axis decomposition.
//!
//! The first mixed slice handles `N = A * p`, where `A > 1` contains only
//! Stockham-supported factors and `p` is one planner-selected Rader prime with
//! multiplicity one. It uses a Cooley-Tukey factorization: batched p-point Rader
//! transforms, an N-point twiddle matrix, then batched A-point Stockham transforms.

use core::f64::consts::TAU;

use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, FftConfig, Precision, TransformKind};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    DispatchGeometry, KernelIr, ScalarType, WorkgroupSize, execute_stockham_ir,
};
use crate::planner::{AxisAlgorithm, FftPlan, RaderMode};
use crate::rader_ir::{
    RaderDirectIr, RaderFftPipelineIr, execute_rader_direct_ir, execute_rader_fft_ir,
};
use crate::scheduler::{has_specialized_gpu_scheduler_policy, plan_gpu_rader_upload_split};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedPassOperation {
    PackPrimeInput,
    TwiddleTranspose,
    ScatterOutput,
}

#[derive(Debug, Clone, Copy)]
struct MixedPassShape {
    scalar: ScalarType,
    direction: Direction,
    logical_len: usize,
    stockham_len: usize,
    prime: usize,
    batch_count: usize,
    device: DeviceProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub direction: Direction,
    pub logical_len: usize,
    pub stockham_len: usize,
    pub prime: usize,
    pub batch_count: usize,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub operation: MixedPassOperation,
}

impl MixedPassIr {
    fn new(name: String, shape: MixedPassShape, operation: MixedPassOperation) -> Result<Self> {
        if shape.device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let local_size = shape
            .logical_len
            .min(shape.device.max_threads_per_block)
            .max(1);
        let workgroup_x = u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
            field: "mixed Stockham/Rader workgroup size",
        })?;
        let dispatch_x =
            u32::try_from(shape.batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                field: "mixed Stockham/Rader dispatch workgroup count",
            })?;
        let pass = Self {
            name,
            scalar: shape.scalar,
            direction: shape.direction,
            logical_len: shape.logical_len,
            stockham_len: shape.stockham_len,
            prime: shape.prime,
            batch_count: shape.batch_count,
            workgroup_size: WorkgroupSize {
                x: workgroup_x,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: dispatch_x,
                y: 1,
                z: 1,
            },
            operation,
        };
        pass.validate()?;
        Ok(pass)
    }

    pub fn validate(&self) -> Result<()> {
        if self.stockham_len <= 1
            || self.prime < 2
            || self.batch_count == 0
            || self.stockham_len.checked_mul(self.prime) != Some(self.logical_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "mixed pass dimensions are inconsistent",
            ));
        }
        if self.workgroup_size.x == 0
            || self.workgroup_size.y == 0
            || self.workgroup_size.z == 0
            || self.dispatch.x as usize != self.batch_count
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "mixed pass workgroup/dispatch metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

// Preserve the public mixed-Rader IR and its allocation-free Direct-Rader case;
// boxing it only to equalize enum sizes would add avoidable planner allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum MixedPrimeRaderIr {
    Direct(RaderDirectIr),
    FftConvolution(Box<RaderFftPipelineIr>),
}

impl MixedPrimeRaderIr {
    fn prime(&self) -> usize {
        match self {
            Self::Direct(ir) => ir.prime,
            Self::FftConvolution(ir) => ir.prime,
        }
    }

    fn batch_count(&self) -> usize {
        match self {
            Self::Direct(ir) => ir.batch_count,
            Self::FftConvolution(ir) => ir.batch_count,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Direct(ir) => ir.validate(),
            Self::FftConvolution(ir) => ir.validate(),
        }
    }

    fn execute(&self, input: &[Complex64]) -> Result<Vec<Complex64>> {
        match self {
            Self::Direct(ir) => execute_rader_direct_ir(ir, input),
            Self::FftConvolution(ir) => execute_rader_fft_ir(ir, input),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MixedRaderStockhamIr {
    pub logical_len: usize,
    pub stockham_len: usize,
    pub prime: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    pub pack_prime: MixedPassIr,
    pub prime_stage: MixedPrimeRaderIr,
    pub twiddle_transpose: MixedPassIr,
    pub stockham_stage: KernelIr,
    pub scatter_output: MixedPassIr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MixedRaderFourStepContext {
    upload_count: usize,
    axis_upload_id: usize,
    stage_start_size: usize,
}

fn physical_mixed_rader_container_context(
    logical_len: usize,
    stockham_len: usize,
    prime: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<(usize, usize, Option<MixedRaderFourStepContext>)> {
    if precision != Precision::F32 || !has_specialized_gpu_scheduler_policy(device) {
        return Ok((logical_len, stockham_len, None));
    }
    let Some(upload_schedule) =
        plan_gpu_rader_upload_split(logical_len, &[prime], &[], precision, device)?
    else {
        return Ok((logical_len, stockham_len, None));
    };
    let mut prime_uploads = upload_schedule
        .axis_split
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, factor)| factor.is_multiple_of(prime));
    let (axis_upload_id, physical_outer_fft_len) =
        prime_uploads.next().ok_or(VkFftError::InvalidKernelIr(
            "mixed Rader multi-upload split lost the FFT-Rader prime component",
        ))?;
    if prime_uploads.next().is_some() {
        return Err(VkFftError::InvalidKernelIr(
            "mixed Rader multi-upload split duplicated the FFT-Rader prime component",
        ));
    }
    let physical_container_fft_num = physical_outer_fft_len / prime;
    if physical_container_fft_num == 0
        || physical_container_fft_num > stockham_len
        || !stockham_len.is_multiple_of(physical_container_fft_num)
    {
        return Err(VkFftError::InvalidKernelIr(
            "mixed Rader multi-upload split produced an inconsistent physical container count",
        ));
    }
    let stage_start_size = upload_schedule.axis_split[..axis_upload_id]
        .iter()
        .try_fold(1usize, |acc, factor| {
            acc.checked_mul(*factor)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "mixed Rader Four-step stage-start product",
                })
        })?;
    Ok((
        physical_outer_fft_len,
        physical_container_fft_num,
        Some(MixedRaderFourStepContext {
            upload_count: upload_schedule.upload_count,
            axis_upload_id,
            stage_start_size,
        }),
    ))
}

impl MixedRaderStockhamIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial mixed Stockham/Rader IR supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial mixed Stockham/Rader IR supports C2C transforms only",
            ));
        }
        let scalar = match plan.config.precision {
            Precision::F32 => ScalarType::F32,
            Precision::F64 if device.supports_f64 => ScalarType::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "mixed Stockham/Rader IR",
                    precision: precision_name(other),
                });
            }
        };
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing mixed Stockham/Rader axis plan",
        ))?;
        let AxisAlgorithm::Rader { stockham, primes } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "mixed Stockham/Rader IR requires a planner-selected Rader axis",
            ));
        };
        if primes.len() != 1 || primes[0].multiplicity != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial mixed Stockham/Rader IR supports exactly one Rader prime with multiplicity one",
            ));
        }
        if stockham.prime_factors.is_empty() {
            return Err(VkFftError::UnsupportedKernelPath(
                "prime-only Rader axes should use the dedicated Rader IR",
            ));
        }
        let stockham_len = stockham
            .prime_factors
            .iter()
            .try_fold(1usize, |acc, factor| {
                acc.checked_mul(*factor)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "mixed Stockham factor product",
                    })
            })?;
        let prime = primes[0].prime;
        let logical_len =
            stockham_len
                .checked_mul(prime)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "mixed Stockham/Rader logical length",
                })?;
        if logical_len != axis.effective_fft_len {
            return Err(VkFftError::InvalidKernelIr(
                "mixed Stockham/Rader factorization does not cover the axis length",
            ));
        }

        let prime_batch_count = plan.config.batch_count.checked_mul(stockham_len).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "mixed Rader batch count",
            },
        )?;
        let stockham_batch_count =
            plan.config
                .batch_count
                .checked_mul(prime)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "mixed Stockham batch count",
                })?;
        let prime_config = FftConfig::new(vec![prime])
            .with_batch_count(prime_batch_count)
            .with_precision(plan.config.precision)
            .with_inverse_normalization(plan.config.normalize_inverse)
            .with_tuning(plan.config.tuning);
        let prime_plan = FftPlan::build(prime_config)?;
        let prime_stage = match primes[0].mode {
            RaderMode::DirectMultiplication => {
                MixedPrimeRaderIr::Direct(RaderDirectIr::build(&prime_plan, direction, device)?)
            }
            RaderMode::FftConvolution { .. } => {
                let (physical_outer_fft_len, physical_container_fft_num, four_step_context) =
                    physical_mixed_rader_container_context(
                        logical_len,
                        stockham_len,
                        prime,
                        plan.config.precision,
                        device,
                    )?;
                let pipeline = if let Some(context) = four_step_context {
                    RaderFftPipelineIr::build_with_container_context_and_outer_four_step(
                        &prime_plan,
                        direction,
                        device,
                        physical_outer_fft_len,
                        physical_container_fft_num,
                        context.upload_count,
                        context.axis_upload_id,
                        context.stage_start_size,
                        plan.config.batch_count,
                        plan.config.zero_padding_for_axis(0).is_some(),
                        plan.config.grouped_batch_for_axis(0),
                        plan.config.precision,
                    )?
                } else {
                    RaderFftPipelineIr::build_with_container_context(
                        &prime_plan,
                        direction,
                        device,
                        physical_outer_fft_len,
                        physical_container_fft_num,
                    )?
                };
                MixedPrimeRaderIr::FftConvolution(Box::new(pipeline))
            }
        };

        let stockham_config = FftConfig::new(vec![stockham_len])
            .with_batch_count(stockham_batch_count)
            .with_precision(plan.config.precision)
            .with_inverse_normalization(plan.config.normalize_inverse)
            .with_tuning(plan.config.tuning);
        let stockham_plan = FftPlan::build(stockham_config)?;
        let stockham_stage = KernelIr::stockham_1d(&stockham_plan, direction, device)?;
        let direction_name = match direction {
            Direction::Forward => "forward",
            Direction::Inverse => "inverse",
        };
        let outer_batch_count = plan.config.batch_count;
        let pass_shape = MixedPassShape {
            scalar,
            direction,
            logical_len,
            stockham_len,
            prime,
            batch_count: outer_batch_count,
            device,
        };
        let pack_prime = MixedPassIr::new(
            format!("vkfft_mixed_pack_{logical_len}_{direction_name}"),
            pass_shape,
            MixedPassOperation::PackPrimeInput,
        )?;
        let twiddle_transpose = MixedPassIr::new(
            format!("vkfft_mixed_twiddle_transpose_{logical_len}_{direction_name}"),
            pass_shape,
            MixedPassOperation::TwiddleTranspose,
        )?;
        let scatter_output = MixedPassIr::new(
            format!("vkfft_mixed_scatter_{logical_len}_{direction_name}"),
            pass_shape,
            MixedPassOperation::ScatterOutput,
        )?;
        let ir = Self {
            logical_len,
            stockham_len,
            prime,
            batch_count: outer_batch_count,
            direction,
            scalar,
            pack_prime,
            prime_stage,
            twiddle_transpose,
            stockham_stage,
            scatter_output,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.stockham_len <= 1
            || self.prime < 2
            || self.batch_count == 0
            || self.stockham_len.checked_mul(self.prime) != Some(self.logical_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "mixed Stockham/Rader dimensions are inconsistent",
            ));
        }
        self.pack_prime.validate()?;
        self.prime_stage.validate()?;
        self.twiddle_transpose.validate()?;
        self.stockham_stage.validate()?;
        self.scatter_output.validate()?;
        for pass in [
            &self.pack_prime,
            &self.twiddle_transpose,
            &self.scatter_output,
        ] {
            if pass.logical_len != self.logical_len
                || pass.stockham_len != self.stockham_len
                || pass.prime != self.prime
                || pass.batch_count != self.batch_count
                || pass.direction != self.direction
                || pass.scalar != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "mixed outer pass metadata is inconsistent",
                ));
            }
        }
        if self.pack_prime.operation != MixedPassOperation::PackPrimeInput
            || self.twiddle_transpose.operation != MixedPassOperation::TwiddleTranspose
            || self.scatter_output.operation != MixedPassOperation::ScatterOutput
            || self.prime_stage.prime() != self.prime
            || self.prime_stage.batch_count() != self.batch_count * self.stockham_len
            || self.stockham_stage.sequence_len != self.stockham_len
            || self.stockham_stage.batch_count != self.batch_count * self.prime
            || self.stockham_stage.direction != self.direction
        {
            return Err(VkFftError::InvalidKernelIr(
                "mixed Stockham/Rader nested stage metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

pub fn execute_mixed_rader_stockham_ir(
    ir: &MixedRaderStockhamIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let expected =
        ir.logical_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "mixed Stockham/Rader input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }

    // n = n1 + A*n2. Pack one p-point transform for every n1 in every batch.
    let mut prime_input = vec![Complex64::new(0.0, 0.0); expected];
    for batch in 0..ir.batch_count {
        let source_base = batch * ir.logical_len;
        for n1 in 0..ir.stockham_len {
            let transform = batch * ir.stockham_len + n1;
            let destination_base = transform * ir.prime;
            for n2 in 0..ir.prime {
                prime_input[destination_base + n2] = input[source_base + n1 + ir.stockham_len * n2];
            }
        }
    }
    let mut prime_output = ir.prime_stage.execute(&prime_input)?;

    // Cooley-Tukey twiddle W_N^(n1*k2).
    let sign = ir.direction.exponent_sign();
    for batch in 0..ir.batch_count {
        for n1 in 0..ir.stockham_len {
            let transform = batch * ir.stockham_len + n1;
            let base = transform * ir.prime;
            for k2 in 0..ir.prime {
                let angle = sign * TAU * (n1 * k2) as f64 / ir.logical_len as f64;
                prime_output[base + k2] *= Complex64::exp_i(angle);
            }
        }
    }

    // Transpose into p batches of A-point Stockham transforms.
    let mut stockham_input = vec![Complex64::new(0.0, 0.0); expected];
    for batch in 0..ir.batch_count {
        for k2 in 0..ir.prime {
            let destination_base = (batch * ir.prime + k2) * ir.stockham_len;
            for n1 in 0..ir.stockham_len {
                let source = (batch * ir.stockham_len + n1) * ir.prime + k2;
                stockham_input[destination_base + n1] = prime_output[source];
            }
        }
    }
    let stockham_output = execute_stockham_ir(&ir.stockham_stage, &stockham_input)?;

    // k = k2 + p*k1.
    let mut output = vec![Complex64::new(0.0, 0.0); expected];
    for batch in 0..ir.batch_count {
        for k2 in 0..ir.prime {
            let source_base = (batch * ir.prime + k2) * ir.stockham_len;
            for k1 in 0..ir.stockham_len {
                output[batch * ir.logical_len + k2 + ir.prime * k1] =
                    stockham_output[source_base + k1];
            }
        }
    }
    Ok(output)
}

fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F16StorageF32Compute => "f16-storage/f32-compute",
        Precision::F32 => "f32",
        Precision::F64 => "f64",
        Precision::F64ComputeF32Storage => "f64-compute/f32-storage",
        Precision::DoubleDouble => "double-double",
        Precision::DoubleDoubleF64Storage => "double-double/f64-storage",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor};
    use crate::reference::dft;

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn sample(length: usize, batch_count: usize) -> Vec<Complex64> {
        (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.071 * x).sin() + x * 0.001, (0.037 * x).cos() - x * 0.002)
            })
            .collect()
    }

    fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    #[test]
    fn mixed_stockham_fft_rader_matches_dft() {
        for length in [34usize, 68, 136, 514, 1028] {
            check_mixed_length(length);
        }
    }

    #[test]
    fn grouped_two_container_rader_fft_matches_independent_cpu_execution() {
        let outer_batch_count = 2usize;
        let mixed_plan =
            FftPlan::build(FftConfig::new(vec![514]).with_batch_count(outer_batch_count)).unwrap();
        let mixed = MixedRaderStockhamIr::build(&mixed_plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(grouped) = &mixed.prime_stage else {
            panic!("514 should use FFT-convolution Rader");
        };
        let crate::RecursiveFftNodeIr::Stockham(grouped_kernel) =
            &grouped.forward_recursive().unwrap().root
        else {
            panic!("grouped Rader convolution should remain a Stockham root");
        };
        assert_eq!(
            grouped_kernel.workgroup_grouping.transforms_per_workgroup,
            2
        );
        assert_eq!(grouped_kernel.workgroup_grouping.threads_per_transform, 16);
        assert_eq!(grouped_kernel.workgroup_size.x, 32);
        assert_eq!(grouped_kernel.dispatch.x, outer_batch_count as u32);
        let subgroup_boundaries = grouped_kernel
            .register_subgroup_boundaries(32)
            .unwrap()
            .unwrap();
        assert!(subgroup_boundaries.iter().all(Option::is_some));
        for boundary in subgroup_boundaries.into_iter().flatten() {
            assert_eq!(boundary.active_lanes, 32);
            assert_eq!(boundary.transforms_per_subgroup, 2);
            assert_eq!(boundary.lanes_per_transform, 16);
            for slot in boundary.target_to_source {
                assert_eq!(slot.len(), 32);
                for (target_lane, source) in slot.into_iter().enumerate() {
                    assert_eq!(
                        source.lane / 16,
                        target_lane / 16,
                        "subgroup shuffle ownership must not cross Rader containers"
                    );
                }
            }
        }
        let wider_subgroup = grouped_kernel
            .register_subgroup_boundaries(64)
            .unwrap()
            .unwrap();
        assert!(wider_subgroup.iter().all(Option::is_none));

        let internal_batch_count = outer_batch_count * 2;
        let independent_plan =
            FftPlan::build(FftConfig::new(vec![257]).with_batch_count(internal_batch_count))
                .unwrap();
        let independent =
            RaderFftPipelineIr::build(&independent_plan, Direction::Forward, device()).unwrap();
        let crate::RecursiveFftNodeIr::Stockham(independent_kernel) =
            &independent.forward_recursive().unwrap().root
        else {
            panic!("independent Rader convolution should remain a Stockham root");
        };
        assert_eq!(
            independent_kernel
                .workgroup_grouping
                .transforms_per_workgroup,
            internal_batch_count
        );
        assert_eq!(
            independent_kernel.workgroup_grouping.threads_per_transform,
            17
        );
        assert_eq!(
            [
                independent_kernel.workgroup_size.x,
                independent_kernel.workgroup_size.y,
            ],
            [17, internal_batch_count as u32]
        );
        assert_eq!(independent_kernel.dispatch.x, 1);

        let input = sample(257, internal_batch_count);
        let grouped_output = execute_stockham_ir(grouped_kernel.as_ref(), &input).unwrap();
        let independent_output = execute_stockham_ir(independent_kernel.as_ref(), &input).unwrap();
        assert!(max_error(&grouped_output, &independent_output) < 1.0e-10);
    }

    #[test]
    fn four_container_p257_reuses_the_same_subgroup_local_ownership_proof() {
        let plan = FftPlan::build(FftConfig::new(vec![1028]).with_batch_count(2)).unwrap();
        let mixed = MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &mixed.prime_stage else {
            panic!("1028 should use FFT-convolution Rader");
        };
        let crate::RecursiveFftNodeIr::Stockham(kernel) = &rader.forward_recursive().unwrap().root
        else {
            panic!("grouped p257 convolution should remain a Stockham root");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 4);
        assert_eq!(kernel.workgroup_grouping.threads_per_transform, 16);
        assert_eq!(kernel.workgroup_size.x, 64);
        let boundaries = kernel
            .register_subgroup_boundaries(32)
            .unwrap()
            .expect("register schedule should expose subgroup boundaries");
        assert!(boundaries.iter().all(Option::is_some));
        for boundary in boundaries.into_iter().flatten() {
            assert_eq!(boundary.subgroup_size, 32);
            assert_eq!(boundary.active_lanes, 32);
            assert_eq!(boundary.transforms_per_subgroup, 2);
            assert_eq!(boundary.lanes_per_transform, 16);
            for slot in boundary.target_to_source {
                assert_eq!(slot.len(), 32);
                for (target_lane, source) in slot.into_iter().enumerate() {
                    assert_eq!(source.lane / 16, target_lane / 16);
                    assert!(source.lane < 32);
                }
            }
        }
    }

    #[test]
    fn grouped_two_container_rader_shared_padding_respects_total_budget() {
        let mut constrained = device();
        constrained.shared_memory_bytes = 4 * 1024;
        constrained.shared_memory_pow2_bytes = 4 * 1024;
        let plan = FftPlan::build(FftConfig::new(vec![514]).with_batch_count(2)).unwrap();
        let mixed = MixedRaderStockhamIr::build(&plan, Direction::Forward, constrained).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &mixed.prime_stage else {
            panic!("514 should use FFT-convolution Rader");
        };
        let crate::RecursiveFftNodeIr::Stockham(kernel) = &rader.forward_recursive().unwrap().root
        else {
            panic!("grouped Rader convolution should remain a Stockham root");
        };
        let layout = kernel.stockham_shared_layout.unwrap();
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 2);
        assert_eq!(layout.logical_elements, 256);
        assert_eq!(layout.allocated_elements, 256);
        assert_eq!(layout.first_stage_stride, 256);
        assert_eq!(layout.read_write_stride, 256);
        assert_eq!(kernel.shared_memory.elements_per_buffer, 512);
        assert_eq!(kernel.required_shared_memory_bytes().unwrap(), 4 * 1024);
    }

    #[test]
    fn grouped_one_stage_rader_scales_to_four_containers() {
        let plan = FftPlan::build(FftConfig::new(vec![68]).with_batch_count(2)).unwrap();
        let ir = MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("68 should use FFT-convolution Rader for its 17 factor");
        };
        let schedule = rader
            .internal_register_schedule
            .as_ref()
            .expect("four p17 containers should carry the power-of-two outer schedule");
        assert_eq!(schedule.container_fft_num, 4);
        assert_eq!(schedule.execution_container_fft_num, 4);
        assert_eq!(schedule.min_rader_fft_thread_num, 4);
        assert_eq!(schedule.execution_threads_per_workgroup, 4);
        assert!(schedule.rader_transpose.is_none());
        let crate::RecursiveFftNodeIr::Stockham(kernel) = &rader.forward_recursive().unwrap().root
        else {
            panic!("grouped p17 convolution should remain a Stockham root");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 4);
        assert_eq!(kernel.workgroup_grouping.threads_per_transform, 1);
        assert_eq!(kernel.workgroup_size.x, 4);
        assert_eq!(kernel.dispatch.x, 2);

        let input = sample(68, 2);
        let actual = execute_mixed_rader_stockham_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for batch in 0..2 {
            let start = batch * 68;
            expected.extend(dft(&input[start..start + 68], Direction::Forward, false));
        }
        assert!(max_error(&actual, &expected) < 1.0e-8 * 68.0);
    }

    #[test]
    fn eight_container_p257_rader_transpose_keeps_mixed_fft_correct() {
        let length = 2056usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("2056 should use FFT-convolution Rader for its 257 factor");
        };
        let schedule = rader
            .internal_register_schedule
            .as_ref()
            .expect("2056 should carry the eight-container p257 schedule");
        assert_eq!(schedule.container_fft_num, 8);
        assert_eq!(schedule.execution_container_fft_num, 8);
        assert_eq!(schedule.execution_threads_per_workgroup, 128);
        assert_eq!(schedule.execution_workgroup_count, 1);
        assert!(schedule.upstream_grouping_is_executable());
        let transpose = schedule
            .rader_transpose
            .as_ref()
            .expect("2056 should enable the first executable raderTranspose slice");
        assert_eq!(transpose.container_fft_num, 8);
        assert_eq!(transpose.workgroup_threads, 128);

        for fft in [
            rader.forward_recursive().unwrap(),
            rader.inverse_recursive().unwrap(),
        ] {
            let crate::RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("transposed p257 convolution should remain one Stockham root");
            };
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 8);
            assert_eq!(kernel.workgroup_grouping.threads_per_transform, 16);
            assert_eq!(kernel.workgroup_size.x, 128);
            assert_eq!(kernel.dispatch.x, 1);
            assert_eq!(kernel.rader_transpose.as_ref(), Some(transpose));
            assert_eq!(kernel.shared_memory.elements_per_buffer, 256 * 8);
            assert_eq!(
                kernel.stockham_shared_layout.unwrap().allocated_elements,
                256
            );
        }

        let input = sample(length, 1);
        let actual = execute_mixed_rader_stockham_ir(&ir, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(
            max_error(&actual, &expected) < 1.5e-8 * length as f64,
            "2056-point mixed FFT mismatch with executable raderTranspose lane geometry"
        );

        let inverse_plan =
            FftPlan::build(FftConfig::new(vec![length]).with_inverse_normalization(true)).unwrap();
        let inverse =
            MixedRaderStockhamIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
        let restored = execute_mixed_rader_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) < 2.0e-8 * length as f64);
    }

    #[test]
    fn sixteen_container_p257_rader_transpose_matches_independent_internal_ffts() {
        let length = 4112usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("4112 should use FFT-convolution Rader for its 257 factor");
        };
        let schedule = rader
            .internal_register_schedule
            .as_ref()
            .expect("4112 should carry the sixteen-container p257 schedule");
        assert_eq!(schedule.container_fft_num, 16);
        assert_eq!(schedule.execution_container_fft_num, 16);
        assert_eq!(schedule.execution_threads_per_workgroup, 256);
        assert_eq!(schedule.execution_workgroup_count, 1);
        let transpose = schedule
            .rader_transpose
            .as_ref()
            .expect("4112 should use the generalized p257 raderTranspose family");
        assert_eq!(transpose.workgroup_threads, 256);
        let crate::RecursiveFftNodeIr::Stockham(kernel) = &rader.forward_recursive().unwrap().root
        else {
            panic!("16-container transposed p257 convolution should remain one Stockham root");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 16);
        assert_eq!(kernel.workgroup_size.x, 256);
        assert_eq!(kernel.dispatch.x, 1);
        assert_eq!(kernel.shared_memory.elements_per_buffer, 256 * 16);
        assert_eq!(kernel.rader_transpose.as_ref(), Some(transpose));

        assert_eq!(
            rader.input_strategy,
            crate::rader_ir::RaderFftInputStrategy::GeneratorOrderStockham
        );
        assert!(matches!(
            kernel.io_mapping,
            crate::StockhamIoMapping::RaderGeneratorReverse(_)
        ));

        // The fused forward kernel now consumes the original 257-value containers
        // directly. Compare it with the old semantic boundary by constructing the
        // exact GatherReverse buffer and running an independent contiguous FFT.
        let prime_input = sample(257, 16);
        let grouped = execute_stockham_ir(kernel.as_ref(), &prime_input).unwrap();
        let mut gathered = vec![Complex64::new(0.0, 0.0); 256 * 16];
        for batch in 0..16 {
            for slot in 0..256 {
                let exponent = (256 - slot) % 256;
                gathered[batch * 256 + slot] =
                    prime_input[batch * 257 + rader.table.permutation[exponent]];
            }
        }
        let independent_plan =
            FftPlan::build(FftConfig::new(vec![256]).with_batch_count(16)).unwrap();
        let independent =
            crate::KernelIr::stockham_1d(&independent_plan, Direction::Forward, device()).unwrap();
        let independent_output = execute_stockham_ir(&independent, &gathered).unwrap();
        assert!(max_error(&grouped, &independent_output) < 1.0e-10);
    }

    #[test]
    fn amd_32k_n4112_capacity_split_keeps_p257_upload_single_container() {
        let profile = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let length = 16usize * 257;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = MixedRaderStockhamIr::build(&plan, Direction::Forward, profile).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("AMD 32 KiB N4112 should keep p257 FFT-convolution Rader");
        };
        let schedule = rader
            .internal_register_schedule
            .as_ref()
            .expect("AMD 32 KiB N4112 p257 upload should retain a register schedule");
        assert_eq!(schedule.outer_fft_len, 257);
        assert_eq!(schedule.container_fft_num, 1);
        assert_eq!(schedule.execution_container_fft_num, 1);
        assert_eq!(schedule.execution_threads_per_workgroup, 16);
        assert_eq!(schedule.execution_workgroup_count, 16);
        assert!(schedule.rader_transpose.is_none());
        let block = rader
            .axis_batch_block
            .expect("AMD 32 KiB p257 upload should retain the upstream axis block");
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 8);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [8, 17]);
        for fft in [
            rader.forward_recursive().unwrap(),
            rader.inverse_recursive().unwrap(),
        ] {
            let crate::RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("AMD split p257 convolution should remain one Stockham root");
            };
            assert!(kernel.rader_transpose.is_none());
            assert!(kernel.required_shared_memory_bytes().unwrap() <= 32 * 1024);
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 8);
            assert_eq!(kernel.workgroup_grouping.threads_per_transform, 17);
            assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [8, 17]);
            assert_eq!(kernel.dispatch.x, 2);
        }

        let input = sample(length, 1);
        let actual = execute_mixed_rader_stockham_ir(&ir, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 1.5e-8 * length as f64);
    }

    #[test]
    fn sixteen_container_f64_rader_transpose_falls_back_when_shared_memory_is_too_small() {
        let mut constrained = device();
        constrained.shared_memory_bytes = 48 * 1024;
        constrained.shared_memory_pow2_bytes = 32 * 1024;
        constrained.supports_f64 = true;
        let plan = FftPlan::build(FftConfig::new(vec![4112]).with_precision(crate::Precision::F64))
            .unwrap();
        let ir = MixedRaderStockhamIr::build(&plan, Direction::Forward, constrained).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("4112 should still use FFT-convolution Rader under the fallback");
        };
        assert!(rader.internal_register_schedule.is_none());
        let crate::RecursiveFftNodeIr::Stockham(kernel) = &rader.forward_recursive().unwrap().root
        else {
            panic!("fallback p257 convolution should remain a Stockham root");
        };
        assert!(kernel.rader_transpose.is_none());
        assert!(kernel.required_shared_memory_bytes().unwrap() <= 48 * 1024);
    }

    #[test]
    fn non_power_of_two_rader_grouping_uses_smooth_outer_container_schedule() {
        let length = 51usize;
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_batch_count(2)).unwrap();
        let ir = MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("51 should use FFT-convolution Rader for its 17 factor");
        };
        let schedule = rader
            .internal_register_schedule
            .as_ref()
            .expect("3 x p17 should use the smooth three-container Rader schedule");
        assert_eq!(schedule.outer_fft_len, 51);
        assert_eq!(schedule.container_fft_num, 3);
        assert_eq!(schedule.execution_container_fft_num, 3);
        assert_eq!(schedule.internal_fft.stage_radices, vec![16]);
        assert!(schedule.rader_transpose.is_none());
        let input = sample(length, 2);
        let actual = execute_mixed_rader_stockham_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for batch in 0..2 {
            let start = batch * length;
            expected.extend(dft(
                &input[start..start + length],
                Direction::Forward,
                false,
            ));
        }
        assert!(max_error(&actual, &expected) < 1.0e-8 * length as f64);
    }

    #[test]
    fn mixed_stockham_direct_rader_matches_dft() {
        check_mixed_length(94);
    }

    fn check_mixed_length(length: usize) {
        let batch_count = 2usize;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let ir = MixedRaderStockhamIr::build(&plan, direction, device()).unwrap();
            if matches!(length, 34 | 68 | 136 | 514 | 1028) {
                let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
                    panic!("{length} should use FFT-convolution Rader");
                };
                let schedule = rader.internal_register_schedule.as_ref().expect(
                    "power-of-two mixed Rader stage should carry container occupancy metadata",
                );
                let (
                    prime,
                    container_count,
                    expected_radices,
                    upstream_threads,
                    execution_containers,
                    execution_threads,
                ) = match length {
                    34 => (17, 2, vec![16], 2, 2, 2),
                    68 => (17, 4, vec![16], 4, 4, 4),
                    136 => (17, 8, vec![16], 8, 8, 8),
                    514 => (257, 2, vec![16, 16], 32, 2, 32),
                    1028 => (257, 4, vec![16, 16], 64, 4, 64),
                    _ => unreachable!(),
                };
                assert_eq!(schedule.prime, prime);
                assert_eq!(schedule.internal_fft.stage_radices, expected_radices);
                assert_eq!(
                    schedule.internal_fft.rhs_transform_count,
                    container_count * batch_count
                );
                assert_eq!(schedule.outer_fft_len, length);
                assert_eq!(schedule.container_fft_num, container_count);
                assert_eq!(schedule.min_rader_fft_thread_num, upstream_threads);
                assert_eq!(schedule.upstream_workgroup_count(), batch_count);
                assert_eq!(schedule.execution_container_fft_num, execution_containers);
                assert_eq!(
                    schedule.execution_workgroup_count,
                    container_count * batch_count / execution_containers
                );
                assert_eq!(schedule.execution_threads_per_workgroup, execution_threads);
                assert!(schedule.upstream_grouping_is_executable());
                assert!(schedule.rader_transpose.is_none());
            }
            assert_eq!(ir.pack_prime.operation, MixedPassOperation::PackPrimeInput);
            assert_eq!(
                ir.twiddle_transpose.operation,
                MixedPassOperation::TwiddleTranspose
            );
            assert_eq!(
                ir.scatter_output.operation,
                MixedPassOperation::ScatterOutput
            );
            let input = sample(length, batch_count);
            let actual = execute_mixed_rader_stockham_ir(&ir, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for batch in 0..batch_count {
                let start = batch * length;
                expected.extend(dft(
                    &input[start..start + length],
                    direction,
                    direction == Direction::Inverse,
                ));
            }
            assert!(
                max_error(&actual, &expected) < 1.0e-8 * length as f64,
                "mixed transform mismatch for length {length} direction {direction:?}"
            );
        }
    }
}
