//! Backend-independent lookup-table generation.
//!
//! VkFFT can either evaluate twiddle factors on the fly or upload tables for
//! Stockham stages, Rader permutations, and Bluestein chirps. These Rust tables
//! are deliberately backend-neutral: they establish exact numerical/indexing
//! semantics before Vulkan-specific packing or compression is introduced.

use core::f64::consts::{PI, TAU};

use crate::complex::Complex64;
use crate::config::Direction;
use crate::double_double::{ComplexDoubleDouble, unit_root as double_double_unit_root};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{KernelIr, KernelOperation};
use crate::planner::RaderPrimePlan;

/// Direction-independent unit roots used by the NVIDIA/Vulkan F64 Stockham LUT
/// path. Shaders conjugate the imaginary component with `VKFFT_SIGN`, so one
/// immutable allocation is reusable by forward and inverse transforms.
pub fn stockham_root_table(sequence_len: usize) -> Result<Vec<Complex64>> {
    if sequence_len == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Stockham root LUT requires a non-zero sequence length",
        ));
    }
    Ok((0..sequence_len)
        .map(|index| Complex64::exp_i(TAU * index as f64 / sequence_len as f64))
        .collect())
}

/// Direction-independent full-period Stockham roots in double-double precision.
/// This matches [`stockham_root_table`]'s positive-exponent indexing so the same
/// immutable table can later be conjugated by a direction sign in GPU code.
pub fn stockham_root_table_double_double(sequence_len: usize) -> Result<Vec<ComplexDoubleDouble>> {
    if sequence_len == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "double-double Stockham root LUT requires a non-zero sequence length",
        ));
    }
    unit_root_table_double_double(sequence_len, Direction::Inverse)
}

/// Full-period unit roots in double-double precision, generated with bounded-error
/// recurrence instead of evaluating the high-precision trigonometric series for every
/// element. Each block is re-anchored with [`double_double_unit_root`], while the values
/// inside the block advance by one true-DD complex multiplication. This keeps recurrence
/// drift bounded independently of the table length and makes large Four-step parent LUTs
/// practical to construct.
pub fn unit_root_table_double_double(
    sequence_len: usize,
    direction: Direction,
) -> Result<Vec<ComplexDoubleDouble>> {
    if sequence_len == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "double-double unit-root LUT requires a non-zero sequence length",
        ));
    }
    const REANCHOR_INTERVAL: usize = 1024;
    let step = double_double_unit_root(1 % sequence_len, sequence_len, direction)?;
    let mut values = Vec::with_capacity(sequence_len);
    let mut block_start = 0usize;
    while block_start < sequence_len {
        let block_end = block_start
            .saturating_add(REANCHOR_INTERVAL)
            .min(sequence_len);
        let mut value = double_double_unit_root(block_start, sequence_len, direction)?;
        values.push(value);
        for _ in block_start + 1..block_end {
            value *= step;
            values.push(value);
        }
        block_start = block_end;
    }
    Ok(values)
}

#[derive(Debug, Clone, PartialEq)]
pub struct StockhamTwiddleStage {
    pub index: usize,
    pub radix: usize,
    pub stage_size: usize,
    /// Row-major `[stage_invocation][lane]` twiddles.
    pub values: Vec<Complex64>,
}

impl StockhamTwiddleStage {
    pub fn get(&self, stage_invocation: usize, lane: usize) -> Option<Complex64> {
        if stage_invocation >= self.stage_size || lane >= self.radix {
            return None;
        }
        self.values
            .get(
                stage_invocation
                    .checked_mul(self.radix)?
                    .checked_add(lane)?,
            )
            .copied()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StockhamTwiddleTable {
    pub sequence_len: usize,
    pub direction: Direction,
    pub stages: Vec<StockhamTwiddleStage>,
}

impl StockhamTwiddleTable {
    /// Materialize the exact inter-stage twiddles represented by a validated
    /// Stockham [`KernelIr`]. The initial layout stores one full `stage_size *
    /// radix` rectangle per stage; backend-specific compression can be added later.
    pub fn from_kernel(kernel: &KernelIr) -> Result<Self> {
        kernel.validate()?;
        let sign = kernel.direction.exponent_sign();
        let mut stages = Vec::new();
        for operation in &kernel.operations {
            let KernelOperation::StockhamStage(stage) = operation else {
                continue;
            };
            let denominator = stage.stage_size.checked_mul(stage.radix).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "Stockham LUT stage denominator",
                },
            )?;
            let count = denominator;
            let mut values = Vec::with_capacity(count);
            for stage_invocation in 0..stage.stage_size {
                for lane in 0..stage.radix {
                    let angle = sign * TAU * (stage_invocation * lane) as f64 / denominator as f64;
                    values.push(Complex64::exp_i(angle));
                }
            }
            stages.push(StockhamTwiddleStage {
                index: stage.index,
                radix: stage.radix,
                stage_size: stage.stage_size,
                values,
            });
        }
        Ok(Self {
            sequence_len: kernel.sequence_len,
            direction: kernel.direction,
            stages,
        })
    }
    pub fn packed_values(&self) -> Vec<Complex64> {
        let total = self.stages.iter().map(|stage| stage.values.len()).sum();
        let mut values = Vec::with_capacity(total);
        for stage in &self.stages {
            values.extend_from_slice(&stage.values);
        }
        values
    }

    pub fn stage_offset(&self, stage_index: usize) -> Option<usize> {
        let mut offset = 0usize;
        for stage in &self.stages {
            if stage.index == stage_index {
                return Some(offset);
            }
            offset = offset.checked_add(stage.values.len())?;
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleStockhamTwiddleStage {
    pub index: usize,
    pub radix: usize,
    pub stage_size: usize,
    pub values: Vec<ComplexDoubleDouble>,
}

impl DoubleDoubleStockhamTwiddleStage {
    pub fn get(&self, stage_invocation: usize, lane: usize) -> Option<ComplexDoubleDouble> {
        if stage_invocation >= self.stage_size || lane >= self.radix {
            return None;
        }
        self.values
            .get(
                stage_invocation
                    .checked_mul(self.radix)?
                    .checked_add(lane)?,
            )
            .copied()
    }
}

/// Stockham stage twiddles generated from double-double unit roots with exactly the
/// same row-major stage/lane indexing as [`StockhamTwiddleTable`].
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleStockhamTwiddleTable {
    pub sequence_len: usize,
    pub direction: Direction,
    pub stages: Vec<DoubleDoubleStockhamTwiddleStage>,
}

impl DoubleDoubleStockhamTwiddleTable {
    pub fn from_kernel(kernel: &KernelIr) -> Result<Self> {
        kernel.validate()?;
        let mut stages = Vec::new();
        for operation in &kernel.operations {
            let KernelOperation::StockhamStage(stage) = operation else {
                continue;
            };
            let denominator = stage.stage_size.checked_mul(stage.radix).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham LUT stage denominator",
                },
            )?;
            let mut values = Vec::with_capacity(denominator);
            for stage_invocation in 0..stage.stage_size {
                for lane in 0..stage.radix {
                    let root_index = stage_invocation.checked_mul(lane).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "double-double Stockham LUT root index",
                        },
                    )?;
                    values.push(double_double_unit_root(
                        root_index,
                        denominator,
                        kernel.direction,
                    )?);
                }
            }
            stages.push(DoubleDoubleStockhamTwiddleStage {
                index: stage.index,
                radix: stage.radix,
                stage_size: stage.stage_size,
                values,
            });
        }
        Ok(Self {
            sequence_len: kernel.sequence_len,
            direction: kernel.direction,
            stages,
        })
    }

    pub fn packed_values(&self) -> Vec<ComplexDoubleDouble> {
        let total = self.stages.iter().map(|stage| stage.values.len()).sum();
        let mut values = Vec::with_capacity(total);
        for stage in &self.stages {
            values.extend_from_slice(&stage.values);
        }
        values
    }

    pub fn stage_offset(&self, stage_index: usize) -> Option<usize> {
        let mut offset = 0usize;
        for stage in &self.stages {
            if stage.index == stage_index {
                return Some(offset);
            }
            offset = offset.checked_add(stage.values.len())?;
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BluesteinTable {
    pub length: usize,
    pub convolution_len: usize,
    pub direction: Direction,
    /// Pre/post chirp `exp(sign * pi * n^2 / N)` for `n in 0..N`.
    pub chirp: Vec<Complex64>,
    /// Symmetric convolution kernel in the padded circular-convolution layout.
    pub convolution_kernel: Vec<Complex64>,
}

impl BluesteinTable {
    pub fn new(length: usize, convolution_len: usize, direction: Direction) -> Result<Self> {
        if length == 0 {
            return Err(VkFftError::InvalidLut(
                "Bluestein transform length must be non-zero",
            ));
        }
        let minimum = length
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Bluestein LUT minimum convolution length",
            })?;
        if convolution_len < minimum {
            return Err(VkFftError::InvalidLut(
                "Bluestein convolution length must be at least 2N-1",
            ));
        }

        let sign = direction.exponent_sign();
        let mut chirp = Vec::with_capacity(length);
        let mut convolution_kernel = vec![Complex64::new(0.0, 0.0); convolution_len];
        for index in 0..length {
            let index_f64 = index as f64;
            let angle = sign * PI * index_f64 * index_f64 / length as f64;
            let value = Complex64::exp_i(angle);
            let inverse = Complex64::exp_i(-angle);
            chirp.push(value);
            convolution_kernel[index] = inverse;
            if index != 0 {
                convolution_kernel[convolution_len - index] = inverse;
            }
        }
        Ok(Self {
            length,
            convolution_len,
            direction,
            chirp,
            convolution_kernel,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleBluesteinTable {
    pub length: usize,
    pub convolution_len: usize,
    pub direction: Direction,
    pub chirp: Vec<ComplexDoubleDouble>,
    pub convolution_kernel: Vec<ComplexDoubleDouble>,
}

impl DoubleDoubleBluesteinTable {
    pub fn new(length: usize, convolution_len: usize, direction: Direction) -> Result<Self> {
        if length == 0 {
            return Err(VkFftError::InvalidLut(
                "double-double Bluestein transform length must be non-zero",
            ));
        }
        let period = length
            .checked_mul(2)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Bluestein chirp period",
            })?;
        let minimum = period
            .checked_sub(1)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Bluestein minimum convolution length",
            })?;
        if convolution_len < minimum {
            return Err(VkFftError::InvalidLut(
                "double-double Bluestein convolution length must be at least 2N-1",
            ));
        }

        let mut chirp = Vec::with_capacity(length);
        let mut convolution_kernel = vec![ComplexDoubleDouble::default(); convolution_len];
        for index in 0..length {
            let squared = index
                .checked_mul(index)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Bluestein chirp index square",
                })?;
            let value = double_double_unit_root(squared % period, period, direction)?;
            let inverse = value.conj();
            chirp.push(value);
            convolution_kernel[index] = inverse;
            if index != 0 {
                convolution_kernel[convolution_len - index] = inverse;
            }
        }
        Ok(Self {
            length,
            convolution_len,
            direction,
            chirp,
            convolution_kernel,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaderTable {
    pub prime: usize,
    pub generator: usize,
    pub direction: Direction,
    /// `generator^i mod prime`, `i in 0..prime-1`.
    pub permutation: Vec<usize>,
    /// Inverse map indexed by `residue - 1`, returning the generator exponent.
    pub permutation_inverse: Vec<usize>,
    /// DFT roots ordered by the same generator-power permutation.
    pub twiddles_by_generator_power: Vec<Complex64>,
}

impl RaderTable {
    pub fn from_prime_plan(plan: &RaderPrimePlan, direction: Direction) -> Result<Self> {
        if plan.prime < 2 || plan.generator == 0 || plan.generator >= plan.prime {
            return Err(VkFftError::InvalidLut(
                "Rader prime/generator metadata is invalid",
            ));
        }
        let count = plan.prime - 1;
        let mut permutation = Vec::with_capacity(count);
        let mut permutation_inverse = vec![usize::MAX; count];
        let mut twiddles_by_generator_power = Vec::with_capacity(count);
        let sign = direction.exponent_sign();
        let mut residue = 1usize;
        for exponent in 0..count {
            if residue == 0
                || residue >= plan.prime
                || permutation_inverse[residue - 1] != usize::MAX
            {
                return Err(VkFftError::InvalidLut(
                    "Rader generator does not span the multiplicative group",
                ));
            }
            permutation.push(residue);
            permutation_inverse[residue - 1] = exponent;
            let angle = sign * TAU * residue as f64 / plan.prime as f64;
            twiddles_by_generator_power.push(Complex64::exp_i(angle));
            residue =
                residue
                    .checked_mul(plan.generator)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Rader generator permutation",
                    })?
                    % plan.prime;
        }
        if residue != 1 || permutation_inverse.contains(&usize::MAX) {
            return Err(VkFftError::InvalidLut(
                "Rader generator does not close after prime-1 powers",
            ));
        }
        Ok(Self {
            prime: plan.prime,
            generator: plan.generator,
            direction,
            permutation,
            permutation_inverse,
            twiddles_by_generator_power,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleRaderTable {
    pub prime: usize,
    pub generator: usize,
    pub direction: Direction,
    pub permutation: Vec<usize>,
    pub permutation_inverse: Vec<usize>,
    pub twiddles_by_generator_power: Vec<ComplexDoubleDouble>,
}

impl DoubleDoubleRaderTable {
    pub fn from_prime_plan(plan: &RaderPrimePlan, direction: Direction) -> Result<Self> {
        if plan.prime < 2 || plan.generator == 0 || plan.generator >= plan.prime {
            return Err(VkFftError::InvalidLut(
                "double-double Rader prime/generator metadata is invalid",
            ));
        }
        let count = plan.prime - 1;
        let mut permutation = Vec::with_capacity(count);
        let mut permutation_inverse = vec![usize::MAX; count];
        let mut twiddles_by_generator_power = Vec::with_capacity(count);
        let mut residue = 1usize;
        for exponent in 0..count {
            if residue == 0
                || residue >= plan.prime
                || permutation_inverse[residue - 1] != usize::MAX
            {
                return Err(VkFftError::InvalidLut(
                    "double-double Rader generator does not span the multiplicative group",
                ));
            }
            permutation.push(residue);
            permutation_inverse[residue - 1] = exponent;
            twiddles_by_generator_power
                .push(double_double_unit_root(residue, plan.prime, direction)?);
            residue =
                residue
                    .checked_mul(plan.generator)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double Rader generator permutation",
                    })?
                    % plan.prime;
        }
        if residue != 1 || permutation_inverse.contains(&usize::MAX) {
            return Err(VkFftError::InvalidLut(
                "double-double Rader generator does not close after prime-1 powers",
            ));
        }
        Ok(Self {
            prime: plan.prime,
            generator: plan.generator,
            direction,
            permutation,
            permutation_inverse,
            twiddles_by_generator_power,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, DeviceProfile, GpuVendor, PlannerTuning};
    use crate::kernel_ir::KernelIr;
    use crate::planner::{AxisAlgorithm, plan_axis_algorithm};
    use crate::reference::{dft, fft};
    use crate::{FftConfig, FftPlan};

    fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    fn dd_root_error(lhs: ComplexDoubleDouble, rhs: ComplexDoubleDouble) -> f64 {
        let re = (lhs.re - rhs.re).abs();
        let im = (lhs.im - rhs.im).abs();
        re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
    }

    #[test]
    fn double_double_unit_root_table_reanchors_without_losing_dd_accuracy() {
        for length in [1usize, 17, 256, 1_024, 4_095, 32_768] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let table = unit_root_table_double_double(length, direction).unwrap();
                assert_eq!(table.len(), length);
                let mut samples = vec![
                    0,
                    1.min(length - 1),
                    length / 4,
                    length / 2,
                    length.saturating_sub(1),
                    1_023.min(length - 1),
                    1_024.min(length - 1),
                    1_025.min(length - 1),
                ];
                for index in (0..length).step_by((length / 19).max(1)) {
                    samples.push(index);
                }
                samples.sort_unstable();
                samples.dedup();
                for index in samples {
                    let expected = double_double_unit_root(index, length, direction).unwrap();
                    let error = dd_root_error(table[index], expected);
                    assert!(
                        error <= 1.0e-26,
                        "DD root recurrence mismatch at N={length}, index={index}, direction={direction:?}: {error:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn double_double_unit_root_table_scales_to_large_four_step_period() {
        let length = 524_288usize;
        let table = unit_root_table_double_double(length, Direction::Forward).unwrap();
        assert_eq!(table.len(), length);
        for index in [
            0usize,
            1,
            1_023,
            1_024,
            1_025,
            length / 7,
            length / 3,
            length / 2,
            length - 1,
        ] {
            let expected = double_double_unit_root(index, length, Direction::Forward).unwrap();
            let error = dd_root_error(table[index], expected);
            assert!(
                error <= 1.0e-26,
                "large DD root recurrence mismatch at index={index}: {error:e}"
            );
        }
    }

    #[test]
    fn stockham_table_matches_ir_stage_formula() {
        let plan = FftPlan::build(FftConfig::new(vec![60])).unwrap();
        let mut device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        device.shared_memory_bytes = 128 * 1024;
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device).unwrap();
        let table = StockhamTwiddleTable::from_kernel(&kernel).unwrap();
        assert_eq!(table.sequence_len, 60);
        for stage in &table.stages {
            assert_eq!(stage.values.len(), stage.stage_size * stage.radix);
            for stage_invocation in 0..stage.stage_size {
                assert_eq!(
                    stage.get(stage_invocation, 0).unwrap(),
                    Complex64::new(1.0, 0.0)
                );
                for lane in 0..stage.radix {
                    let denominator = stage.stage_size * stage.radix;
                    let angle = -TAU * (stage_invocation * lane) as f64 / denominator as f64;
                    let expected = Complex64::exp_i(angle);
                    assert!(
                        (stage.get(stage_invocation, lane).unwrap() - expected).norm_sqr()
                            < 1.0e-28
                    );
                }
            }
        }
    }

    #[test]
    fn double_double_root_table_matches_f64_indexing_and_retains_residuals() {
        let roots64 = stockham_root_table(17).unwrap();
        let roots_dd = stockham_root_table_double_double(17).unwrap();
        assert_eq!(roots_dd.len(), roots64.len());
        let mut residual_count = 0usize;
        for (dd, f64_value) in roots_dd.iter().zip(&roots64) {
            let projected = dd.to_complex64();
            assert!((projected - *f64_value).norm_sqr() < 1.0e-28);
            if dd.re.lo != 0.0 || dd.im.lo != 0.0 {
                residual_count += 1;
            }
        }
        assert!(residual_count > 0);
        assert_eq!(roots_dd[0].re, crate::DoubleDouble::ONE);
        assert_eq!(roots_dd[0].im, crate::DoubleDouble::ZERO);
    }

    #[test]
    fn double_double_stockham_table_preserves_stage_indexing() {
        let plan = FftPlan::build(FftConfig::new(vec![60])).unwrap();
        let mut device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        device.shared_memory_bytes = 128 * 1024;
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device).unwrap();
        let f64_table = StockhamTwiddleTable::from_kernel(&kernel).unwrap();
        let dd_table = DoubleDoubleStockhamTwiddleTable::from_kernel(&kernel).unwrap();
        assert_eq!(dd_table.sequence_len, f64_table.sequence_len);
        assert_eq!(dd_table.stages.len(), f64_table.stages.len());
        assert_eq!(
            dd_table.packed_values().len(),
            f64_table.packed_values().len()
        );
        for (dd_stage, f64_stage) in dd_table.stages.iter().zip(&f64_table.stages) {
            assert_eq!(dd_stage.index, f64_stage.index);
            assert_eq!(dd_stage.radix, f64_stage.radix);
            assert_eq!(dd_stage.stage_size, f64_stage.stage_size);
            assert_eq!(
                dd_table.stage_offset(dd_stage.index),
                f64_table.stage_offset(f64_stage.index)
            );
            for invocation in 0..dd_stage.stage_size {
                for lane in 0..dd_stage.radix {
                    let dd = dd_stage.get(invocation, lane).unwrap().to_complex64();
                    let ordinary = f64_stage.get(invocation, lane).unwrap();
                    assert!((dd - ordinary).norm_sqr() < 1.0e-28);
                }
            }
        }
    }

    #[test]
    fn bluestein_table_reconstructs_dft() {
        let length = 17usize;
        let convolution_len = 36usize;
        let table = BluesteinTable::new(length, convolution_len, Direction::Forward).unwrap();
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.31 * x).sin() + x * 0.01, (0.13 * x).cos() - x * 0.02)
            })
            .collect::<Vec<_>>();
        let mut a = vec![Complex64::new(0.0, 0.0); convolution_len];
        for index in 0..length {
            a[index] = input[index] * table.chirp[index];
        }
        let a_frequency = fft(&a, Direction::Forward, false).unwrap();
        let b_frequency = fft(&table.convolution_kernel, Direction::Forward, false).unwrap();
        let product = a_frequency
            .into_iter()
            .zip(b_frequency)
            .map(|(lhs, rhs)| lhs * rhs)
            .collect::<Vec<_>>();
        let convolution = fft(&product, Direction::Inverse, true).unwrap();
        let actual = convolution
            .into_iter()
            .take(length)
            .enumerate()
            .map(|(index, value)| value * table.chirp[index])
            .collect::<Vec<_>>();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 1.0e-9 * length as f64);
    }

    #[test]
    fn double_double_bluestein_table_matches_f64_layout_and_keeps_residuals() {
        let ordinary = BluesteinTable::new(17, 36, Direction::Forward).unwrap();
        let precise = DoubleDoubleBluesteinTable::new(17, 36, Direction::Forward).unwrap();
        assert_eq!(precise.length, ordinary.length);
        assert_eq!(precise.convolution_len, ordinary.convolution_len);
        let mut residuals = 0usize;
        for (dd, f64_value) in precise.chirp.iter().zip(&ordinary.chirp) {
            assert!((dd.to_complex64() - *f64_value).norm_sqr() < 1.0e-28);
            residuals += usize::from(dd.re.lo != 0.0 || dd.im.lo != 0.0);
        }
        assert!(residuals > 0);
        for index in 1..17 {
            assert_eq!(
                precise.convolution_kernel[index],
                precise.convolution_kernel[36 - index]
            );
            assert_eq!(
                precise.convolution_kernel[index],
                precise.chirp[index].conj()
            );
        }
    }

    #[test]
    fn double_double_rader_table_reuses_generator_permutation_exactly() {
        let algorithm = plan_axis_algorithm(17, PlannerTuning::portable()).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = algorithm else {
            panic!("17 should use Rader with portable tuning");
        };
        let ordinary = RaderTable::from_prime_plan(&primes[0], Direction::Forward).unwrap();
        let precise =
            DoubleDoubleRaderTable::from_prime_plan(&primes[0], Direction::Forward).unwrap();
        assert_eq!(precise.permutation, ordinary.permutation);
        assert_eq!(precise.permutation_inverse, ordinary.permutation_inverse);
        let mut residuals = 0usize;
        for (dd, f64_value) in precise
            .twiddles_by_generator_power
            .iter()
            .zip(&ordinary.twiddles_by_generator_power)
        {
            assert!((dd.to_complex64() - *f64_value).norm_sqr() < 1.0e-28);
            residuals += usize::from(dd.re.lo != 0.0 || dd.im.lo != 0.0);
        }
        assert!(residuals > 0);
    }

    #[test]
    fn rader_table_covers_every_nonzero_residue() {
        let algorithm = plan_axis_algorithm(17, PlannerTuning::portable()).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = algorithm else {
            panic!("17 should use Rader with portable tuning");
        };
        let table = RaderTable::from_prime_plan(&primes[0], Direction::Forward).unwrap();
        let mut sorted = table.permutation.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (1usize..17).collect::<Vec<_>>());
        for (exponent, &residue) in table.permutation.iter().enumerate() {
            assert_eq!(table.permutation_inverse[residue - 1], exponent);
            let expected = Complex64::exp_i(-TAU * residue as f64 / 17.0);
            assert!((table.twiddles_by_generator_power[exponent] - expected).norm_sqr() < 1.0e-28);
        }
    }
}
