//! Generic one-dimensional complex FFT composition.
//!
//! Planner-selected Stockham/Rader axes use `RecursiveFftIr`; axes that require
//! Bluestein use `BluesteinPipelineIr`. Consumers such as multidimensional and
//! real transforms can therefore depend on one typed 1D abstraction without
//! re-implementing the planner's algorithm-family dispatch.

use crate::bluestein_ir::{BluesteinPipelineIr, execute_bluestein_ir};
use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, TransformKind};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::ScalarType;
use crate::planner::{AxisAlgorithm, FftPlan};
use crate::recursive_ir::{RecursiveFftIr, execute_recursive_fft_ir};
use crate::zero_pad_ir::ZeroPadPassIr;

#[derive(Debug, Clone, PartialEq)]
pub enum OneDimFftIr {
    Recursive(Box<RecursiveFftIr>),
    Bluestein(Box<BluesteinPipelineIr>),
}

impl OneDimFftIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "generic one-dimensional FFT IR requires exactly one dimension",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "generic one-dimensional FFT IR supports C2C transforms only",
            ));
        }
        match &plan.axes[0].algorithm {
            AxisAlgorithm::Bluestein { .. } => Ok(Self::Bluestein(Box::new(
                BluesteinPipelineIr::build(plan, direction, device)?,
            ))),
            AxisAlgorithm::Stockham { .. } | AxisAlgorithm::Rader { .. } => Ok(Self::Recursive(
                Box::new(RecursiveFftIr::build(plan, direction, device)?),
            )),
        }
    }

    pub(crate) fn build_for_convolution_stockham(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1
            || plan.config.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "performConvolution Stockham builder requires a one-dimensional C2C plan",
            ));
        }
        match &plan.axes[0].algorithm {
            AxisAlgorithm::Stockham { .. } => Ok(Self::Recursive(Box::new(
                RecursiveFftIr::build_for_convolution(plan, direction, device)?,
            ))),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "performConvolution Stockham builder requires a Stockham axis",
            )),
        }
    }

    pub(crate) fn build_for_convolution_stockham_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1
            || plan.config.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "internal performConvolution Stockham builder requires a one-dimensional C2C plan",
            ));
        }
        match &plan.axes[0].algorithm {
            AxisAlgorithm::Stockham { .. } => Ok(Self::Recursive(Box::new(
                RecursiveFftIr::build_for_convolution_internal_compute_storage(
                    plan, direction, device,
                )?,
            ))),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "internal performConvolution Stockham builder requires a Stockham axis",
            )),
        }
    }

    pub(crate) fn build_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "internal one-dimensional FFT IR requires exactly one dimension",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "internal one-dimensional FFT IR supports C2C transforms only",
            ));
        }
        match &plan.axes[0].algorithm {
            AxisAlgorithm::Bluestein { .. } => Ok(Self::Bluestein(Box::new(
                BluesteinPipelineIr::build_internal_compute_storage(plan, direction, device)?,
            ))),
            AxisAlgorithm::Stockham { .. } | AxisAlgorithm::Rader { .. } => {
                Ok(Self::Recursive(Box::new(
                    RecursiveFftIr::build_internal_compute_storage(plan, direction, device)?,
                )))
            }
        }
    }

    /// Apply the axis-0 one-dimensional C2C axis-block optimization. Composition
    /// layers may opt into this only when their packed child corresponds to the
    /// physical fastest axis.
    pub(crate) fn with_axis0_single_upload_block(self, device: DeviceProfile) -> Result<Self> {
        match self {
            Self::Recursive(ir) => Ok(Self::Recursive(Box::new(
                (*ir).with_axis0_single_upload_block(device)?,
            ))),
            Self::Bluestein(_) => Ok(self),
        }
    }

    /// Apply the covered higher/strided-axis block geometry to a packed ND child.
    pub(crate) fn with_other_axis_single_upload_block(
        self,
        fastest_axis_len: usize,
        device: DeviceProfile,
    ) -> Result<Self> {
        match self {
            Self::Recursive(ir) => Ok(Self::Recursive(Box::new(
                (*ir).with_other_axis_single_upload_block(fastest_axis_len, device)?,
            ))),
            Self::Bluestein(ir) => Ok(Self::Bluestein(Box::new((*ir).with_other_axis_blocks(
                fastest_axis_len,
                None,
                None,
                device,
            )?))),
        }
    }

    pub(crate) fn with_other_axis_single_upload_block_with_grouped_batch(
        self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        match self {
            Self::Recursive(ir) => Ok(Self::Recursive(Box::new(
                (*ir).with_other_axis_single_upload_block_with_grouped_batch(
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?,
            ))),
            Self::Bluestein(ir) => Ok(Self::Bluestein(Box::new((*ir).with_other_axis_blocks(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?))),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Recursive(ir) => ir.validate(),
            Self::Bluestein(ir) => ir.validate(),
        }
    }

    pub fn logical_len(&self) -> usize {
        match self {
            Self::Recursive(ir) => ir.logical_len,
            Self::Bluestein(ir) => ir.logical_len,
        }
    }

    pub fn batch_count(&self) -> usize {
        match self {
            Self::Recursive(ir) => ir.batch_count,
            Self::Bluestein(ir) => ir.batch_count,
        }
    }

    pub fn grouped_batch(&self) -> usize {
        match self {
            Self::Recursive(ir) => ir.axis0_grouped_batch_override.unwrap_or(1),
            Self::Bluestein(ir) => ir.grouped_batch,
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Recursive(ir) => ir.direction,
            Self::Bluestein(ir) => ir.direction,
        }
    }

    pub fn zero_pad_pass(&self) -> Option<&ZeroPadPassIr> {
        match self {
            Self::Recursive(ir) => ir.zero_pad_pass.as_ref(),
            Self::Bluestein(ir) => ir.zero_pad_pass.as_ref(),
        }
    }

    pub fn scalar(&self) -> ScalarType {
        match self {
            Self::Recursive(ir) => ir.scalar,
            Self::Bluestein(ir) => ir.scalar,
        }
    }

    pub fn external_storage_scalar(&self) -> ScalarType {
        match self {
            Self::Recursive(ir) => ir.external_storage_scalar(),
            Self::Bluestein(ir) => ir.external_storage_scalar(),
        }
    }
}

pub fn execute_one_dim_fft_ir(ir: &OneDimFftIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    ir.validate()?;
    match ir {
        OneDimFftIr::Recursive(ir) => execute_recursive_fft_ir(ir, input),
        OneDimFftIr::Bluestein(ir) => execute_bluestein_ir(ir, input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, FftConfig, GpuVendor, ZeroPaddingDomain};
    use crate::reference::{dft, fft};

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
                Complex64::new(
                    (0.083 * x).sin() + x * 0.0007,
                    (0.029 * x).cos() - x * 0.0004,
                )
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
    fn top_level_batched_stockham_consumes_axis0_swapped_block_geometry() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let plan = FftPlan::build(FftConfig::new(vec![64]).with_batch_count(32)).unwrap();
        let ir = OneDimFftIr::build(&plan, Direction::Forward, profile)
            .unwrap()
            .with_axis0_single_upload_block(profile)
            .unwrap();
        let OneDimFftIr::Recursive(recursive) = &ir else {
            panic!("64-point Stockham unexpectedly selected Bluestein");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("single-upload 64-point Stockham should remain one root kernel");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 16);
        assert_eq!(kernel.workgroup_grouping.threads_per_transform, 8);
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [16, 8]);
        assert_eq!(kernel.dispatch.x, 2);
        assert_eq!(kernel.dispatch.y, 1);

        let input = sample(64, 32);
        let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
        for batch in 0..32 {
            let start = batch * 64;
            let expected = fft(&input[start..start + 64], Direction::Forward, false).unwrap();
            assert!(max_error(&actual[start..start + 64], &expected) <= 1.0e-9 * 64.0);
        }
    }

    #[test]
    fn top_level_zero_padded_stockham_groups_without_axis_swap() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let config = FftConfig::new(vec![64])
            .with_batch_count(32)
            .with_zero_padding(0, 16, 32)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = OneDimFftIr::build(&plan, Direction::Forward, profile)
            .unwrap()
            .with_axis0_single_upload_block(profile)
            .unwrap();
        let OneDimFftIr::Recursive(recursive) = &ir else {
            panic!("zero-padded 64-point Stockham unexpectedly selected Bluestein");
        };
        assert!(recursive.zero_pad_pass.is_some());
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("zero-padded single-upload Stockham should remain one root kernel");
        };
        assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 16);
        assert_eq!(kernel.workgroup_grouping.threads_per_transform, 8);
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::ThreadsXTransformsY
        );
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [8, 16]);
        assert_eq!(kernel.dispatch.x, 2);

        let input = sample(64, 32);
        let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
        for batch in 0..32 {
            let start = batch * 64;
            let mut expected_input = input[start..start + 64].to_vec();
            expected_input[16..32].fill(Complex64::default());
            let expected = fft(&expected_input, Direction::Forward, false).unwrap();
            assert!(max_error(&actual[start..start + 64], &expected) <= 1.0e-9 * 64.0);
        }
    }

    #[test]
    fn zero_padding_reaches_smooth_rader_axis_block_and_suppresses_swap() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 19usize;
        let batch_count = 32usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_zero_padding(0, 5, 12)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = OneDimFftIr::build(&plan, Direction::Forward, profile)
            .unwrap()
            .with_axis0_single_upload_block(profile)
            .unwrap();
        let OneDimFftIr::Recursive(recursive) = &ir else {
            panic!("zero-padded p19 should use recursive Rader IR");
        };
        assert!(recursive.zero_pad_pass.is_some());
        let crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) = &recursive.root else {
            panic!("zero-padded p19 should keep an FFT-Rader root");
        };
        let block = rader
            .axis_batch_block
            .expect("zero-padded p19 should keep axis-level grouping");
        assert_eq!(block.threads_per_transform, 4);
        assert_eq!(block.grouped_batch, 32);
        assert!(!block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [4, 32]);
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) =
            &rader.forward_recursive().unwrap().root
        else {
            panic!("zero-padded p19 convolution should remain one Stockham root");
        };
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [4, 32]);
        assert_eq!(kernel.dispatch.x, 1);

        let input = sample(length, batch_count);
        let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
        for batch in 0..batch_count {
            let start = batch * length;
            let mut expected_input = input[start..start + length].to_vec();
            expected_input[5..12].fill(Complex64::default());
            let expected = fft(&expected_input, Direction::Forward, false).unwrap();
            assert!(max_error(&actual[start..start + length], &expected) <= 2.0e-9 * length as f64);
        }
    }

    #[test]
    fn exhaustive_small_lengths_cover_all_planner_families() {
        let mut saw_stockham = false;
        let mut saw_rader = false;
        let mut saw_bluestein = false;
        for length in 1usize..=192 {
            for direction in [Direction::Forward, Direction::Inverse] {
                let plan = FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_inverse_normalization(direction == Direction::Inverse),
                )
                .unwrap();
                match plan.axes[0].algorithm {
                    AxisAlgorithm::Stockham { .. } => saw_stockham = true,
                    AxisAlgorithm::Rader { .. } => saw_rader = true,
                    AxisAlgorithm::Bluestein { .. } => saw_bluestein = true,
                }
                let ir = OneDimFftIr::build(&plan, direction, device()).unwrap();
                let input = sample(length, 1);
                let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
                let expected = dft(&input, direction, direction == Direction::Inverse);
                assert!(
                    max_error(&actual, &expected) <= 2.0e-8 * length as f64,
                    "length {length} {direction:?} did not match the direct DFT"
                );
            }
        }
        assert!(saw_stockham && saw_rader && saw_bluestein);
    }

    #[test]
    fn exhaustive_medium_lengths_match_independent_fft_reference() {
        let mut saw_stockham = false;
        let mut saw_rader = false;
        let mut saw_bluestein = false;
        for length in 193usize..=1024 {
            for direction in [Direction::Forward, Direction::Inverse] {
                let normalize = direction == Direction::Inverse;
                let plan = FftPlan::build(
                    FftConfig::new(vec![length]).with_inverse_normalization(normalize),
                )
                .unwrap();
                match plan.axes[0].algorithm {
                    AxisAlgorithm::Stockham { .. } => saw_stockham = true,
                    AxisAlgorithm::Rader { .. } => saw_rader = true,
                    AxisAlgorithm::Bluestein { .. } => saw_bluestein = true,
                }
                let ir = OneDimFftIr::build(&plan, direction, device()).unwrap();
                let input = sample(length, 1);
                let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
                let expected = fft(&input, direction, normalize).unwrap();
                assert!(
                    max_error(&actual, &expected) <= 3.0e-8 * length as f64,
                    "medium length {length} {direction:?} did not match the independent FFT reference"
                );
            }
        }
        assert!(saw_stockham && saw_rader && saw_bluestein);
    }

    #[test]
    fn representative_large_lengths_match_independent_fft_reference() {
        for length in [
            1536usize, 3072, 3840, 4001, 4095, 4096, 4112, 6144, 8192, 16_384,
        ] {
            let input = sample(length, 1);
            for direction in [Direction::Forward, Direction::Inverse] {
                let normalize = direction == Direction::Inverse;
                let plan = FftPlan::build(
                    FftConfig::new(vec![length]).with_inverse_normalization(normalize),
                )
                .unwrap();
                let ir = OneDimFftIr::build(&plan, direction, device()).unwrap();
                let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
                let expected = fft(&input, direction, normalize).unwrap();
                assert!(
                    max_error(&actual, &expected) <= 5.0e-8 * length as f64,
                    "large length {length} {direction:?} did not match the independent FFT reference"
                );
            }
        }
    }

    #[test]
    fn spatial_zero_padding_matches_manual_zeroing_across_algorithm_families() {
        for (length, force_bluestein) in [(64usize, false), (103, true), (257, false)] {
            let left = length / 2;
            let right = length;
            let tuning = if force_bluestein {
                let mut tuning = crate::PlannerTuning::portable();
                tuning.max_rader_fft_prime = 100;
                tuning
            } else {
                crate::PlannerTuning::default()
            };
            let config = FftConfig::new(vec![length])
                .with_tuning(tuning)
                .with_zero_padding(0, left, right)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            let ir = OneDimFftIr::build(&plan, Direction::Forward, device()).unwrap();
            assert!(ir.zero_pad_pass().is_some());

            let mut input = sample(length, 1);
            for (offset, value) in input[left..right].iter_mut().enumerate() {
                *value = Complex64::new(10_000.0 + offset as f64, -20_000.0 - offset as f64);
            }
            let mut manual = input.clone();
            manual[left..right].fill(Complex64::default());
            let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
            let expected = fft(&manual, Direction::Forward, false).unwrap();
            assert!(
                max_error(&actual, &expected) <= 4.0e-8 * length as f64,
                "forward zero padding mismatch for N={length}"
            );

            let inverse_config = FftConfig::new(vec![length])
                .with_tuning(tuning)
                .with_inverse_normalization(true)
                .with_zero_padding(0, left, right)
                .unwrap();
            let inverse_plan = FftPlan::build(inverse_config).unwrap();
            let inverse = OneDimFftIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            let spectrum = fft(&sample(length, 1), Direction::Forward, false).unwrap();
            let actual = execute_one_dim_fft_ir(&inverse, &spectrum).unwrap();
            let mut expected = fft(&spectrum, Direction::Inverse, true).unwrap();
            expected[left..right].fill(Complex64::default());
            assert!(
                max_error(&actual, &expected) <= 4.0e-8 * length as f64,
                "inverse zero padding mismatch for N={length}"
            );
        }
    }

    #[test]
    fn frequency_zero_padding_matches_manual_frequency_boundary_across_algorithm_families() {
        for (length, force_bluestein) in [(64usize, false), (103, true), (257, false)] {
            let left = length / 3;
            let right = 2 * length / 3;
            let tuning = if force_bluestein {
                let mut tuning = crate::PlannerTuning::portable();
                tuning.max_rader_fft_prime = 100;
                tuning
            } else {
                crate::PlannerTuning::default()
            };

            let forward_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_zero_padding(0, left, right)
                    .unwrap()
                    .with_zero_padding_domain(ZeroPaddingDomain::Frequency),
            )
            .unwrap();
            let forward = OneDimFftIr::build(&forward_plan, Direction::Forward, device()).unwrap();
            assert!(
                forward
                    .zero_pad_pass()
                    .is_some_and(|pass| pass.operation.is_output_boundary())
            );
            let input = sample(length, 1);
            let actual = execute_one_dim_fft_ir(&forward, &input).unwrap();
            let mut expected = fft(&input, Direction::Forward, false).unwrap();
            expected[left..right].fill(Complex64::default());
            assert!(
                max_error(&actual, &expected) <= 4.0e-8 * length as f64,
                "forward frequency zero padding mismatch for N={length}"
            );

            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_tuning(tuning)
                    .with_inverse_normalization(true)
                    .with_zero_padding(0, left, right)
                    .unwrap()
                    .with_zero_padding_domain(ZeroPaddingDomain::Frequency),
            )
            .unwrap();
            let inverse = OneDimFftIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            assert!(
                inverse
                    .zero_pad_pass()
                    .is_some_and(|pass| pass.operation.is_input_boundary())
            );
            let spectrum = fft(&sample(length, 1), Direction::Forward, false).unwrap();
            let actual = execute_one_dim_fft_ir(&inverse, &spectrum).unwrap();
            let mut manual_spectrum = spectrum.clone();
            manual_spectrum[left..right].fill(Complex64::default());
            let expected = fft(&manual_spectrum, Direction::Inverse, true).unwrap();
            assert!(
                max_error(&actual, &expected) <= 4.0e-8 * length as f64,
                "inverse frequency zero padding mismatch for N={length}"
            );
        }
    }

    #[test]
    fn dispatches_recursive_and_bluestein_families() {
        for (length, recursive_rader, force_bluestein) in [
            (60usize, false, false),
            (17, false, false),
            (107, true, false),
            (103, false, true),
        ] {
            let batch_count = 2usize;
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut tuning =
                    crate::PlannerTuning::portable().with_recursive_fft_rader(recursive_rader);
                if force_bluestein {
                    tuning.max_rader_fft_prime = 100;
                }
                let plan = FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_tuning(tuning),
                )
                .unwrap();
                let ir = OneDimFftIr::build(&plan, direction, device()).unwrap();
                if force_bluestein {
                    assert!(matches!(ir, OneDimFftIr::Bluestein(_)));
                } else {
                    assert!(matches!(ir, OneDimFftIr::Recursive(_)));
                }
                let input = sample(length, batch_count);
                let actual = execute_one_dim_fft_ir(&ir, &input).unwrap();
                let mut expected = Vec::with_capacity(actual.len());
                for batch in 0..batch_count {
                    let base = batch * length;
                    expected.extend(
                        fft(
                            &input[base..base + length],
                            direction,
                            direction == Direction::Inverse,
                        )
                        .unwrap(),
                    );
                }
                assert!(max_error(&actual, &expected) < 5.0e-8 * length as f64);
            }
        }
    }
}
