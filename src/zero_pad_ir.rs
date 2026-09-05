//! Typed upstream-style zero-padding boundary passes.
//!
//! VkFFT's `performZeropadding` keeps the physical FFT extent unchanged and treats
//! `[left, right)` as a logical zero interval. Spatial padding owns the forward input
//! / inverse output boundary; `frequencyZeroPadding` reverses that ownership to the
//! forward output / inverse input boundary.

use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, Precision, ZeroPaddingDomain, ZeroPaddingRange};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{DispatchGeometry, ScalarType, WorkgroupSize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPadPassOperation {
    PrepareForwardInput,
    FinalizeInverseOutput,
    FinalizeForwardOutput,
    PrepareInverseInput,
}

impl ZeroPadPassOperation {
    pub const fn is_input_boundary(self) -> bool {
        matches!(self, Self::PrepareForwardInput | Self::PrepareInverseInput)
    }

    pub const fn is_output_boundary(self) -> bool {
        !self.is_input_boundary()
    }

    const fn name_suffix(self) -> &'static str {
        match self {
            Self::PrepareForwardInput => "forward_input",
            Self::FinalizeInverseOutput => "inverse_output",
            Self::FinalizeForwardOutput => "forward_output",
            Self::PrepareInverseInput => "inverse_input",
        }
    }

    const fn for_boundary(direction: Direction, domain: ZeroPaddingDomain) -> Self {
        match (direction, domain) {
            (Direction::Forward, ZeroPaddingDomain::Spatial) => Self::PrepareForwardInput,
            (Direction::Inverse, ZeroPaddingDomain::Spatial) => Self::FinalizeInverseOutput,
            (Direction::Forward, ZeroPaddingDomain::Frequency) => Self::FinalizeForwardOutput,
            (Direction::Inverse, ZeroPaddingDomain::Frequency) => Self::PrepareInverseInput,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroPadPassIr {
    pub name: String,
    /// Arithmetic type used on the compute side of the boundary.
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub direction: Direction,
    pub logical_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub range: ZeroPaddingRange,
    pub operation: ZeroPadPassOperation,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
}

impl ZeroPadPassIr {
    pub fn build(
        logical_len: usize,
        batch_count: usize,
        precision: Precision,
        direction: Direction,
        range: ZeroPaddingRange,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_with_domain(
            logical_len,
            batch_count,
            precision,
            direction,
            range,
            ZeroPaddingDomain::Spatial,
            device,
        )
    }

    pub fn build_with_domain(
        logical_len: usize,
        batch_count: usize,
        precision: Precision,
        direction: Direction,
        range: ZeroPaddingRange,
        domain: ZeroPaddingDomain,
        device: DeviceProfile,
    ) -> Result<Self> {
        if logical_len == 0
            || batch_count == 0
            || range.left > range.right
            || range.right > logical_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "zero-pad boundary pass has invalid dimensions or interval",
            ));
        }
        let workgroup_x = logical_len
            .min(device.max_threads_per_block.max(1))
            .clamp(1, 256);
        let operation = ZeroPadPassOperation::for_boundary(direction, domain);
        let (scalar, external_storage_scalar) = match precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => (ScalarType::F64, ScalarType::F64),
            Precision::F64 => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "zero-padding boundary",
                    precision: "f64",
                });
            }
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32)
            }
            Precision::F64ComputeF32Storage => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "zero-padding boundary",
                    precision: "f64-compute/f32-storage",
                });
            }
            Precision::DoubleDouble => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "zero-padding boundary",
                    precision: "double-double",
                });
            }
            Precision::DoubleDoubleF64Storage => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "zero-padding boundary",
                    precision: "double-double/f64-storage",
                });
            }
        };
        let (input_storage_scalar, output_storage_scalar) = if operation.is_input_boundary() {
            (external_storage_scalar, scalar)
        } else {
            (scalar, external_storage_scalar)
        };
        let pass = Self {
            name: format!(
                "vkfft_zero_pad_{}_{}_{}_{}",
                operation.name_suffix(),
                logical_len,
                range.left,
                range.right
            ),
            scalar,
            input_storage_scalar,
            output_storage_scalar,
            direction,
            logical_len,
            batch_count,
            grouped_batch: 1,
            range,
            operation,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(workgroup_x).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "zero-pad workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "zero-pad batch count",
                })?,
                y: 1,
                z: 1,
            },
        };
        pass.validate()?;
        Ok(pass)
    }

    pub(crate) fn build_storage_preserving_with_domain(
        logical_len: usize,
        batch_count: usize,
        storage_scalar: ScalarType,
        direction: Direction,
        range: ZeroPaddingRange,
        grouped_batch: usize,
        domain: ZeroPaddingDomain,
    ) -> Result<Self> {
        if logical_len == 0
            || batch_count == 0
            || grouped_batch == 0
            || range.left > range.right
            || range.right > logical_len
            || !matches!(storage_scalar, ScalarType::F64 | ScalarType::DoubleDouble)
        {
            return Err(VkFftError::InvalidKernelIr(
                "storage-preserving zero-pad boundary metadata is inconsistent",
            ));
        }
        let operation = ZeroPadPassOperation::for_boundary(direction, domain);
        let workgroup_x = logical_len.clamp(1, 64);
        let pass = Self {
            name: format!(
                "vkfft_zero_pad_storage_{}_{}_{}_{}",
                operation.name_suffix(),
                logical_len,
                range.left,
                range.right
            ),
            scalar: storage_scalar,
            input_storage_scalar: storage_scalar,
            output_storage_scalar: storage_scalar,
            direction,
            logical_len,
            batch_count,
            grouped_batch,
            range,
            operation,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(workgroup_x).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "storage-preserving zero-pad workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(batch_count.div_ceil(grouped_batch)).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "storage-preserving zero-pad grouped dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            },
        };
        pass.validate()?;
        Ok(pass)
    }

    pub(crate) fn with_grouped_batch(mut self, grouped_batch: usize) -> Result<Self> {
        if grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "zero-pad groupedBatch must be non-zero",
            ));
        }
        self.grouped_batch = grouped_batch;
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "zero-pad grouped dispatch count",
                }
            })?;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0
            || self.batch_count == 0
            || self.range.left > self.range.right
            || self.range.right > self.logical_len
            || self.workgroup_size.x == 0
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.grouped_batch == 0
            || self.dispatch.x as usize != self.batch_count.div_ceil(self.grouped_batch)
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "invalid zero-padding boundary pass",
            ));
        }
        if self.scalar == ScalarType::F16 {
            return Err(VkFftError::InvalidKernelIr(
                "binary16 cannot be used as zero-padding compute scalar",
            ));
        }
        let supported_storage = |storage: ScalarType| {
            storage == self.scalar
                || matches!(
                    (self.scalar, storage),
                    (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
                )
        };
        if !supported_storage(self.input_storage_scalar)
            || !supported_storage(self.output_storage_scalar)
            || if self.operation.is_input_boundary() {
                self.output_storage_scalar != self.scalar
            } else {
                self.input_storage_scalar != self.scalar
            }
        {
            return Err(VkFftError::InvalidKernelIr(
                "zero-padding storage boundary is inconsistent with compute precision/direction",
            ));
        }
        if !matches!(
            (self.direction, self.operation),
            (
                Direction::Forward,
                ZeroPadPassOperation::PrepareForwardInput
            ) | (
                Direction::Forward,
                ZeroPadPassOperation::FinalizeForwardOutput
            ) | (
                Direction::Inverse,
                ZeroPadPassOperation::FinalizeInverseOutput
            ) | (
                Direction::Inverse,
                ZeroPadPassOperation::PrepareInverseInput
            )
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "zero-padding operation does not match transform direction",
            ));
        }
        Ok(())
    }
}

pub fn execute_zero_pad_pass(pass: &ZeroPadPassIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    pass.validate()?;
    let expected =
        pass.logical_len
            .checked_mul(pass.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "zero-pad boundary element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut output = input.to_vec();
    for batch in 0..pass.batch_count {
        let base = batch * pass.logical_len;
        output[base + pass.range.left..base + pass.range.right].fill(Complex64::default());
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdZeroPadPassIr {
    pub name: String,
    /// Arithmetic type used on the compute side of the boundary.
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub direction: Direction,
    pub dimensions: Vec<usize>,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub ranges: Vec<Option<ZeroPaddingRange>>,
    pub operation: ZeroPadPassOperation,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
}

impl NdZeroPadPassIr {
    pub fn build(
        dimensions: &[usize],
        batch_count: usize,
        precision: Precision,
        direction: Direction,
        ranges: &[Option<ZeroPaddingRange>],
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_with_domain(
            dimensions,
            batch_count,
            precision,
            direction,
            ranges,
            ZeroPaddingDomain::Spatial,
            device,
        )
    }

    pub fn build_with_domain(
        dimensions: &[usize],
        batch_count: usize,
        precision: Precision,
        direction: Direction,
        ranges: &[Option<ZeroPaddingRange>],
        domain: ZeroPaddingDomain,
        device: DeviceProfile,
    ) -> Result<Self> {
        if dimensions.is_empty() || dimensions.len() != ranges.len() || batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional zero-pad metadata does not match tensor dimensions",
            ));
        }
        let tensor_len = dimensions.iter().try_fold(1usize, |product, length| {
            if *length == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional zero-pad dimensions must be non-zero",
                ));
            }
            product
                .checked_mul(*length)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional zero-pad tensor length",
                })
        })?;
        for (axis, (range, length)) in ranges.iter().zip(dimensions).enumerate() {
            if let Some(range) = range
                && (range.left > range.right || range.right > *length)
            {
                return Err(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left: range.left,
                    right: range.right,
                    length: *length,
                });
            }
        }
        let (scalar, external_storage_scalar) = match precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => (ScalarType::F64, ScalarType::F64),
            Precision::F64 => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional zero-padding boundary",
                    precision: "f64",
                });
            }
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32)
            }
            Precision::F64ComputeF32Storage => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional zero-padding boundary",
                    precision: "f64-compute/f32-storage",
                });
            }
            Precision::DoubleDouble => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional zero-padding boundary",
                    precision: "double-double",
                });
            }
            Precision::DoubleDoubleF64Storage => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional zero-padding boundary",
                    precision: "double-double/f64-storage",
                });
            }
        };
        let local_size = tensor_len.min(device.max_threads_per_block.max(1)).min(256);
        let operation = ZeroPadPassOperation::for_boundary(direction, domain);
        let (input_storage_scalar, output_storage_scalar) = if operation.is_input_boundary() {
            (external_storage_scalar, scalar)
        } else {
            (scalar, external_storage_scalar)
        };
        let pass = Self {
            name: format!(
                "vkfft_nd_zero_pad_{}_{:?}",
                operation.name_suffix(),
                dimensions
            ),
            scalar,
            input_storage_scalar,
            output_storage_scalar,
            direction,
            dimensions: dimensions.to_vec(),
            tensor_len,
            batch_count,
            grouped_batch: 1,
            ranges: ranges.to_vec(),
            operation,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "multidimensional zero-pad workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "multidimensional zero-pad batch count",
                })?,
                y: 1,
                z: 1,
            },
        };
        pass.validate()?;
        Ok(pass)
    }

    pub(crate) fn with_grouped_batch(mut self, grouped_batch: usize) -> Result<Self> {
        if grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional zero-pad groupedBatch must be non-zero",
            ));
        }
        self.grouped_batch = grouped_batch;
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "multidimensional zero-pad grouped dispatch count",
                }
            })?;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_compute_storage_boundary(mut self) -> Result<Self> {
        if self.operation.is_input_boundary() {
            self.input_storage_scalar = self.scalar;
        } else {
            self.output_storage_scalar = self.scalar;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.is_empty()
            || self.dimensions.len() != self.ranges.len()
            || self.tensor_len == 0
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.workgroup_size.x == 0
            || self.dispatch.x as usize != self.batch_count.div_ceil(self.grouped_batch)
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "invalid multidimensional zero-padding boundary pass",
            ));
        }
        if self.scalar == ScalarType::F16 {
            return Err(VkFftError::InvalidKernelIr(
                "binary16 cannot be used as multidimensional zero-padding compute scalar",
            ));
        }
        let supported_storage = |storage: ScalarType| {
            storage == self.scalar
                || matches!(
                    (self.scalar, storage),
                    (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
                )
        };
        if !supported_storage(self.input_storage_scalar)
            || !supported_storage(self.output_storage_scalar)
            || if self.operation.is_input_boundary() {
                self.output_storage_scalar != self.scalar
            } else {
                self.input_storage_scalar != self.scalar
            }
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional zero-padding storage boundary is inconsistent with compute precision/direction",
            ));
        }
        if !matches!(
            (self.direction, self.operation),
            (
                Direction::Forward,
                ZeroPadPassOperation::PrepareForwardInput
            ) | (
                Direction::Forward,
                ZeroPadPassOperation::FinalizeForwardOutput
            ) | (
                Direction::Inverse,
                ZeroPadPassOperation::FinalizeInverseOutput
            ) | (
                Direction::Inverse,
                ZeroPadPassOperation::PrepareInverseInput
            )
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional zero-padding operation does not match transform direction",
            ));
        }
        let mut product = 1usize;
        for (axis, (length, range)) in self.dimensions.iter().zip(&self.ranges).enumerate() {
            product = product
                .checked_mul(*length)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional zero-pad validation tensor length",
                })?;
            if let Some(range) = range
                && (range.left > range.right || range.right > *length)
            {
                return Err(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left: range.left,
                    right: range.right,
                    length: *length,
                });
            }
        }
        if product != self.tensor_len {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional zero-pad tensor length is inconsistent",
            ));
        }
        Ok(())
    }

    pub fn contains_linear_index(&self, mut index: usize) -> bool {
        for axis in (0..self.dimensions.len()).rev() {
            let length = self.dimensions[axis];
            let coordinate = index % length;
            index /= length;
            if self.ranges[axis].is_some_and(|range| range.contains(coordinate)) {
                return true;
            }
        }
        false
    }
}

pub fn execute_nd_zero_pad_pass(
    pass: &NdZeroPadPassIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    pass.validate()?;
    let expected =
        pass.tensor_len
            .checked_mul(pass.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional zero-pad element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut output = input.to_vec();
    for batch in 0..pass.batch_count {
        let base = batch * pass.tensor_len;
        for linear in 0..pass.tensor_len {
            if pass.contains_linear_index(linear) {
                output[base + linear] = Complex64::default();
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor};

    #[test]
    fn mixed_storage_zero_pad_boundary_is_directional() {
        let mut device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        device.supports_f64 = true;
        for (precision, compute, storage) in [
            (
                Precision::F16StorageF32Compute,
                ScalarType::F32,
                ScalarType::F16,
            ),
            (
                Precision::F64ComputeF32Storage,
                ScalarType::F64,
                ScalarType::F32,
            ),
        ] {
            let forward = ZeroPadPassIr::build(
                64,
                2,
                precision,
                Direction::Forward,
                ZeroPaddingRange::new(32, 64),
                device,
            )
            .unwrap();
            assert_eq!(forward.scalar, compute);
            assert_eq!(forward.input_storage_scalar, storage);
            assert_eq!(forward.output_storage_scalar, compute);

            let inverse = ZeroPadPassIr::build(
                64,
                2,
                precision,
                Direction::Inverse,
                ZeroPaddingRange::new(32, 64),
                device,
            )
            .unwrap();
            assert_eq!(inverse.scalar, compute);
            assert_eq!(inverse.input_storage_scalar, compute);
            assert_eq!(inverse.output_storage_scalar, storage);
        }
    }

    #[test]
    fn frequency_zero_padding_boundary_reverses_directional_ownership() {
        let mut device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        device.supports_f64 = true;
        for (precision, compute, storage) in [
            (
                Precision::F16StorageF32Compute,
                ScalarType::F32,
                ScalarType::F16,
            ),
            (
                Precision::F64ComputeF32Storage,
                ScalarType::F64,
                ScalarType::F32,
            ),
        ] {
            let forward = ZeroPadPassIr::build_with_domain(
                64,
                2,
                precision,
                Direction::Forward,
                ZeroPaddingRange::new(16, 32),
                ZeroPaddingDomain::Frequency,
                device,
            )
            .unwrap();
            assert_eq!(
                forward.operation,
                ZeroPadPassOperation::FinalizeForwardOutput
            );
            assert_eq!(forward.input_storage_scalar, compute);
            assert_eq!(forward.output_storage_scalar, storage);

            let inverse = ZeroPadPassIr::build_with_domain(
                64,
                2,
                precision,
                Direction::Inverse,
                ZeroPaddingRange::new(16, 32),
                ZeroPaddingDomain::Frequency,
                device,
            )
            .unwrap();
            assert_eq!(inverse.operation, ZeroPadPassOperation::PrepareInverseInput);
            assert_eq!(inverse.input_storage_scalar, storage);
            assert_eq!(inverse.output_storage_scalar, compute);

            let ranges = vec![None, Some(ZeroPaddingRange::new(2, 4))];
            let nd_forward = NdZeroPadPassIr::build_with_domain(
                &[3, 4],
                2,
                precision,
                Direction::Forward,
                &ranges,
                ZeroPaddingDomain::Frequency,
                device,
            )
            .unwrap();
            assert_eq!(
                nd_forward.operation,
                ZeroPadPassOperation::FinalizeForwardOutput
            );
            assert_eq!(nd_forward.input_storage_scalar, compute);
            assert_eq!(nd_forward.output_storage_scalar, storage);

            let nd_inverse = NdZeroPadPassIr::build_with_domain(
                &[3, 4],
                2,
                precision,
                Direction::Inverse,
                &ranges,
                ZeroPaddingDomain::Frequency,
                device,
            )
            .unwrap();
            assert_eq!(
                nd_inverse.operation,
                ZeroPadPassOperation::PrepareInverseInput
            );
            assert_eq!(nd_inverse.input_storage_scalar, storage);
            assert_eq!(nd_inverse.output_storage_scalar, compute);
        }
    }

    #[test]
    fn storage_preserving_zero_pad_allows_tail_group_larger_than_batch() {
        let pass = ZeroPadPassIr::build_storage_preserving_with_domain(
            8,
            2,
            ScalarType::DoubleDouble,
            Direction::Forward,
            ZeroPaddingRange::new(2, 5),
            3,
            ZeroPaddingDomain::Spatial,
        )
        .unwrap();
        assert_eq!(pass.grouped_batch, 3);
        assert_eq!(pass.dispatch.x, 1);
        pass.validate().unwrap();

        let input = (0..16)
            .map(|index| Complex64::new(index as f64 + 1.0, -(index as f64)))
            .collect::<Vec<_>>();
        let output = execute_zero_pad_pass(&pass, &input).unwrap();
        for batch in 0..2 {
            let base = batch * 8;
            assert_eq!(&output[base..base + 2], &input[base..base + 2]);
            assert!(
                output[base + 2..base + 5]
                    .iter()
                    .all(|value| *value == Complex64::default())
            );
            assert_eq!(&output[base + 5..base + 8], &input[base + 5..base + 8]);
        }
    }

    #[test]
    fn multidimensional_mixed_storage_zero_pad_boundary_is_directional() {
        let mut device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        device.supports_f64 = true;
        let ranges = vec![None, Some(ZeroPaddingRange::new(2, 4))];
        for (precision, compute, storage) in [
            (
                Precision::F16StorageF32Compute,
                ScalarType::F32,
                ScalarType::F16,
            ),
            (
                Precision::F64ComputeF32Storage,
                ScalarType::F64,
                ScalarType::F32,
            ),
        ] {
            let forward =
                NdZeroPadPassIr::build(&[3, 4], 2, precision, Direction::Forward, &ranges, device)
                    .unwrap();
            assert_eq!(forward.scalar, compute);
            assert_eq!(forward.input_storage_scalar, storage);
            assert_eq!(forward.output_storage_scalar, compute);

            let inverse =
                NdZeroPadPassIr::build(&[3, 4], 2, precision, Direction::Inverse, &ranges, device)
                    .unwrap();
            assert_eq!(inverse.scalar, compute);
            assert_eq!(inverse.input_storage_scalar, compute);
            assert_eq!(inverse.output_storage_scalar, storage);
        }
    }

    #[test]
    fn zero_pad_pass_preserves_extent_and_zeros_only_the_requested_interval() {
        let pass = ZeroPadPassIr::build(
            8,
            2,
            Precision::F32,
            Direction::Forward,
            ZeroPaddingRange::new(4, 8),
            DeviceProfile::generic(Backend::Cuda, GpuVendor::Nvidia),
        )
        .unwrap();
        let input = (0..16)
            .map(|index| Complex64::new(index as f64 + 1.0, -(index as f64)))
            .collect::<Vec<_>>();
        let output = execute_zero_pad_pass(&pass, &input).unwrap();
        assert_eq!(output.len(), input.len());
        for batch in 0..2 {
            let base = batch * 8;
            assert_eq!(&output[base..base + 4], &input[base..base + 4]);
            assert!(
                output[base + 4..base + 8]
                    .iter()
                    .all(|value| *value == Complex64::default())
            );
        }
    }
}
