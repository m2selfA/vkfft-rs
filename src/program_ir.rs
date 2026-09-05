//! Backend-neutral multi-pass program/resource IR.
//!
//! Kernel IR describes math inside one dispatch. `ProgramIr` describes how several
//! dispatches share external buffers, scratch storage, immutable LUTs, and barriers.
//! It is intentionally backend-neutral so Bluestein, FFT-convolution Rader, future
//! mixed-radix programs, and Four-step transforms can share one execution model.

use crate::bluestein_ir::BluesteinPipelineIr;
use crate::complex::Complex64;
use crate::convolution_ir::{ConvolutionIr, NdConvolutionIr, NdRealConvolutionIr};
use crate::double_double::ComplexDoubleDouble;
use crate::double_double_ir::{
    DoubleDoubleBluesteinIr, DoubleDoubleDirectRaderIr, DoubleDoubleFftRaderIr,
    DoubleDoubleNdFftIr, DoubleDoubleNdR2rIr, DoubleDoubleNdRealFftIr, DoubleDoubleOneDimIr,
    DoubleDoubleR2rAlgorithm, DoubleDoubleR2rIr, DoubleDoubleRealFftIr, DoubleDoubleStockhamIr,
};
use crate::double_double_recursive_ir::{
    DoubleDoubleCooleyTukeyOutputModifier, DoubleDoubleForcedRaderThreeUploadComponentIr,
    DoubleDoubleRecursiveFftIr, DoubleDoubleRecursiveFftNodeIr,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{BufferAccess, BufferRole, DispatchGeometry, KernelIr, ScalarType};
use crate::mixed_ir::{MixedPrimeRaderIr, MixedRaderStockhamIr};
use crate::nd_ir::NdFftIr;
use crate::nd_real_ir::NdRealFftIr;
use crate::one_dim_ir::OneDimFftIr;
use crate::r2r_ir::{NdR2rIr, R2rIr};
use crate::rader_ir::{RaderDirectIr, RaderFftInputStrategy, RaderFftPipelineIr};
use crate::real_ir::{RealFftIr, RealFftKind};
use crate::recursive_ir::{
    CooleyTukeyInputModifier, CooleyTukeyOutputModifier, RecursiveFftIr, RecursiveFftNodeIr,
};
use crate::zero_pad_ir::ZeroPadPassIr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramResourceId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramResourceKind {
    Input,
    Output,
    Scratch,
    LookupTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgramElementShape {
    Complex,
    Scalar,
}

impl ProgramElementShape {
    pub const fn element_bytes(self, scalar: ScalarType) -> usize {
        match self {
            Self::Complex => scalar.complex_bytes(),
            Self::Scalar => scalar.bytes(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalBufferLayout {
    pub logical_len: usize,
    pub physical_stride: usize,
    pub batch_count: usize,
    pub element_shape: ProgramElementShape,
}

impl ExternalBufferLayout {
    pub fn logical_elements(self) -> Result<usize> {
        self.logical_len
            .checked_mul(self.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "program external logical element count",
            })
    }

    pub fn physical_elements(self) -> Result<usize> {
        self.physical_stride
            .checked_mul(self.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "program external physical element count",
            })
    }

    fn validate(self) -> Result<()> {
        if self.logical_len == 0 || self.physical_stride == 0 || self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "program external buffer layout must be non-zero",
            ));
        }
        if self.logical_len > self.physical_stride {
            return Err(VkFftError::InvalidKernelIr(
                "program external logical length must fit inside physical stride",
            ));
        }
        let _ = self.logical_elements()?;
        let _ = self.physical_elements()?;
        Ok(())
    }
}

impl ProgramResource {
    pub const fn element_shape(&self) -> ProgramElementShape {
        match self.external_layout {
            Some(layout) => layout.element_shape,
            None => ProgramElementShape::Complex,
        }
    }

    pub const fn element_bytes(&self) -> usize {
        self.element_shape().element_bytes(self.scalar)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgramResourceInitialization {
    ExternalInput,
    Zeroed,
    Complex64(Vec<Complex64>),
    /// Immutable true double-double complex data. Each logical element contains
    /// four binary64 words (`re.hi`, `re.lo`, `im.hi`, `im.lo`).
    ComplexDoubleDouble(Vec<ComplexDoubleDouble>),
    /// Lazily materialized direction-independent Stockham unit roots. Keeping only
    /// the period in ProgramIr avoids embedding multi-megabyte Four-step tables in
    /// the IR while preserving deterministic immutable-LUT identity.
    StockhamUnitRoots {
        len: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProgramResource {
    pub id: ProgramResourceId,
    pub name: String,
    pub kind: ProgramResourceKind,
    pub scalar: ScalarType,
    pub elements: usize,
    pub external_layout: Option<ExternalBufferLayout>,
    pub initialization: ProgramResourceInitialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramAllocationId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramAllocationKind {
    ExternalInput,
    ExternalOutput,
    Scratch,
    LookupTable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramAllocation {
    pub id: ProgramAllocationId,
    pub kind: ProgramAllocationKind,
    pub scalar: ScalarType,
    pub elements: usize,
    pub resources: Vec<ProgramResourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramMemoryPlan {
    pub allocations: Vec<ProgramAllocation>,
    pub resource_allocations: Vec<ProgramAllocationId>,
}

impl ProgramMemoryPlan {
    pub fn allocation_for(&self, resource: ProgramResourceId) -> Result<ProgramAllocationId> {
        self.resource_allocations
            .get(resource.0)
            .copied()
            .ok_or(VkFftError::InvalidKernelIr(
                "program memory plan references an unknown logical resource",
            ))
    }

    pub fn allocated_elements(&self) -> Result<usize> {
        self.allocations
            .iter()
            .try_fold(0usize, |total, allocation| {
                total
                    .checked_add(allocation.elements)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "program physical allocation element count",
                    })
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramPassBinding {
    pub binding: u32,
    pub resource: ProgramResourceId,
    pub role: BufferRole,
    pub access: BufferAccess,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramPass {
    pub name: String,
    pub dispatch: DispatchGeometry,
    pub bindings: Vec<ProgramPassBinding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProgramIr {
    pub name: String,
    pub scalar: ScalarType,
    pub resources: Vec<ProgramResource>,
    pub passes: Vec<ProgramPass>,
}

impl ProgramIr {
    pub fn double_double_stockham(ir: &DoubleDoubleStockhamIr) -> Result<Self> {
        ir.validate()?;
        if ir.sequence_len > 64 {
            let core = Self::double_double_stockham_multi_pass(ir)?;
            return Self::wrap_double_double_zero_pad(
                ir.zero_pad_pass.as_ref(),
                ir.sequence_len,
                ir.batch_count,
                "double_double_stockham",
                core,
            );
        }
        let elements =
            ir.sequence_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham program element count",
                })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Stockham program requires DD or F64 external storage",
                ));
            }
        };
        let layout = ExternalBufferLayout {
            logical_len: ir.sequence_len,
            physical_stride: ir.sequence_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let twiddles = ir.twiddles.packed_values();
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let mut bindings = vec![
            binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
        ];
        if !twiddles.is_empty() {
            let lut_id = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: lut_id,
                name: "double_double_stockham_twiddles".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: twiddles.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(twiddles),
            });
            bindings.push(ProgramPassBinding {
                binding: 2,
                resource: lut_id,
                role: BufferRole::TwiddleLookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        let dispatch_x =
            u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double Stockham program dispatch count",
            })?;
        let program = Self {
            name: format!("{}_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes: vec![ProgramPass {
                name: ir.name.clone(),
                dispatch: DispatchGeometry {
                    x: dispatch_x,
                    y: 1,
                    z: 1,
                },
                bindings,
            }],
        };
        program.validate()?;
        Self::wrap_double_double_zero_pad(
            ir.zero_pad_pass.as_ref(),
            ir.sequence_len,
            ir.batch_count,
            "double_double_stockham",
            program,
        )
    }

    fn double_double_stockham_internal(ir: &DoubleDoubleStockhamIr) -> Result<Self> {
        if ir.external_storage != crate::PrecisionStorage::DoubleDouble
            || ir.zero_pad_pass.is_some()
        {
            return Err(VkFftError::InvalidKernelIr(
                "internal double-double Stockham child requires unmodified DD boundaries",
            ));
        }
        let mut program = Self::double_double_stockham(ir)?;
        if ir.sequence_len <= 64 {
            return Ok(program);
        }
        if program.passes.len() != ir.stages.len() + 2 || ir.stages.is_empty() {
            return Err(VkFftError::InvalidKernelIr(
                "internal double-double Stockham child has an unexpected pass graph",
            ));
        }
        program.passes.remove(0);
        program.passes.pop();
        let first_input = program
            .passes
            .first_mut()
            .and_then(|pass| {
                pass.bindings
                    .iter_mut()
                    .find(|binding| binding.binding == 0)
            })
            .ok_or(VkFftError::InvalidKernelIr(
                "internal double-double Stockham first stage is missing input binding",
            ))?;
        first_input.resource = ProgramResourceId(0);
        let final_output = program
            .passes
            .last_mut()
            .and_then(|pass| {
                pass.bindings
                    .iter_mut()
                    .find(|binding| binding.binding == 1)
            })
            .ok_or(VkFftError::InvalidKernelIr(
                "internal double-double Stockham final stage is missing output binding",
            ))?;
        final_output.resource = ProgramResourceId(1);
        program.name = format!("{}_internal_program", ir.name);
        program.validate()?;
        Ok(program)
    }

    fn double_double_stockham_multi_pass(ir: &DoubleDoubleStockhamIr) -> Result<Self> {
        let elements =
            ir.sequence_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham multi-pass element count",
                })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Stockham multi-pass requires DD or F64 external storage",
                ));
            }
        };
        let layout = ExternalBufferLayout {
            logical_len: ir.sequence_len,
            physical_stride: ir.sequence_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let twiddles = ir.twiddles.packed_values();
        let resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_stockham_scratch_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "double_double_stockham_scratch_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(4),
                name: "double_double_stockham_twiddles".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: twiddles.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(twiddles),
            },
        ];
        let dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double Stockham multi-pass dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        let mut passes = Vec::with_capacity(ir.stages.len() + 2);
        passes.push(ProgramPass {
            name: format!("{}_promote", ir.name),
            dispatch,
            bindings: vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 2, BufferRole::Output, BufferAccess::WriteOnly),
            ],
        });
        let mut source = 2usize;
        let mut target = 3usize;
        for stage in &ir.stages {
            passes.push(ProgramPass {
                name: format!("{}_stage_{}", ir.name, stage.index),
                dispatch,
                bindings: vec![
                    binding(0, source, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, target, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, 4, BufferRole::TwiddleLookupTable, BufferAccess::ReadOnly),
                ],
            });
            core::mem::swap(&mut source, &mut target);
        }
        passes.push(ProgramPass {
            name: format!("{}_finalize", ir.name),
            dispatch,
            bindings: vec![
                binding(0, source, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
            ],
        });
        let program = Self {
            name: format!("{}_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_direct_rader(ir: &DoubleDoubleDirectRaderIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.prime
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double direct Rader program element count",
                })?;
        let storage_scalar = |storage| match storage {
            crate::PrecisionStorage::DoubleDouble => Ok(ScalarType::DoubleDouble),
            crate::PrecisionStorage::F64 => Ok(ScalarType::F64),
            _ => Err(VkFftError::InvalidKernelIr(
                "double-double direct Rader program requires DD or F64 caller storage",
            )),
        };
        let input_scalar = storage_scalar(ir.input_storage)?;
        let output_scalar = storage_scalar(ir.output_storage)?;
        let layout = ExternalBufferLayout {
            logical_len: ir.prime,
            physical_stride: ir.prime,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let lut = ir.table.twiddles_by_generator_power.clone();
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: input_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: output_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_rader_twiddles".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: lut.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(lut),
            },
        ];
        let mut bindings = vec![
            binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
            binding(2, 2, BufferRole::LookupTable, BufferAccess::ReadOnly),
        ];
        let parent_period = match ir.io_mapping {
            crate::StockhamIoMapping::FourStepRight(mapping) => Some(mapping.logical_len),
            crate::StockhamIoMapping::FourStepThreeUpload2(mapping) => Some(mapping.logical_len),
            crate::StockhamIoMapping::FourStepThreeUpload1(mapping) => {
                let [a, b, _] = mapping.axis_split;
                Some(a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double direct Rader three-upload parent root period",
                })?)
            }
            _ => None,
        };
        if let Some(period) = parent_period {
            let roots = crate::lut::unit_root_table_double_double(period, ir.direction)?;
            let root_id = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: root_id,
                name: "double_double_rader_four_step_roots".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: roots.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(roots),
            });
            bindings.push(ProgramPassBinding {
                binding: 3,
                resource: root_id,
                role: BufferRole::TwiddleLookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        let program = Self {
            name: format!("{}_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes: vec![ProgramPass {
                name: ir.name.clone(),
                dispatch: DispatchGeometry {
                    x: u32::try_from(ir.batch_group_count()).map_err(|_| {
                        VkFftError::ValueOutOfRange {
                            field: "double-double direct Rader program dispatch count",
                        }
                    })?,
                    y: 1,
                    z: 1,
                },
                bindings,
            }],
        };
        program.validate()?;
        Self::wrap_double_double_zero_pad(
            ir.zero_pad_pass.as_ref(),
            ir.prime,
            ir.batch_count,
            "double_double_direct_rader",
            program,
        )
    }

    pub fn double_double_one_dim(ir: &DoubleDoubleOneDimIr) -> Result<Self> {
        match ir {
            DoubleDoubleOneDimIr::Stockham(ir) => Self::double_double_stockham(ir),
            DoubleDoubleOneDimIr::DirectRader(ir) => Self::double_double_direct_rader(ir),
            DoubleDoubleOneDimIr::FftRader(ir) => Self::double_double_fft_rader(ir),
            DoubleDoubleOneDimIr::Bluestein(ir) => Self::double_double_bluestein(ir),
            DoubleDoubleOneDimIr::Recursive(ir) => Self::double_double_recursive(ir),
        }
    }

    pub fn double_double_recursive(ir: &DoubleDoubleRecursiveFftIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double recursive program element count",
                })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double recursive ProgramIr requires DD or F64 external storage",
                ));
            }
        };
        let layout = ExternalBufferLayout {
            logical_len: ir.logical_len,
            physical_stride: ir.logical_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let core = if ir.two_upload_four_step_plan.is_some() {
            Self::double_double_recursive_two_upload_four_step(ir, resources)?
        } else if ir.three_upload_four_step_plan.is_some() {
            Self::double_double_recursive_three_upload_four_step(ir, resources)?
        } else if let Some(components) = ir.forced_rader_three_upload_mapped_components()? {
            Self::double_double_recursive_forced_rader_three_upload(ir, resources, components)?
        } else if let Some((high, _mapping)) = ir.forced_rader_two_upload_mapped_high_stockham()? {
            Self::double_double_recursive_forced_rader_stockham_high_boundary(ir, resources, high)?
        } else if let Some(mapped_high) = ir.forced_rader_two_upload_mapped_high_component()? {
            Self::double_double_recursive_forced_rader_high_boundary(ir, resources, mapped_high)?
        } else {
            let mut passes = Vec::new();
            let mut serial = 0usize;
            flatten_double_double_recursive_node(
                &ir.root,
                ProgramResourceId(0),
                ProgramResourceId(1),
                &mut resources,
                &mut passes,
                &mut serial,
            )?;
            let program = Self {
                name: format!("{}_program", ir.name),
                scalar: ScalarType::DoubleDouble,
                resources,
                passes,
            };
            program.validate()?;
            program
        };
        Self::wrap_double_double_zero_pad(
            ir.zero_pad_pass.as_ref(),
            ir.logical_len,
            ir.batch_count,
            "double_double_recursive",
            core,
        )
    }

    fn wrap_double_double_zero_pad(
        pass: Option<&ZeroPadPassIr>,
        logical_len: usize,
        batch_count: usize,
        stem: &str,
        mut program: Self,
    ) -> Result<Self> {
        let Some(pass) = pass else {
            return Ok(program);
        };
        pass.validate()?;
        let input = program.input_resource()?.id;
        let output = program.output_resource()?.id;
        let elements =
            logical_len
                .checked_mul(batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double zero-pad scratch element count",
                })?;
        let boundary = ProgramResourceId(program.resources.len());
        let input_boundary = pass.operation.is_input_boundary();
        program.resources.push(ProgramResource {
            id: boundary,
            name: format!(
                "{stem}_zero_padded_{}",
                if input_boundary { "input" } else { "output" }
            ),
            kind: ProgramResourceKind::Scratch,
            scalar: if input_boundary {
                pass.output_storage_scalar
            } else {
                pass.input_storage_scalar
            },
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        if input_boundary {
            for program_pass in &mut program.passes {
                for pass_binding in &mut program_pass.bindings {
                    if pass_binding.resource == input {
                        pass_binding.resource = boundary;
                    }
                }
            }
            program.passes.insert(
                0,
                ProgramPass {
                    name: pass.name.clone(),
                    dispatch: pass.dispatch,
                    bindings: vec![
                        ProgramPassBinding {
                            binding: 0,
                            resource: input,
                            role: BufferRole::Input,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 1,
                            resource: boundary,
                            role: BufferRole::Output,
                            access: BufferAccess::WriteOnly,
                        },
                    ],
                },
            );
        } else {
            for program_pass in &mut program.passes {
                for pass_binding in &mut program_pass.bindings {
                    if pass_binding.resource == output {
                        pass_binding.resource = boundary;
                    }
                }
            }
            program.passes.push(ProgramPass {
                name: pass.name.clone(),
                dispatch: pass.dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: boundary,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    ProgramPassBinding {
                        binding: 1,
                        resource: output,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
        }
        program.validate()?;
        Ok(program)
    }

    fn double_double_recursive_forced_rader_stockham_high_boundary(
        ir: &DoubleDoubleRecursiveFftIr,
        mut resources: Vec<ProgramResource>,
        high: &DoubleDoubleStockhamIr,
    ) -> Result<Self> {
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham-high forced-Rader boundary requires a Cooley-Tukey root",
            ));
        };
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham-high forced-Rader scratch element count",
                })?;
        let exchange = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: exchange,
            name: "double_double_rader_four_step_exchange".to_owned(),
            kind: ProgramResourceKind::Scratch,
            scalar: ScalarType::DoubleDouble,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        let high_twiddle_values = high.twiddles.packed_values();
        let high_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: high_twiddles,
            name: "double_double_forced_rader_two_upload_1_stockham_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: high_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(high_twiddle_values),
        });
        let parent_roots = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: parent_roots,
            name: "double_double_forced_rader_two_upload_1_parent_roots".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: root.twiddles.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                root.twiddles.clone(),
            ),
        });
        let passes = vec![ProgramPass {
            name: format!("{}_forced_rader_two_upload_1", ir.name),
            dispatch: DispatchGeometry {
                x: u32::try_from(high.batch_group_count()).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "double-double forced-Rader high Stockham dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            },
            bindings: vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                ProgramPassBinding {
                    binding: 1,
                    resource: exchange,
                    role: BufferRole::Output,
                    access: BufferAccess::WriteOnly,
                },
                ProgramPassBinding {
                    binding: 2,
                    resource: high_twiddles,
                    role: BufferRole::TwiddleLookupTable,
                    access: BufferAccess::ReadOnly,
                },
                ProgramPassBinding {
                    binding: 3,
                    resource: parent_roots,
                    role: BufferRole::Auxiliary,
                    access: BufferAccess::ReadOnly,
                },
            ],
        }];
        Self::finish_double_double_recursive_forced_rader_two_upload(
            ir, resources, passes, 0, exchange, elements,
        )
    }

    fn double_double_recursive_forced_rader_high_boundary(
        ir: &DoubleDoubleRecursiveFftIr,
        mut resources: Vec<ProgramResource>,
        mapped_high: DoubleDoubleRecursiveFftNodeIr,
    ) -> Result<Self> {
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_) = &ir.root else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double mapped forced-Rader boundary requires a Cooley-Tukey root",
            ));
        };
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double mapped forced-Rader scratch element count",
                })?;
        let exchange = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: exchange,
            name: "double_double_rader_four_step_exchange".to_owned(),
            kind: ProgramResourceKind::Scratch,
            scalar: ScalarType::DoubleDouble,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        let mut passes = Vec::new();
        let mut serial = 0usize;
        flatten_double_double_recursive_node(
            &mapped_high,
            ProgramResourceId(0),
            exchange,
            &mut resources,
            &mut passes,
            &mut serial,
        )?;
        Self::finish_double_double_recursive_forced_rader_two_upload(
            ir, resources, passes, serial, exchange, elements,
        )
    }

    fn finish_double_double_recursive_forced_rader_two_upload(
        ir: &DoubleDoubleRecursiveFftIr,
        mut resources: Vec<ProgramResource>,
        mut passes: Vec<ProgramPass>,
        mut serial: usize,
        exchange: ProgramResourceId,
        elements: usize,
    ) -> Result<Self> {
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double mapped forced-Rader finalize requires a Cooley-Tukey root",
            ));
        };
        if let Some(mapped_low) = ir.forced_rader_two_upload_mapped_low_component()? {
            flatten_double_double_recursive_node(
                &mapped_low,
                exchange,
                ProgramResourceId(1),
                &mut resources,
                &mut passes,
                &mut serial,
            )?;
        } else if let Some((low, _mapping)) = ir.forced_rader_two_upload_mapped_low_stockham()? {
            let child = ProgramIr::double_double_stockham(low)?;
            serial += 1;
            append_double_double_recursive_child(
                &child,
                &format!("vkfft_dd_recursive_leaf_{}", serial),
                exchange,
                ProgramResourceId(1),
                &mut resources,
                &mut passes,
                true,
            )?;
            let final_pass = passes.last_mut().ok_or(VkFftError::InvalidKernelIr(
                "double-double mapped forced-Rader low upload emitted no final pass",
            ))?;
            if low.sequence_len <= 64 {
                if !final_pass.name.ends_with(&low.name) {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double mapped forced-Rader monolithic low upload does not end in its Stockham pass",
                    ));
                }
                final_pass.name = format!("{}_four_step_left", low.name);
            } else {
                let expected_finalize = format!("{}_finalize", low.name);
                if !final_pass.name.ends_with(&expected_finalize) {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double mapped forced-Rader low upload does not end in a Stockham finalize pass",
                    ));
                }
                final_pass.name = format!("{}_four_step_left_finalize", low.name);
            }
        } else {
            let left_output = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: left_output,
                name: "double_double_rader_four_step_left_output".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            });
            flatten_double_double_recursive_node(
                &root.left,
                exchange,
                left_output,
                &mut resources,
                &mut passes,
                &mut serial,
            )?;
            passes.push(two_buffer_pass(
                &root.scatter_output.name,
                root.scatter_output.dispatch,
                left_output.0,
                1,
            ));
        }
        let program = Self {
            name: format!("{}_mapped_rader_four_step_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    fn double_double_recursive_forced_rader_three_upload(
        ir: &DoubleDoubleRecursiveFftIr,
        mut resources: Vec<ProgramResource>,
        components: Vec<DoubleDoubleForcedRaderThreeUploadComponentIr>,
    ) -> Result<Self> {
        if components.len() != 3
            || components
                .iter()
                .map(DoubleDoubleForcedRaderThreeUploadComponentIr::upload_id)
                .collect::<Vec<_>>()
                != vec![2, 1, 0]
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double forced-Rader three-upload components are out of execution order",
            ));
        }
        let schedule =
            ir.rader_forced_upload_schedule
                .as_ref()
                .ok_or(VkFftError::InvalidKernelIr(
                    "double-double forced-Rader three-upload ProgramIr requires schedule metadata",
                ))?;
        let [a, b, c]: [usize; 3] = schedule.axis_split.as_slice().try_into().map_err(|_| {
            VkFftError::InvalidKernelIr(
                "double-double forced-Rader three-upload ProgramIr requires three axis splits",
            )
        })?;
        let component_lengths = components
            .iter()
            .map(DoubleDoubleForcedRaderThreeUploadComponentIr::logical_len)
            .collect::<Vec<_>>();
        if component_lengths != vec![c, b, a] {
            return Err(VkFftError::InvalidKernelIr(
                "double-double forced-Rader three-upload component lengths do not match axisSplit",
            ));
        }
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double forced-Rader three-upload scratch element count",
                })?;
        let output_is_compute = resources
            .get(1)
            .ok_or(VkFftError::InvalidKernelIr(
                "double-double forced-Rader three-upload output resource is missing",
            ))?
            .scalar
            == ScalarType::DoubleDouble;
        let scratch0 = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: scratch0,
            name: "double_double_forced_rader_three_upload_scratch_0".to_owned(),
            kind: ProgramResourceKind::Scratch,
            scalar: ScalarType::DoubleDouble,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        let scratch1 = if output_is_compute {
            None
        } else {
            let scratch = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: scratch,
                name: "double_double_forced_rader_three_upload_scratch_1".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            });
            Some(scratch)
        };
        let (exchange0, exchange1) = if output_is_compute {
            (ProgramResourceId(1), scratch0)
        } else {
            (
                scratch0,
                scratch1.expect("mixed-storage path allocates scratch1"),
            )
        };
        let mut passes = Vec::new();
        let mut serial = 0usize;
        for component in components {
            let upload_id = component.upload_id();
            let (input, output) = match upload_id {
                2 => (ProgramResourceId(0), exchange0),
                1 => (exchange0, exchange1),
                0 => (exchange1, ProgramResourceId(1)),
                _ => unreachable!(),
            };
            match component {
                DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive { ir: node, .. } => {
                    flatten_double_double_recursive_node(
                        &node,
                        input,
                        output,
                        &mut resources,
                        &mut passes,
                        &mut serial,
                    )?;
                }
                DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                    ir: stockham, ..
                } => {
                    let twiddle_values = stockham.twiddles.packed_values();
                    let twiddle_id = ProgramResourceId(resources.len());
                    resources.push(ProgramResource {
                        id: twiddle_id,
                        name: format!(
                            "double_double_forced_rader_three_upload_{upload_id}_stockham_twiddles"
                        ),
                        kind: ProgramResourceKind::LookupTable,
                        scalar: ScalarType::DoubleDouble,
                        elements: twiddle_values.len(),
                        external_layout: None,
                        initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                            twiddle_values,
                        ),
                    });
                    let mut bindings = vec![
                        ProgramPassBinding {
                            binding: 0,
                            resource: input,
                            role: BufferRole::Input,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 1,
                            resource: output,
                            role: BufferRole::Output,
                            access: BufferAccess::WriteOnly,
                        },
                        ProgramPassBinding {
                            binding: 2,
                            resource: twiddle_id,
                            role: BufferRole::TwiddleLookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ];
                    let parent_period = match upload_id {
                        2 => Some(ir.logical_len),
                        1 => Some(a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double forced-Rader three-upload AB period",
                        })?),
                        0 => None,
                        _ => unreachable!(),
                    };
                    if let Some(period) = parent_period {
                        let roots =
                            crate::lut::unit_root_table_double_double(period, ir.direction)?;
                        let root_id = ProgramResourceId(resources.len());
                        resources.push(ProgramResource {
                            id: root_id,
                            name: format!(
                                "double_double_forced_rader_three_upload_{upload_id}_parent_roots"
                            ),
                            kind: ProgramResourceKind::LookupTable,
                            scalar: ScalarType::DoubleDouble,
                            elements: roots.len(),
                            external_layout: None,
                            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                                roots,
                            ),
                        });
                        bindings.push(ProgramPassBinding {
                            binding: 3,
                            resource: root_id,
                            role: BufferRole::Auxiliary,
                            access: BufferAccess::ReadOnly,
                        });
                    }
                    passes.push(ProgramPass {
                        name: format!("{}_forced_rader_three_upload_{upload_id}", ir.name),
                        dispatch: DispatchGeometry {
                            x: u32::try_from(stockham.batch_group_count()).map_err(|_| {
                                VkFftError::ValueOutOfRange {
                                    field: "double-double forced-Rader three-upload Stockham dispatch count",
                                }
                            })?,
                            y: 1,
                            z: 1,
                        },
                        bindings,
                    });
                }
            }
        }
        let program = Self {
            name: format!("{}_forced_rader_three_upload_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    fn double_double_recursive_two_upload_four_step(
        ir: &DoubleDoubleRecursiveFftIr,
        mut resources: Vec<ProgramResource>,
    ) -> Result<Self> {
        let plan = ir
            .two_upload_four_step_plan
            .ok_or(VkFftError::InvalidKernelIr(
                "double-double Four-step ProgramIr requires plan metadata",
            ))?;
        plan.validate()?;
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Four-step ProgramIr requires a Cooley-Tukey root",
            ));
        };
        let (
            DoubleDoubleRecursiveFftNodeIr::Stockham(left),
            DoubleDoubleRecursiveFftNodeIr::Stockham(right),
        ) = (&root.left, &root.right)
        else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Four-step ProgramIr requires Stockham leaves",
            ));
        };
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Four-step scratch element count",
                })?;
        let scratch = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: scratch,
            name: "double_double_four_step_scratch".to_owned(),
            kind: ProgramResourceKind::Scratch,
            scalar: ScalarType::DoubleDouble,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        let right_twiddle_values = right.twiddles.packed_values();
        let right_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: right_twiddles,
            name: "double_double_four_step_upload_1_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: right_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                right_twiddle_values,
            ),
        });
        let root_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: root_twiddles,
            name: "double_double_four_step_root_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: root.twiddles.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                root.twiddles.clone(),
            ),
        });
        let left_twiddle_values = left.twiddles.packed_values();
        let left_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: left_twiddles,
            name: "double_double_four_step_upload_0_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: left_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(left_twiddle_values),
        });
        let right_dispatch = DispatchGeometry {
            x: u32::try_from(right.batch_group_count()).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "double-double Four-step upload 1 dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        let left_dispatch = DispatchGeometry {
            x: u32::try_from(left.batch_group_count()).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "double-double Four-step upload 0 dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        let passes = vec![
            ProgramPass {
                name: format!("{}_four_step_upload_1", ir.name),
                dispatch: right_dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: scratch,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                    ProgramPassBinding {
                        binding: 2,
                        resource: right_twiddles,
                        role: BufferRole::TwiddleLookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                    ProgramPassBinding {
                        binding: 3,
                        resource: root_twiddles,
                        role: BufferRole::Auxiliary,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            },
            ProgramPass {
                name: format!("{}_four_step_upload_0", ir.name),
                dispatch: left_dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: scratch,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                    ProgramPassBinding {
                        binding: 2,
                        resource: left_twiddles,
                        role: BufferRole::TwiddleLookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            },
        ];
        let program = Self {
            name: format!("{}_four_step_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    fn double_double_recursive_three_upload_four_step(
        ir: &DoubleDoubleRecursiveFftIr,
        mut resources: Vec<ProgramResource>,
    ) -> Result<Self> {
        let plan = ir
            .three_upload_four_step_plan
            .ok_or(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step ProgramIr requires plan metadata",
            ))?;
        plan.validate()?;
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step ProgramIr requires a Cooley-Tukey root",
            ));
        };
        let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &root.left else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step ProgramIr requires a low Stockham leaf",
            ));
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step ProgramIr requires an upper Cooley-Tukey node",
            ));
        };
        let (
            DoubleDoubleRecursiveFftNodeIr::Stockham(middle),
            DoubleDoubleRecursiveFftNodeIr::Stockham(high),
        ) = (&upper.left, &upper.right)
        else {
            return Err(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step ProgramIr requires middle/high Stockham leaves",
            ));
        };
        let [a, b, c] = plan.axis_split;
        if low.sequence_len != a || middle.sequence_len != b || high.sequence_len != c {
            return Err(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step ProgramIr leaf lengths do not match axisSplit",
            ));
        }
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double three-upload Four-step scratch element count",
                })?;
        let output_is_compute = resources
            .get(1)
            .ok_or(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step output resource is missing",
            ))?
            .scalar
            == ScalarType::DoubleDouble;
        let scratch0 = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: scratch0,
            name: "double_double_three_upload_four_step_scratch_0".to_owned(),
            kind: ProgramResourceKind::Scratch,
            scalar: ScalarType::DoubleDouble,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        let scratch1 = if output_is_compute {
            None
        } else {
            let scratch = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: scratch,
                name: "double_double_three_upload_four_step_scratch_1".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            });
            Some(scratch)
        };
        let (exchange0, exchange1) = if output_is_compute {
            (ProgramResourceId(1), scratch0)
        } else {
            (
                scratch0,
                scratch1.expect("mixed-storage path allocates scratch1"),
            )
        };

        let high_twiddle_values = high.twiddles.packed_values();
        let high_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: high_twiddles,
            name: "double_double_three_upload_four_step_upload_2_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: high_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(high_twiddle_values),
        });
        let root_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: root_twiddles,
            name: "double_double_three_upload_four_step_root_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: root.twiddles.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                root.twiddles.clone(),
            ),
        });

        let middle_twiddle_values = middle.twiddles.packed_values();
        let middle_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: middle_twiddles,
            name: "double_double_three_upload_four_step_upload_1_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: middle_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                middle_twiddle_values,
            ),
        });
        let ab = a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double three-upload Four-step AB twiddle period",
        })?;
        let ab_twiddle_values = crate::lut::unit_root_table_double_double(ab, ir.direction)?;
        let ab_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: ab_twiddles,
            name: "double_double_three_upload_four_step_ab_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: ab_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(ab_twiddle_values),
        });

        let low_twiddle_values = low.twiddles.packed_values();
        let low_twiddles = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: low_twiddles,
            name: "double_double_three_upload_four_step_upload_0_twiddles".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: low_twiddle_values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(low_twiddle_values),
        });

        let dispatch = |leaf: &DoubleDoubleStockhamIr, field| {
            Ok(DispatchGeometry {
                x: u32::try_from(leaf.batch_group_count())
                    .map_err(|_| VkFftError::ValueOutOfRange { field })?,
                y: 1,
                z: 1,
            })
        };
        let passes = vec![
            ProgramPass {
                name: format!("{}_four_step_upload_2", ir.name),
                dispatch: dispatch(high, "double-double Four-step upload 2 dispatch count")?,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: exchange0,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                    ProgramPassBinding {
                        binding: 2,
                        resource: high_twiddles,
                        role: BufferRole::TwiddleLookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                    ProgramPassBinding {
                        binding: 3,
                        resource: root_twiddles,
                        role: BufferRole::Auxiliary,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            },
            ProgramPass {
                name: format!("{}_four_step_upload_1", ir.name),
                dispatch: dispatch(middle, "double-double Four-step upload 1 dispatch count")?,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: exchange0,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    ProgramPassBinding {
                        binding: 1,
                        resource: exchange1,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                    ProgramPassBinding {
                        binding: 2,
                        resource: middle_twiddles,
                        role: BufferRole::TwiddleLookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                    ProgramPassBinding {
                        binding: 3,
                        resource: ab_twiddles,
                        role: BufferRole::Auxiliary,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            },
            ProgramPass {
                name: format!("{}_four_step_upload_0", ir.name),
                dispatch: dispatch(low, "double-double Four-step upload 0 dispatch count")?,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: exchange1,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                    ProgramPassBinding {
                        binding: 2,
                        resource: low_twiddles,
                        role: BufferRole::TwiddleLookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            },
        ];
        let program = Self {
            name: format!("{}_three_upload_four_step_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_real(ir: &DoubleDoubleRealFftIr) -> Result<Self> {
        ir.validate()?;
        let full_elements =
            ir.length
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double real full element count",
                })?;
        let compact_elements = ir.half_spectrum_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double real compact element count",
            },
        )?;
        let child_elements = ir
            .transform
            .sequence_len()
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double real child element count",
            })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double real ProgramIr requires DD or F64 external storage",
                ));
            }
        };
        let (input_elements, output_elements, input_len, output_len) = match ir.kind {
            crate::RealFftKind::RealToComplex => (
                full_elements,
                compact_elements,
                ir.length,
                ir.half_spectrum_len,
            ),
            crate::RealFftKind::ComplexToReal => (
                compact_elements,
                full_elements,
                ir.half_spectrum_len,
                ir.length,
            ),
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements: input_elements,
                external_layout: Some(ExternalBufferLayout {
                    logical_len: input_len,
                    physical_stride: input_len,
                    batch_count: ir.batch_count,
                    element_shape: match ir.kind {
                        crate::RealFftKind::RealToComplex => ProgramElementShape::Scalar,
                        crate::RealFftKind::ComplexToReal => ProgramElementShape::Complex,
                    },
                }),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements: output_elements,
                external_layout: Some(ExternalBufferLayout {
                    logical_len: output_len,
                    physical_stride: output_len,
                    batch_count: ir.batch_count,
                    element_shape: match ir.kind {
                        crate::RealFftKind::RealToComplex => ProgramElementShape::Complex,
                        crate::RealFftKind::ComplexToReal => ProgramElementShape::Scalar,
                    },
                }),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_real_child_input".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: child_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "double_double_real_child_output".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: child_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let child = Self::double_double_one_dim(&ir.transform)?;
        if child.scalar != ScalarType::DoubleDouble
            || child.input_resource()?.scalar != ScalarType::DoubleDouble
            || child.output_resource()?.scalar != ScalarType::DoubleDouble
            || child.input_resource()?.elements != child_elements
            || child.output_resource()?.elements != child_elements
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real child ProgramIr must expose child-sized DD boundaries",
            ));
        }
        let mut resource_map = vec![None; child.resources.len()];
        for resource in &child.resources {
            let mapped = match resource.kind {
                ProgramResourceKind::Input => ProgramResourceId(2),
                ProgramResourceKind::Output => ProgramResourceId(3),
                _ => {
                    let new_id = ProgramResourceId(resources.len());
                    let mut cloned = resource.clone();
                    cloned.id = new_id;
                    cloned.name = format!("double_double_real_{}", resource.name);
                    cloned.external_layout = None;
                    resources.push(cloned);
                    new_id
                }
            };
            resource_map[resource.id.0] = Some(mapped);
        }
        let even_roots_resource = if ir.even_half_size {
            let id = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id,
                name: "double_double_real_even_roots".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: ir.even_roots.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                    ir.even_roots.clone(),
                ),
            });
            Some(id)
        } else {
            None
        };
        let boundary_dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double real boundary dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        let mut first_bindings = vec![
            binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, 2, BufferRole::Output, BufferAccess::WriteOnly),
        ];
        if ir.kind == crate::RealFftKind::ComplexToReal
            && let Some(roots) = even_roots_resource
        {
            first_bindings.push(ProgramPassBinding {
                binding: 2,
                resource: roots,
                role: BufferRole::LookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        let mut passes = Vec::with_capacity(child.passes.len() + 2);
        passes.push(ProgramPass {
            name: match (ir.kind, ir.even_half_size) {
                (crate::RealFftKind::RealToComplex, true) => {
                    "vkfft_dd_real_r2c_half_size_pack".to_owned()
                }
                (crate::RealFftKind::ComplexToReal, true) => {
                    "vkfft_dd_real_c2r_half_size_preprocess".to_owned()
                }
                (crate::RealFftKind::RealToComplex, false) => {
                    "vkfft_dd_real_r2c_promote".to_owned()
                }
                (crate::RealFftKind::ComplexToReal, false) => "vkfft_dd_real_c2r_expand".to_owned(),
            },
            dispatch: boundary_dispatch,
            bindings: first_bindings,
        });
        for child_pass in &child.passes {
            let mut pass = child_pass.clone();
            pass.name = format!("vkfft_dd_real_{}", child_pass.name);
            for binding in &mut pass.bindings {
                binding.resource = resource_map
                    .get(binding.resource.0)
                    .and_then(|mapped| *mapped)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "double-double real child resource mapping is incomplete",
                    ))?;
            }
            passes.push(pass);
        }
        let mut last_bindings = vec![
            binding(0, 3, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
        ];
        if ir.kind == crate::RealFftKind::RealToComplex
            && let Some(roots) = even_roots_resource
        {
            last_bindings.push(ProgramPassBinding {
                binding: 2,
                resource: roots,
                role: BufferRole::LookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        passes.push(ProgramPass {
            name: match (ir.kind, ir.even_half_size) {
                (crate::RealFftKind::RealToComplex, true) => {
                    "vkfft_dd_real_r2c_half_size_postprocess".to_owned()
                }
                (crate::RealFftKind::ComplexToReal, true) => {
                    "vkfft_dd_real_c2r_half_size_unpack".to_owned()
                }
                (crate::RealFftKind::RealToComplex, false) => {
                    "vkfft_dd_real_r2c_compact".to_owned()
                }
                (crate::RealFftKind::ComplexToReal, false) => {
                    "vkfft_dd_real_c2r_finalize".to_owned()
                }
            },
            dispatch: boundary_dispatch,
            bindings: last_bindings,
        });
        let program = Self {
            name: format!(
                "vkfft_dd_real_{}_{}",
                match ir.kind {
                    crate::RealFftKind::RealToComplex => "r2c",
                    crate::RealFftKind::ComplexToReal => "c2r",
                },
                ir.length
            ),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_nd_real(ir: &DoubleDoubleNdRealFftIr) -> Result<Self> {
        ir.validate()?;
        let full_elements = ir.full_tensor_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double ND real full element count",
            },
        )?;
        let compact_elements = ir.compact_tensor_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double ND real compact element count",
            },
        )?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND real ProgramIr requires DD or F64 external storage",
                ));
            }
        };
        let formatted_io = ir.formatted_io.as_ref();
        let input_elements = formatted_io
            .input_external_layout
            .batch_stride
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double ND real physical input element count",
            })?;
        let output_elements = formatted_io
            .output_external_layout
            .batch_stride
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double ND real physical output element count",
            })?;
        let input_layout = ExternalBufferLayout {
            logical_len: if formatted_io.input_formatted_copy.is_some() {
                formatted_io.input_external_layout.batch_stride
            } else {
                ir.input_tensor_len()
            },
            physical_stride: formatted_io.input_external_layout.batch_stride,
            batch_count: ir.batch_count,
            element_shape: match ir.kind {
                crate::RealFftKind::RealToComplex => ProgramElementShape::Scalar,
                crate::RealFftKind::ComplexToReal => ProgramElementShape::Complex,
            },
        };
        let output_layout = ExternalBufferLayout {
            logical_len: if formatted_io.output_formatted_copy.is_some() {
                formatted_io.output_external_layout.batch_stride
            } else {
                ir.output_tensor_len()
            },
            physical_stride: formatted_io.output_external_layout.batch_stride,
            batch_count: ir.batch_count,
            element_shape: match ir.kind {
                crate::RealFftKind::RealToComplex => ProgramElementShape::Complex,
                crate::RealFftKind::ComplexToReal => ProgramElementShape::Scalar,
            },
        };
        let full_scalar_layout = ExternalBufferLayout {
            logical_len: full_elements,
            physical_stride: full_elements,
            batch_count: 1,
            element_shape: ProgramElementShape::Scalar,
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements: input_elements,
                external_layout: Some(input_layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements: output_elements,
                external_layout: Some(output_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_nd_real_full_scalar".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: full_elements,
                external_layout: Some(full_scalar_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "double_double_nd_real_compact_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: compact_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(4),
                name: "double_double_nd_real_compact_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: compact_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(5),
                name: "double_double_nd_real_lines_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: compact_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(6),
                name: "double_double_nd_real_lines_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: compact_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let boundary_dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double ND real boundary dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        let mut passes = Vec::new();
        let input_internal = match ir.kind {
            crate::RealFftKind::RealToComplex => ProgramResourceId(2),
            crate::RealFftKind::ComplexToReal => ProgramResourceId(3),
        };
        if let Some(copy) = &formatted_io.input_formatted_copy {
            passes.push(ProgramPass {
                name: copy.name.clone(),
                dispatch: copy.dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: input_internal,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
        } else {
            passes.push(ProgramPass {
                name: "vkfft_dd_nd_real_external_input_promote".to_owned(),
                dispatch: boundary_dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: input_internal,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
        }

        let append_child = |child: &ProgramIr,
                            prefix: &str,
                            input: ProgramResourceId,
                            output: ProgramResourceId,
                            resources: &mut Vec<ProgramResource>,
                            passes: &mut Vec<ProgramPass>|
         -> Result<()> {
            let mut resource_map = vec![None; child.resources.len()];
            for resource in &child.resources {
                let mapped = match resource.kind {
                    ProgramResourceKind::Input => input,
                    ProgramResourceKind::Output => output,
                    _ => {
                        let new_id = ProgramResourceId(resources.len());
                        let mut cloned = resource.clone();
                        cloned.id = new_id;
                        cloned.name = format!("{prefix}_{}", resource.name);
                        cloned.external_layout = None;
                        resources.push(cloned);
                        new_id
                    }
                };
                resource_map[resource.id.0] = Some(mapped);
            }
            for child_pass in &child.passes {
                let mut pass = child_pass.clone();
                pass.name = format!("{prefix}_{}", child_pass.name);
                for binding in &mut pass.bindings {
                    binding.resource = resource_map
                        .get(binding.resource.0)
                        .and_then(|mapped| *mapped)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "double-double ND real child resource mapping is incomplete",
                        ))?;
                }
                passes.push(pass);
            }
            Ok(())
        };

        let real_child = Self::double_double_real(&ir.real_axis)?;
        let expected_real_input = match ir.kind {
            crate::RealFftKind::RealToComplex => full_elements,
            crate::RealFftKind::ComplexToReal => compact_elements,
        };
        let expected_real_output = match ir.kind {
            crate::RealFftKind::RealToComplex => compact_elements,
            crate::RealFftKind::ComplexToReal => full_elements,
        };
        if real_child.input_resource()?.scalar != ScalarType::DoubleDouble
            || real_child.output_resource()?.scalar != ScalarType::DoubleDouble
            || real_child.input_resource()?.elements != expected_real_input
            || real_child.output_resource()?.elements != expected_real_output
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND real last-axis child has incompatible boundaries",
            ));
        }

        let mut compact_source = ProgramResourceId(3);
        let mut compact_target = ProgramResourceId(4);
        if ir.kind == crate::RealFftKind::RealToComplex {
            append_child(
                &real_child,
                "vkfft_dd_nd_real_last_axis",
                ProgramResourceId(2),
                compact_source,
                &mut resources,
                &mut passes,
            )?;
        }

        for axis in &ir.complex_axes {
            let dispatch = DispatchGeometry {
                x: u32::try_from(ir.batch_count.div_ceil(axis.grouped_batch)).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "double-double ND real complex-axis dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            };
            passes.push(ProgramPass {
                name: format!("vkfft_dd_nd_real_axis_{}_pack", axis.axis),
                dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: compact_source,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 5, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });
            let child = Self::double_double_one_dim(&axis.transform)?;
            if child.input_resource()?.scalar != ScalarType::DoubleDouble
                || child.output_resource()?.scalar != ScalarType::DoubleDouble
                || child.input_resource()?.elements != compact_elements
                || child.output_resource()?.elements != compact_elements
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND real complex-axis child has incompatible boundaries",
                ));
            }
            append_child(
                &child,
                &format!("vkfft_dd_nd_real_axis_{}", axis.axis),
                ProgramResourceId(5),
                ProgramResourceId(6),
                &mut resources,
                &mut passes,
            )?;
            passes.push(ProgramPass {
                name: format!("vkfft_dd_nd_real_axis_{}_scatter", axis.axis),
                dispatch,
                bindings: vec![
                    binding(0, 6, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: compact_target,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
            compact_source = compact_target;
            compact_target = if compact_target == ProgramResourceId(3) {
                ProgramResourceId(4)
            } else {
                ProgramResourceId(3)
            };
        }

        if ir.kind == crate::RealFftKind::ComplexToReal {
            append_child(
                &real_child,
                "vkfft_dd_nd_real_last_axis",
                compact_source,
                ProgramResourceId(2),
                &mut resources,
                &mut passes,
            )?;
        }
        let output_internal = match ir.kind {
            crate::RealFftKind::RealToComplex => compact_source,
            crate::RealFftKind::ComplexToReal => ProgramResourceId(2),
        };
        if let Some(copy) = &formatted_io.output_formatted_copy {
            passes.push(ProgramPass {
                name: copy.name.clone(),
                dispatch: copy.dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: output_internal,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });
        } else {
            passes.push(ProgramPass {
                name: "vkfft_dd_nd_real_external_output_finalize".to_owned(),
                dispatch: boundary_dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: output_internal,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });
        }
        let program = Self {
            name: format!("vkfft_dd_nd_real_{:?}_{:?}", ir.dimensions, ir.kind),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_nd(ir: &DoubleDoubleNdFftIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.tensor_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double ND program element count",
                })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND program requires DD or F64 external storage",
                ));
            }
        };
        let input_layout = ExternalBufferLayout {
            logical_len: if ir.input_formatted_copy.is_some() {
                ir.input_external_layout.batch_stride
            } else {
                ir.tensor_len
            },
            physical_stride: ir.input_external_layout.batch_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let output_layout = ExternalBufferLayout {
            logical_len: if ir.output_formatted_copy.is_some() {
                ir.output_external_layout.batch_stride
            } else {
                ir.tensor_len
            },
            physical_stride: ir.output_external_layout.batch_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let input_elements = input_layout.physical_elements()?;
        let output_elements = output_layout.physical_elements()?;
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements: input_elements,
                external_layout: Some(input_layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements: output_elements,
                external_layout: Some(output_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_nd_tensor_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "double_double_nd_tensor_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(4),
                name: "double_double_nd_lines_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(5),
                name: "double_double_nd_lines_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let mut passes = Vec::new();
        let mut tensor_source = if let Some(copy) = &ir.input_formatted_copy {
            passes.push(two_buffer_pass(&copy.name, copy.dispatch, 0, 2));
            ProgramResourceId(2)
        } else {
            ProgramResourceId(0)
        };
        let mut tensor_target = if tensor_source == ProgramResourceId(2) {
            ProgramResourceId(3)
        } else {
            ProgramResourceId(2)
        };
        let mut formatted_output_source = None;
        for (axis_index, axis) in ir.axes.iter().enumerate() {
            let dispatch = DispatchGeometry {
                x: u32::try_from(ir.batch_count.div_ceil(axis.grouped_batch)).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "double-double ND grouped dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            };
            passes.push(ProgramPass {
                name: format!("vkfft_dd_nd_axis_{}_pack", axis.axis),
                dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: tensor_source,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 4, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });

            let child = Self::double_double_one_dim(&axis.transform)?;
            if child.scalar != ScalarType::DoubleDouble
                || child.input_resource()?.scalar != ScalarType::DoubleDouble
                || child.output_resource()?.scalar != ScalarType::DoubleDouble
                || child.input_resource()?.elements != elements
                || child.output_resource()?.elements != elements
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND child ProgramIr must expose DD tensor-sized boundaries",
                ));
            }
            let mut resource_map = vec![None; child.resources.len()];
            for resource in &child.resources {
                let mapped = match resource.kind {
                    ProgramResourceKind::Input => ProgramResourceId(4),
                    ProgramResourceKind::Output => ProgramResourceId(5),
                    _ => {
                        let new_id = ProgramResourceId(resources.len());
                        let mut cloned = resource.clone();
                        cloned.id = new_id;
                        cloned.name =
                            format!("double_double_nd_axis_{}_{}", axis.axis, resource.name);
                        cloned.external_layout = None;
                        resources.push(cloned);
                        new_id
                    }
                };
                resource_map[resource.id.0] = Some(mapped);
            }
            for child_pass in &child.passes {
                let mut pass = child_pass.clone();
                pass.name = format!("vkfft_dd_nd_axis_{}_{}", axis.axis, child_pass.name);
                for binding in &mut pass.bindings {
                    binding.resource = resource_map
                        .get(binding.resource.0)
                        .and_then(|mapped| *mapped)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "double-double ND child resource mapping is incomplete",
                        ))?;
                }
                passes.push(pass);
            }

            let last_axis = axis_index + 1 == ir.axes.len();
            let scatter_target = if last_axis && ir.output_formatted_copy.is_none() {
                ProgramResourceId(1)
            } else {
                tensor_target
            };
            passes.push(ProgramPass {
                name: format!("vkfft_dd_nd_axis_{}_scatter", axis.axis),
                dispatch,
                bindings: vec![
                    binding(0, 5, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: scatter_target,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
            if last_axis {
                formatted_output_source = ir.output_formatted_copy.as_ref().map(|_| scatter_target);
            } else {
                tensor_source = tensor_target;
                tensor_target = if tensor_target == ProgramResourceId(2) {
                    ProgramResourceId(3)
                } else {
                    ProgramResourceId(2)
                };
            }
        }
        if let (Some(copy), Some(source)) = (&ir.output_formatted_copy, formatted_output_source) {
            passes.push(ProgramPass {
                name: copy.name.clone(),
                dispatch: copy.dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: source,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });
        }
        let program = Self {
            name: format!("vkfft_dd_nd_program_{}d", ir.dimensions.len()),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_nd_r2r(ir: &DoubleDoubleNdR2rIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.tensor_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double ND R2R program element count",
                })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R program requires DD or F64 external storage",
                ));
            }
        };
        let input_external_stride = if ir.input_formatted_copy.is_some() {
            ir.input_external_layout.batch_stride
        } else {
            ir.tensor_len
        };
        let output_external_stride = if ir.output_formatted_copy.is_some() {
            ir.output_external_layout.batch_stride
        } else {
            ir.tensor_len
        };
        let input_external_layout = ExternalBufferLayout {
            logical_len: input_external_stride,
            physical_stride: input_external_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Scalar,
        };
        let output_external_layout = ExternalBufferLayout {
            logical_len: output_external_stride,
            physical_stride: output_external_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Scalar,
        };
        let input_external_elements = input_external_layout.physical_elements()?;
        let output_external_elements = output_external_layout.physical_elements()?;
        let scalar_scratch_layout = ExternalBufferLayout {
            logical_len: elements,
            physical_stride: elements,
            batch_count: 1,
            element_shape: ProgramElementShape::Scalar,
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements: input_external_elements,
                external_layout: Some(input_external_layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements: output_external_elements,
                external_layout: Some(output_external_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_nd_r2r_tensor_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: Some(scalar_scratch_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "double_double_nd_r2r_tensor_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: Some(scalar_scratch_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(4),
                name: "double_double_nd_r2r_lines_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: Some(scalar_scratch_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(5),
                name: "double_double_nd_r2r_lines_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: Some(scalar_scratch_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let mut passes = Vec::new();
        let mut tensor_source = if let Some(copy) = &ir.input_formatted_copy {
            let dense = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: dense,
                name: "double_double_nd_r2r_formatted_input".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: Some(scalar_scratch_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            });
            passes.push(ProgramPass {
                name: copy.name.clone(),
                dispatch: copy.dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: dense,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
            dense
        } else {
            ProgramResourceId(0)
        };
        let formatted_output = if ir.output_formatted_copy.is_some() {
            let dense = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: dense,
                name: "double_double_nd_r2r_formatted_output".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements,
                external_layout: Some(scalar_scratch_layout),
                initialization: ProgramResourceInitialization::Zeroed,
            });
            Some(dense)
        } else {
            None
        };
        let mut tensor_target = ProgramResourceId(2);
        for (axis_index, axis) in ir.axes.iter().enumerate() {
            let dispatch = DispatchGeometry {
                x: u32::try_from(ir.batch_count.div_ceil(axis.grouped_batch)).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "double-double ND R2R grouped dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            };
            passes.push(ProgramPass {
                name: format!("vkfft_dd_nd_r2r_axis_{}_pack", axis.axis),
                dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: tensor_source,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 4, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });
            let child = Self::double_double_r2r(&axis.transform)?;
            if child.scalar != ScalarType::DoubleDouble
                || child.input_resource()?.scalar != ScalarType::DoubleDouble
                || child.output_resource()?.scalar != ScalarType::DoubleDouble
                || child.input_resource()?.element_shape() != ProgramElementShape::Scalar
                || child.output_resource()?.element_shape() != ProgramElementShape::Scalar
                || child.input_resource()?.elements != elements
                || child.output_resource()?.elements != elements
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R child ProgramIr must expose tensor-sized scalar DD boundaries",
                ));
            }
            let mut resource_map = vec![None; child.resources.len()];
            for resource in &child.resources {
                let mapped = match resource.kind {
                    ProgramResourceKind::Input => ProgramResourceId(4),
                    ProgramResourceKind::Output => ProgramResourceId(5),
                    _ => {
                        let new_id = ProgramResourceId(resources.len());
                        let mut cloned = resource.clone();
                        cloned.id = new_id;
                        cloned.name =
                            format!("double_double_nd_r2r_axis_{}_{}", axis.axis, resource.name);
                        cloned.external_layout = None;
                        resources.push(cloned);
                        new_id
                    }
                };
                resource_map[resource.id.0] = Some(mapped);
            }
            for child_pass in &child.passes {
                let mut pass = child_pass.clone();
                pass.name = format!("vkfft_dd_nd_r2r_axis_{}_{}", axis.axis, child_pass.name);
                for binding in &mut pass.bindings {
                    binding.resource = resource_map
                        .get(binding.resource.0)
                        .and_then(|mapped| *mapped)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "double-double ND R2R child resource mapping is incomplete",
                        ))?;
                }
                passes.push(pass);
            }
            let last_axis = axis_index + 1 == ir.axes.len();
            let scatter_target = if last_axis {
                formatted_output.unwrap_or(ProgramResourceId(1))
            } else {
                tensor_target
            };
            passes.push(ProgramPass {
                name: format!("vkfft_dd_nd_r2r_axis_{}_scatter", axis.axis),
                dispatch,
                bindings: vec![
                    binding(0, 5, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: scatter_target,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ],
            });
            if !last_axis {
                tensor_source = tensor_target;
                tensor_target = if tensor_target == ProgramResourceId(2) {
                    ProgramResourceId(3)
                } else {
                    ProgramResourceId(2)
                };
            }
        }
        if let Some(copy) = &ir.output_formatted_copy {
            passes.push(ProgramPass {
                name: copy.name.clone(),
                dispatch: copy.dispatch,
                bindings: vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: formatted_output.expect("formatted DD R2R output scratch"),
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                ],
            });
        }
        let program = Self {
            name: format!("vkfft_dd_nd_r2r_program_{}d", ir.dimensions.len()),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_bluestein(ir: &DoubleDoubleBluesteinIr) -> Result<Self> {
        ir.validate()?;
        let logical_elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Bluestein program logical element count",
                })?;
        let convolution_elements = ir.convolution_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double Bluestein program convolution element count",
            },
        )?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Bluestein program requires DD or F64 external storage",
                ));
            }
        };
        let fused_inverse_stockham = matches!(
            &ir.inverse_fft,
            crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child)
                if double_double_stockham_input_lookup_eligible(child)
        );
        let layout = ExternalBufferLayout {
            logical_len: ir.logical_len,
            physical_stride: ir.logical_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements: logical_elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements: logical_elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let push_convolution_scratch =
            |resources: &mut Vec<ProgramResource>, name: &str| -> ProgramResourceId {
                let id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id,
                    name: name.to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ScalarType::DoubleDouble,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                });
                id
            };
        let forward_input_id =
            push_convolution_scratch(&mut resources, "double_double_bluestein_forward_input");
        let forward_output_id =
            push_convolution_scratch(&mut resources, "double_double_bluestein_forward_output");
        let inverse_input_id = if fused_inverse_stockham {
            None
        } else {
            Some(push_convolution_scratch(
                &mut resources,
                "double_double_bluestein_inverse_input",
            ))
        };
        let inverse_output_id =
            push_convolution_scratch(&mut resources, "double_double_bluestein_inverse_output");
        let chirp_id = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: chirp_id,
            name: "double_double_bluestein_chirp".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: ir.table.chirp.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                ir.table.chirp.clone(),
            ),
        });
        let kernel_id = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: kernel_id,
            name: "double_double_bluestein_kernel_spectrum".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: ir.kernel_spectrum.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                ir.kernel_spectrum.clone(),
            ),
        });
        let dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double Bluestein program dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        let mut passes = Vec::new();
        passes.push(ProgramPass {
            name: format!("{}_preprocess", ir.name),
            dispatch,
            bindings: vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(
                    1,
                    forward_input_id.0,
                    BufferRole::Output,
                    BufferAccess::WriteOnly,
                ),
                ProgramPassBinding {
                    binding: 2,
                    resource: chirp_id,
                    role: BufferRole::LookupTable,
                    access: BufferAccess::ReadOnly,
                },
            ],
        });

        let forward_child = match &ir.forward_fft {
            crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child) => {
                ProgramIr::double_double_stockham_internal(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) => {
                ProgramIr::double_double_recursive(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) => {
                ProgramIr::double_double_bluestein(child)?
            }
        };
        append_double_double_recursive_child(
            &forward_child,
            "vkfft_dd_bluestein_forward",
            forward_input_id,
            forward_output_id,
            &mut resources,
            &mut passes,
            false,
        )?;

        if !fused_inverse_stockham {
            passes.push(ProgramPass {
                name: format!("{}_multiply", ir.name),
                dispatch,
                bindings: vec![
                    binding(
                        0,
                        forward_output_id.0,
                        BufferRole::Input,
                        BufferAccess::ReadOnly,
                    ),
                    binding(
                        1,
                        inverse_input_id
                            .expect("explicit DD Bluestein multiply requires inverse input scratch")
                            .0,
                        BufferRole::Output,
                        BufferAccess::WriteOnly,
                    ),
                    ProgramPassBinding {
                        binding: 2,
                        resource: kernel_id,
                        role: BufferRole::LookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            });
        }

        let inverse_child = match &ir.inverse_fft {
            crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child) => {
                ProgramIr::double_double_stockham_internal(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) => {
                ProgramIr::double_double_recursive(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) => {
                ProgramIr::double_double_bluestein(child)?
            }
        };
        let inverse_pass_start = passes.len();
        append_double_double_recursive_child(
            &inverse_child,
            "vkfft_dd_bluestein_inverse",
            if fused_inverse_stockham {
                forward_output_id
            } else {
                inverse_input_id.expect("explicit DD Bluestein inverse requires input scratch")
            },
            inverse_output_id,
            &mut resources,
            &mut passes,
            false,
        )?;
        if fused_inverse_stockham {
            attach_double_double_stockham_input_lookup(
                passes
                    .get_mut(inverse_pass_start)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "fused DD Bluestein inverse Stockham child emitted no pass",
                    ))?,
                kernel_id,
            )?;
        }

        passes.push(ProgramPass {
            name: format!("{}_postprocess", ir.name),
            dispatch,
            bindings: vec![
                binding(
                    0,
                    inverse_output_id.0,
                    BufferRole::Input,
                    BufferAccess::ReadOnly,
                ),
                binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                ProgramPassBinding {
                    binding: 2,
                    resource: chirp_id,
                    role: BufferRole::LookupTable,
                    access: BufferAccess::ReadOnly,
                },
            ],
        });
        let program = Self {
            name: format!("{}_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn double_double_fft_rader(ir: &DoubleDoubleFftRaderIr) -> Result<Self> {
        ir.validate()?;
        let external_elements =
            ir.prime
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double FFT Rader program external element count",
                })?;
        let scratch_elements = ir.convolution_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double FFT Rader program scratch element count",
            },
        )?;
        let storage_scalar = |storage| match storage {
            crate::PrecisionStorage::DoubleDouble => Ok(ScalarType::DoubleDouble),
            crate::PrecisionStorage::F64 => Ok(ScalarType::F64),
            _ => Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader program requires DD or F64 caller storage",
            )),
        };
        let input_scalar = storage_scalar(ir.input_storage)?;
        let output_scalar = storage_scalar(ir.output_storage)?;
        let layout = ExternalBufferLayout {
            logical_len: ir.prime,
            physical_stride: ir.prime,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: input_scalar,
                elements: external_elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: output_scalar,
                elements: external_elements,
                external_layout: Some(layout),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "double_double_rader_scratch_a".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: scratch_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "double_double_rader_scratch_b".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ScalarType::DoubleDouble,
                elements: scratch_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let kernel_id = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: kernel_id,
            name: "double_double_rader_kernel_spectrum".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: ir.kernel_spectrum.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                ir.kernel_spectrum.clone(),
            ),
        });
        let dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double FFT Rader program dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        let convolution_grouped_batch = ir
            .forward_fft
            .stockham_axis_batch_block()
            .map_or(ir.grouped_batch, |block| block.grouped_batch);
        let convolution_dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_count.div_ceil(convolution_grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "double-double FFT Rader convolution dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        let mut passes = Vec::new();
        passes.push(ProgramPass {
            name: format!("{}_gather", ir.name),
            dispatch,
            bindings: vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 2, BufferRole::Output, BufferAccess::WriteOnly),
            ],
        });

        let forward_child = match &ir.forward_fft {
            crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child) => {
                ProgramIr::double_double_stockham_internal(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) => {
                ProgramIr::double_double_recursive(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) => {
                ProgramIr::double_double_bluestein(child)?
            }
        };
        append_double_double_recursive_child(
            &forward_child,
            "vkfft_dd_fft_rader_forward",
            ProgramResourceId(2),
            ProgramResourceId(3),
            &mut resources,
            &mut passes,
            false,
        )?;
        let fused_inverse_stockham = matches!(
            &ir.inverse_fft,
            crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child)
                if double_double_stockham_input_lookup_eligible(child)
        );
        if !fused_inverse_stockham {
            passes.push(ProgramPass {
                name: format!("{}_multiply", ir.name),
                dispatch: convolution_dispatch,
                bindings: vec![
                    binding(0, 3, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 2, BufferRole::Output, BufferAccess::WriteOnly),
                    ProgramPassBinding {
                        binding: 2,
                        resource: kernel_id,
                        role: BufferRole::LookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            });
        }
        let inverse_child = match &ir.inverse_fft {
            crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child) => {
                ProgramIr::double_double_stockham_internal(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) => {
                ProgramIr::double_double_recursive(child)?
            }
            crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) => {
                ProgramIr::double_double_bluestein(child)?
            }
        };
        let inverse_pass_start = passes.len();
        append_double_double_recursive_child(
            &inverse_child,
            "vkfft_dd_fft_rader_inverse",
            if fused_inverse_stockham {
                ProgramResourceId(3)
            } else {
                ProgramResourceId(2)
            },
            if fused_inverse_stockham {
                ProgramResourceId(2)
            } else {
                ProgramResourceId(3)
            },
            &mut resources,
            &mut passes,
            false,
        )?;
        if fused_inverse_stockham {
            attach_double_double_stockham_input_lookup(
                passes
                    .get_mut(inverse_pass_start)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "fused DD FFT-Rader inverse Stockham child emitted no pass",
                    ))?,
                kernel_id,
            )?;
        }
        let inverse_output = if fused_inverse_stockham { 2 } else { 3 };
        let mut scatter_bindings = vec![
            binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
            binding(
                2,
                inverse_output,
                BufferRole::Auxiliary,
                BufferAccess::ReadOnly,
            ),
        ];
        let parent_period = match ir.io_mapping {
            crate::StockhamIoMapping::FourStepRight(mapping) => Some(mapping.logical_len),
            crate::StockhamIoMapping::FourStepThreeUpload2(mapping) => Some(mapping.logical_len),
            crate::StockhamIoMapping::FourStepThreeUpload1(mapping) => {
                let [a, b, _] = mapping.axis_split;
                Some(a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double FFT Rader three-upload parent root period",
                })?)
            }
            _ => None,
        };
        if let Some(period) = parent_period {
            let roots = crate::lut::unit_root_table_double_double(period, ir.direction)?;
            let root_id = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: root_id,
                name: "double_double_fft_rader_four_step_roots".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: roots.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(roots),
            });
            scatter_bindings.push(ProgramPassBinding {
                binding: 3,
                resource: root_id,
                role: BufferRole::TwiddleLookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        passes.push(ProgramPass {
            name: format!("{}_scatter", ir.name),
            dispatch,
            bindings: scatter_bindings,
        });
        let program = Self {
            name: format!("{}_program", ir.name),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Self::wrap_double_double_zero_pad(
            ir.zero_pad_pass.as_ref(),
            ir.prime,
            ir.batch_count,
            "double_double_fft_rader",
            program,
        )
    }

    pub fn stockham(kernel: &KernelIr) -> Result<Self> {
        kernel.validate()?;
        let elements = kernel.sequence_len.checked_mul(kernel.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "Stockham program element count",
            },
        )?;
        let layout = ExternalBufferLayout {
            logical_len: kernel.sequence_len,
            physical_stride: kernel.sequence_len,
            batch_count: kernel.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: kernel.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: kernel.bindings[0].scalar,
                    elements,
                    external_layout: Some(layout),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: kernel.bindings[1].scalar,
                    elements,
                    external_layout: Some(layout),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        builder.push_stockham_pass(
            kernel,
            ProgramResourceId(0),
            ProgramResourceId(1),
            None,
            None,
        )?;
        let program = Self {
            name: format!("{}_program", kernel.name),
            scalar: kernel.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    fn double_double_r2r_fused_physical_stockham(
        ir: &DoubleDoubleR2rIr,
        elements: usize,
        external_scalar: ScalarType,
        external: ExternalBufferLayout,
    ) -> Result<Option<Self>> {
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, phases, .. } = &ir.algorithm else {
            return Ok(None);
        };
        if !matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::III)
        ) {
            return Ok(None);
        }
        let DoubleDoubleOneDimIr::Stockham(stockham) = fft.as_ref() else {
            return Ok(None);
        };
        if stockham.axis_batch_block.is_none()
            || stockham.external_storage != crate::PrecisionStorage::DoubleDouble
            || stockham.zero_pad_pass.is_some()
            || stockham.normalize
            || stockham.sequence_len != ir.length
            || phases.len() != ir.length
        {
            return Ok(None);
        }

        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements,
                external_layout: Some(external),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements,
                external_layout: Some(external),
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let mut bindings = vec![
            binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
        ];
        let twiddles = stockham.twiddles.packed_values();
        if !twiddles.is_empty() {
            let twiddle_id = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: twiddle_id,
                name: "double_double_r2r_fused_stockham_twiddles".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: twiddles.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(twiddles),
            });
            bindings.push(ProgramPassBinding {
                binding: 2,
                resource: twiddle_id,
                role: BufferRole::TwiddleLookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        let phases_id = ProgramResourceId(resources.len());
        resources.push(ProgramResource {
            id: phases_id,
            name: "double_double_r2r_phases".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: phases.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases.clone()),
        });
        bindings.push(ProgramPassBinding {
            binding: 3,
            resource: phases_id,
            role: BufferRole::LookupTable,
            access: BufferAccess::ReadOnly,
        });
        let label = match ir.effective_transform {
            crate::R2rTransform::Dct(crate::DctType::II) => "dct2",
            crate::R2rTransform::Dct(crate::DctType::III) => "dct3",
            crate::R2rTransform::Dst(crate::DstType::II) => "dst2",
            crate::R2rTransform::Dst(crate::DstType::III) => "dst3",
            _ => unreachable!("validated fused DD Type-II/III transform"),
        };
        let program = Self {
            name: format!("vkfft_dd_r2r_{label}_stockham_fused_{}_program", ir.length),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes: vec![ProgramPass {
                name: format!("vkfft_dd_r2r_{label}_stockham_fused_{}", ir.length),
                dispatch: DispatchGeometry {
                    x: u32::try_from(stockham.batch_group_count()).map_err(|_| {
                        VkFftError::ValueOutOfRange {
                            field: "fused double-double R2R Stockham dispatch count",
                        }
                    })?,
                    y: 1,
                    z: 1,
                },
                bindings,
            }],
        };
        program.validate()?;
        Ok(Some(program))
    }

    fn double_double_r2r_fused_two_upload_four_step(
        ir: &DoubleDoubleR2rIr,
        elements: usize,
        external_scalar: ScalarType,
        external: ExternalBufferLayout,
    ) -> Result<Option<Self>> {
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, phases, .. } = &ir.algorithm else {
            return Ok(None);
        };
        if !matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::III)
        ) {
            return Ok(None);
        }
        let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
            return Ok(None);
        };
        if recursive.two_upload_four_step_plan.is_none()
            || recursive.three_upload_four_step_plan.is_some()
            || recursive.rader_forced_upload_schedule.is_some()
            || recursive.external_storage != crate::PrecisionStorage::DoubleDouble
            || recursive.zero_pad_pass.is_some()
            || recursive.normalize
            || recursive.logical_len != ir.length
            || recursive.batch_count != ir.batch_count
            || recursive.grouped_batch != ir.grouped_batch
            || phases.len() != ir.length
        {
            return Ok(None);
        }

        let mut program = Self::double_double_recursive(recursive)?;
        if program.passes.len() != 2
            || program.input_resource()?.scalar != ScalarType::DoubleDouble
            || program.output_resource()?.scalar != ScalarType::DoubleDouble
        {
            return Ok(None);
        }
        for resource in &mut program.resources {
            match resource.kind {
                ProgramResourceKind::Input => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                ProgramResourceKind::Output => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                _ => {}
            }
        }
        let phases_id = ProgramResourceId(program.resources.len());
        program.resources.push(ProgramResource {
            id: phases_id,
            name: "double_double_r2r_phases".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: phases.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases.clone()),
        });
        let label = match ir.effective_transform {
            crate::R2rTransform::Dct(crate::DctType::II) => "dct2",
            crate::R2rTransform::Dct(crate::DctType::III) => "dct3",
            crate::R2rTransform::Dst(crate::DstType::II) => "dst2",
            crate::R2rTransform::Dst(crate::DstType::III) => "dst3",
            _ => unreachable!("validated two-upload DD Type-II/III fusion"),
        };
        let phase_binding = ProgramPassBinding {
            binding: if matches!(
                ir.effective_transform,
                crate::R2rTransform::Dct(crate::DctType::III)
                    | crate::R2rTransform::Dst(crate::DstType::III)
            ) {
                4
            } else {
                3
            },
            resource: phases_id,
            role: BufferRole::LookupTable,
            access: BufferAccess::ReadOnly,
        };
        if matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::III)
        ) {
            program.passes[0].bindings.push(phase_binding);
        } else {
            program.passes[1].bindings.push(phase_binding);
        }
        program.passes[0].name = format!("vkfft_dd_r2r_{label}_four_step_upload_1");
        program.passes[1].name = format!("vkfft_dd_r2r_{label}_four_step_upload_0");
        program.name = format!("vkfft_dd_r2r_{label}_four_step_fused_{}_program", ir.length);
        program.validate()?;
        Ok(Some(program))
    }

    fn double_double_r2r_fused_forced_rader_two_upload(
        ir: &DoubleDoubleR2rIr,
        elements: usize,
        external_scalar: ScalarType,
        external: ExternalBufferLayout,
    ) -> Result<Option<Self>> {
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, phases, .. } = &ir.algorithm else {
            return Ok(None);
        };
        if !matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::III)
        ) {
            return Ok(None);
        }
        let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
            return Ok(None);
        };
        if recursive.stockham_upload_schedule.is_some()
            || recursive.two_upload_four_step_plan.is_some()
            || recursive.three_upload_four_step_plan.is_some()
            || recursive.external_storage != crate::PrecisionStorage::DoubleDouble
            || recursive.zero_pad_pass.is_some()
            || recursive.normalize
            || recursive.logical_len != ir.length
            || recursive.batch_count != ir.batch_count
            || recursive.grouped_batch != ir.grouped_batch
            || phases.len() != ir.length
        {
            return Ok(None);
        }
        let Some(mapped_high) = recursive.forced_rader_two_upload_mapped_high_component()? else {
            return Ok(None);
        };
        let crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(high) = mapped_high else {
            return Ok(None);
        };
        if !matches!(
            high.pack_right.input_modifier,
            crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepRight(
                _
            )
        ) || high.pack_right.input_storage != crate::PrecisionStorage::DoubleDouble
            || high.pack_right.output_storage != crate::PrecisionStorage::DoubleDouble
        {
            return Ok(None);
        }
        let Some((low, mapping)) = recursive.forced_rader_two_upload_mapped_low_stockham()? else {
            return Ok(None);
        };
        if low.sequence_len > 64 || low.axis_batch_block.is_none() {
            return Ok(None);
        }
        let crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepRight(
            high_mapping,
        ) = high.pack_right.input_modifier
        else {
            unreachable!("validated forced-two recursive high mapping");
        };
        if high_mapping != mapping {
            return Ok(None);
        }

        let mut program = Self::double_double_recursive(recursive)?;
        let expected_last = format!("{}_four_step_left", low.name);
        if program.passes.len() < 2
            || program.input_resource()?.scalar != ScalarType::DoubleDouble
            || program.output_resource()?.scalar != ScalarType::DoubleDouble
            || program
                .passes
                .first()
                .is_none_or(|pass| pass.name != high.pack_right.name)
            || program
                .passes
                .last()
                .is_none_or(|pass| !pass.name.ends_with(&expected_last))
        {
            return Ok(None);
        }
        for resource in &mut program.resources {
            match resource.kind {
                ProgramResourceKind::Input => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                ProgramResourceKind::Output => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                _ => {}
            }
        }
        let phases_id = ProgramResourceId(program.resources.len());
        program.resources.push(ProgramResource {
            id: phases_id,
            name: "double_double_r2r_phases".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: phases.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases.clone()),
        });
        let label = match ir.effective_transform {
            crate::R2rTransform::Dct(crate::DctType::II) => "dct2",
            crate::R2rTransform::Dct(crate::DctType::III) => "dct3",
            crate::R2rTransform::Dst(crate::DstType::II) => "dst2",
            crate::R2rTransform::Dst(crate::DstType::III) => "dst3",
            _ => unreachable!("validated forced-two-upload DD Type-II/III fusion"),
        };
        let phase_binding = ProgramPassBinding {
            binding: if matches!(
                ir.effective_transform,
                crate::R2rTransform::Dct(crate::DctType::III)
                    | crate::R2rTransform::Dst(crate::DstType::III)
            ) {
                2
            } else {
                3
            },
            resource: phases_id,
            role: BufferRole::LookupTable,
            access: BufferAccess::ReadOnly,
        };
        let last_pass = program.passes.len() - 1;
        if matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::III)
        ) {
            program.passes[0].bindings.push(phase_binding);
        } else {
            program.passes[last_pass].bindings.push(phase_binding);
        }
        program.passes[0].name = format!("vkfft_dd_r2r_{label}_forced_rader_two_upload_1");
        program.passes[last_pass].name = format!("vkfft_dd_r2r_{label}_forced_rader_two_upload_0");
        program.name = format!(
            "vkfft_dd_r2r_{label}_forced_rader_two_upload_fused_{}_program",
            ir.length
        );
        program.validate()?;
        Ok(Some(program))
    }

    fn double_double_r2r_fused_forced_rader_two_upload_recursive_output(
        ir: &DoubleDoubleR2rIr,
        elements: usize,
        external_scalar: ScalarType,
        external: ExternalBufferLayout,
    ) -> Result<Option<Self>> {
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, phases, .. } = &ir.algorithm else {
            return Ok(None);
        };
        if !matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::III)
        ) {
            return Ok(None);
        }
        let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
            return Ok(None);
        };
        if recursive.stockham_upload_schedule.is_some()
            || recursive.two_upload_four_step_plan.is_some()
            || recursive.three_upload_four_step_plan.is_some()
            || recursive.external_storage != crate::PrecisionStorage::DoubleDouble
            || recursive.zero_pad_pass.is_some()
            || recursive.normalize
            || recursive.logical_len != ir.length
            || recursive.batch_count != ir.batch_count
            || recursive.grouped_batch != ir.grouped_batch
            || phases.len() != ir.length
        {
            return Ok(None);
        }
        let (mapping, first_pass_suffix, fft_rader_scatter_suffix) = if let Some((high, mapping)) =
            recursive.forced_rader_two_upload_mapped_high_stockham()?
        {
            if high.axis_batch_block.is_none() {
                return Ok(None);
            }
            (mapping, "forced_rader_two_upload_1".to_owned(), None)
        } else {
            let Some(mapped_high) = recursive.forced_rader_two_upload_mapped_high_component()?
            else {
                return Ok(None);
            };
            match mapped_high {
                crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(high) => {
                    let crate::StockhamIoMapping::FourStepRight(mapping) = high.io_mapping else {
                        return Ok(None);
                    };
                    if high.prime != mapping.right_len
                        || high.axis_batch_block.is_none()
                        || high.input_storage != crate::PrecisionStorage::DoubleDouble
                        || high.output_storage != crate::PrecisionStorage::DoubleDouble
                        || high.zero_pad_pass.is_some()
                    {
                        return Ok(None);
                    }
                    (mapping, high.name.clone(), None)
                }
                crate::DoubleDoubleRecursiveFftNodeIr::FftRader(high) => {
                    let crate::StockhamIoMapping::FourStepRight(mapping) = high.io_mapping else {
                        return Ok(None);
                    };
                    if high.prime != mapping.right_len
                        || high.caller_axis_batch_block.is_none()
                        || high.external_storage != crate::PrecisionStorage::DoubleDouble
                        || high.input_storage != crate::PrecisionStorage::DoubleDouble
                        || high.output_storage != crate::PrecisionStorage::DoubleDouble
                        || high.zero_pad_pass.is_some()
                    {
                        return Ok(None);
                    }
                    (
                        mapping,
                        format!("{}_gather", high.name),
                        Some(format!("{}_scatter", high.name)),
                    )
                }
                _ => return Ok(None),
            }
        };
        let Some(mapped_low) = recursive.forced_rader_two_upload_mapped_low_component()? else {
            return Ok(None);
        };
        let (low_pass_name, low_pass_exact, low_phase_binding) = match mapped_low {
            crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) => {
                if !matches!(
                    low.scatter_output.output_modifier,
                    crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                        if low_mapping == mapping
                ) || low.scatter_output.input_storage != crate::PrecisionStorage::DoubleDouble
                    || low.scatter_output.output_storage != crate::PrecisionStorage::DoubleDouble
                {
                    return Ok(None);
                }
                (low.scatter_output.name.clone(), true, 2u32)
            }
            crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(low) => {
                let crate::StockhamIoMapping::FourStepLeft(low_mapping) = low.io_mapping else {
                    return Ok(None);
                };
                if low_mapping != mapping
                    || low.prime != mapping.left_len
                    || low.axis_batch_block.is_none()
                    || low.input_storage != crate::PrecisionStorage::DoubleDouble
                    || low.output_storage != crate::PrecisionStorage::DoubleDouble
                    || low.zero_pad_pass.is_some()
                {
                    return Ok(None);
                }
                (low.name.clone(), false, 3u32)
            }
            crate::DoubleDoubleRecursiveFftNodeIr::FftRader(low) => {
                let crate::StockhamIoMapping::FourStepLeft(low_mapping) = low.io_mapping else {
                    return Ok(None);
                };
                if low_mapping != mapping
                    || low.prime != mapping.left_len
                    || low.caller_axis_batch_block.is_none()
                    || low.external_storage != crate::PrecisionStorage::DoubleDouble
                    || low.input_storage != crate::PrecisionStorage::DoubleDouble
                    || low.output_storage != crate::PrecisionStorage::DoubleDouble
                    || low.zero_pad_pass.is_some()
                {
                    return Ok(None);
                }
                (format!("{}_scatter", low.name), false, 3u32)
            }
            _ => return Ok(None),
        };

        let mut program = Self::double_double_recursive(recursive)?;
        if program.passes.len() < 2
            || program.input_resource()?.scalar != ScalarType::DoubleDouble
            || program.output_resource()?.scalar != ScalarType::DoubleDouble
            || !program
                .passes
                .first()
                .is_some_and(|pass| pass.name.ends_with(&first_pass_suffix))
            || !program.passes.last().is_some_and(|pass| {
                if low_pass_exact {
                    pass.name == low_pass_name
                } else {
                    pass.name.ends_with(&low_pass_name)
                }
            })
        {
            return Ok(None);
        }
        for resource in &mut program.resources {
            match resource.kind {
                ProgramResourceKind::Input => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                ProgramResourceKind::Output => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                _ => {}
            }
        }
        let phases_id = ProgramResourceId(program.resources.len());
        program.resources.push(ProgramResource {
            id: phases_id,
            name: "double_double_r2r_phases".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: phases.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases.clone()),
        });
        let label = match ir.effective_transform {
            crate::R2rTransform::Dct(crate::DctType::II) => "dct2",
            crate::R2rTransform::Dct(crate::DctType::III) => "dct3",
            crate::R2rTransform::Dst(crate::DstType::II) => "dst2",
            crate::R2rTransform::Dst(crate::DstType::III) => "dst3",
            _ => unreachable!("validated forced-two recursive-output DD Type-II/III fusion"),
        };
        let last_pass = program.passes.len() - 1;
        if matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::III)
        ) {
            if let Some(scatter_suffix) = fft_rader_scatter_suffix.as_deref() {
                program.passes[0].bindings.push(ProgramPassBinding {
                    binding: 2,
                    resource: phases_id,
                    role: BufferRole::LookupTable,
                    access: BufferAccess::ReadOnly,
                });
                let Some(scatter) = program
                    .passes
                    .iter_mut()
                    .take(last_pass)
                    .find(|pass| pass.name.ends_with(scatter_suffix))
                else {
                    return Ok(None);
                };
                scatter.bindings.push(ProgramPassBinding {
                    binding: 4,
                    resource: phases_id,
                    role: BufferRole::LookupTable,
                    access: BufferAccess::ReadOnly,
                });
            } else {
                program.passes[0].bindings.push(ProgramPassBinding {
                    binding: 4,
                    resource: phases_id,
                    role: BufferRole::LookupTable,
                    access: BufferAccess::ReadOnly,
                });
            }
        } else {
            program.passes[last_pass].bindings.push(ProgramPassBinding {
                binding: low_phase_binding,
                resource: phases_id,
                role: BufferRole::LookupTable,
                access: BufferAccess::ReadOnly,
            });
        }
        program.passes[0].name = format!("vkfft_dd_r2r_{label}_forced_rader_two_upload_1");
        program.passes[last_pass].name =
            format!("vkfft_dd_r2r_{label}_forced_rader_two_upload_0_recursive");
        program.name = format!(
            "vkfft_dd_r2r_{label}_forced_rader_two_upload_recursive_output_fused_{}_program",
            ir.length
        );
        program.validate()?;
        Ok(Some(program))
    }

    fn double_double_r2r_fused_forced_rader_three_upload(
        ir: &DoubleDoubleR2rIr,
        elements: usize,
        external_scalar: ScalarType,
        external: ExternalBufferLayout,
    ) -> Result<Option<Self>> {
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, phases, .. } = &ir.algorithm else {
            return Ok(None);
        };
        if !matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::III)
        ) {
            return Ok(None);
        }
        let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
            return Ok(None);
        };
        if recursive.stockham_upload_schedule.is_some()
            || recursive.two_upload_four_step_plan.is_some()
            || recursive.three_upload_four_step_plan.is_some()
            || recursive.external_storage != crate::PrecisionStorage::DoubleDouble
            || recursive.zero_pad_pass.is_some()
            || recursive.normalize
            || recursive.logical_len != ir.length
            || recursive.batch_count != ir.batch_count
            || recursive.grouped_batch != ir.grouped_batch
            || phases.len() != ir.length
        {
            return Ok(None);
        }
        let Some(components) = recursive.forced_rader_three_upload_mapped_components()? else {
            return Ok(None);
        };
        if components.len() != 3 {
            return Ok(None);
        }
        let recursive_u2_pack_name = match components.first() {
            Some(
                crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                    upload_id: 2,
                    ..
                },
            ) => None,
            Some(
                crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                    upload_id: 2,
                    ir: crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley),
                },
            ) if matches!(
                cooley.pack_right.input_modifier,
                crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepThreeUpload2(_)
            ) && cooley.pack_right.input_storage == crate::PrecisionStorage::DoubleDouble
                && cooley.pack_right.output_storage == crate::PrecisionStorage::DoubleDouble =>
            {
                Some(cooley.pack_right.name.clone())
            }
            _ => return Ok(None),
        };
        if !matches!(
            components.last(),
            Some(
                crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                    upload_id: 0,
                    ..
                }
            )
        ) {
            return Ok(None);
        }

        let mut program = Self::double_double_recursive(recursive)?;
        let first_pass_matches = match recursive_u2_pack_name.as_deref() {
            Some(name) => program.passes.first().is_some_and(|pass| pass.name == name),
            None => program
                .passes
                .first()
                .is_some_and(|pass| pass.name.ends_with("forced_rader_three_upload_2")),
        };
        if program.passes.len() < 3
            || program.input_resource()?.scalar != ScalarType::DoubleDouble
            || program.output_resource()?.scalar != ScalarType::DoubleDouble
            || !first_pass_matches
            || !program
                .passes
                .last()
                .is_some_and(|pass| pass.name.ends_with("forced_rader_three_upload_0"))
        {
            return Ok(None);
        }

        // The complex-DD forced-three-upload builder intentionally reuses the external
        // output resource as the u2->u1 exchange when caller storage is compute-sized.
        // R2R needs that resource to be a scalar caller boundary, so split the exchange
        // into a dedicated full-DD scratch before changing the external output ABI.
        let external_output = ProgramResourceId(1);
        let exchange0 = ProgramResourceId(program.resources.len());
        program.resources.push(ProgramResource {
            id: exchange0,
            name: "double_double_r2r_forced_three_upload_exchange_0".to_owned(),
            kind: ProgramResourceKind::Scratch,
            scalar: ScalarType::DoubleDouble,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        let last_pass = program.passes.len() - 1;
        for (index, pass) in program.passes.iter_mut().enumerate() {
            if index == last_pass {
                continue;
            }
            for binding in &mut pass.bindings {
                if binding.resource == external_output {
                    binding.resource = exchange0;
                }
            }
        }

        for resource in &mut program.resources {
            match resource.kind {
                ProgramResourceKind::Input => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                ProgramResourceKind::Output => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                _ => {}
            }
        }
        let phases_id = ProgramResourceId(program.resources.len());
        program.resources.push(ProgramResource {
            id: phases_id,
            name: "double_double_r2r_phases".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: phases.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases.clone()),
        });
        let label = match ir.effective_transform {
            crate::R2rTransform::Dct(crate::DctType::II) => "dct2",
            crate::R2rTransform::Dct(crate::DctType::III) => "dct3",
            crate::R2rTransform::Dst(crate::DstType::II) => "dst2",
            crate::R2rTransform::Dst(crate::DstType::III) => "dst3",
            _ => unreachable!("validated forced-three-upload DD Type-II/III fusion"),
        };
        let phase_binding = ProgramPassBinding {
            binding: if matches!(
                ir.effective_transform,
                crate::R2rTransform::Dct(crate::DctType::III)
                    | crate::R2rTransform::Dst(crate::DstType::III)
            ) {
                if recursive_u2_pack_name.is_some() {
                    2
                } else {
                    4
                }
            } else {
                3
            },
            resource: phases_id,
            role: BufferRole::LookupTable,
            access: BufferAccess::ReadOnly,
        };
        if matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::III)
        ) {
            program.passes[0].bindings.push(phase_binding);
        } else {
            program.passes[last_pass].bindings.push(phase_binding);
        }
        program.passes[0].name = format!("vkfft_dd_r2r_{label}_forced_rader_three_upload_2");
        program.passes[last_pass].name =
            format!("vkfft_dd_r2r_{label}_forced_rader_three_upload_0");
        program.name = format!(
            "vkfft_dd_r2r_{label}_forced_rader_three_upload_fused_{}_program",
            ir.length
        );
        program.validate()?;
        Ok(Some(program))
    }

    fn double_double_r2r_fused_bluestein(
        ir: &DoubleDoubleR2rIr,
        elements: usize,
        external_scalar: ScalarType,
        external: ExternalBufferLayout,
    ) -> Result<Option<Self>> {
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, phases, .. } = &ir.algorithm else {
            return Ok(None);
        };
        if !matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::III)
        ) {
            return Ok(None);
        }
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
            return Ok(None);
        };
        if bluestein.logical_len != ir.length
            || bluestein.batch_count != ir.batch_count
            || bluestein.grouped_batch != ir.grouped_batch
            || bluestein.external_storage != crate::PrecisionStorage::DoubleDouble
            || bluestein.zero_padding.is_some()
            || phases.len() != ir.length
        {
            return Ok(None);
        }

        let mut program = Self::double_double_bluestein(bluestein)?;
        if program.passes.len() < 2
            || program.input_resource()?.scalar != ScalarType::DoubleDouble
            || program.output_resource()?.scalar != ScalarType::DoubleDouble
            || !program
                .passes
                .first()
                .is_some_and(|pass| pass.name == format!("{}_preprocess", bluestein.name))
            || !program
                .passes
                .last()
                .is_some_and(|pass| pass.name == format!("{}_postprocess", bluestein.name))
        {
            return Ok(None);
        }
        for resource in &mut program.resources {
            match resource.kind {
                ProgramResourceKind::Input => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                ProgramResourceKind::Output => {
                    resource.scalar = external_scalar;
                    resource.elements = elements;
                    resource.external_layout = Some(external);
                }
                _ => {}
            }
        }
        let phases_id = ProgramResourceId(program.resources.len());
        program.resources.push(ProgramResource {
            id: phases_id,
            name: "double_double_r2r_phases".to_owned(),
            kind: ProgramResourceKind::LookupTable,
            scalar: ScalarType::DoubleDouble,
            elements: phases.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases.clone()),
        });
        let label = match ir.effective_transform {
            crate::R2rTransform::Dct(crate::DctType::II) => "dct2",
            crate::R2rTransform::Dct(crate::DctType::III) => "dct3",
            crate::R2rTransform::Dst(crate::DstType::II) => "dst2",
            crate::R2rTransform::Dst(crate::DstType::III) => "dst3",
            _ => unreachable!("validated DD Bluestein Type-II/III fusion"),
        };
        let phase_binding = ProgramPassBinding {
            binding: 3,
            resource: phases_id,
            role: BufferRole::LookupTable,
            access: BufferAccess::ReadOnly,
        };
        let last_pass = program.passes.len() - 1;
        if matches!(
            ir.effective_transform,
            crate::R2rTransform::Dct(crate::DctType::III)
                | crate::R2rTransform::Dst(crate::DstType::III)
        ) {
            program.passes[0].bindings.push(phase_binding);
        } else {
            program.passes[last_pass].bindings.push(phase_binding);
        }
        program.passes[0].name = format!("vkfft_dd_r2r_{label}_bluestein_preprocess");
        program.passes[last_pass].name = format!("vkfft_dd_r2r_{label}_bluestein_postprocess");
        program.name = format!("vkfft_dd_r2r_{label}_bluestein_fused_{}_program", ir.length);
        program.validate()?;
        Ok(Some(program))
    }

    pub fn double_double_r2r(ir: &DoubleDoubleR2rIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.length
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double DCT/DST program element count",
                })?;
        let external_scalar = match ir.external_storage {
            crate::PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
            crate::PrecisionStorage::F64 => ScalarType::F64,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double R2R ProgramIr requires DD or F64 external storage",
                ));
            }
        };
        let external = ExternalBufferLayout {
            logical_len: ir.length,
            physical_stride: ir.length,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Scalar,
        };
        if let Some(program) =
            Self::double_double_r2r_fused_bluestein(ir, elements, external_scalar, external)?
        {
            return Ok(program);
        }
        if let Some(program) = Self::double_double_r2r_fused_physical_stockham(
            ir,
            elements,
            external_scalar,
            external,
        )? {
            return Ok(program);
        }
        if let Some(program) = Self::double_double_r2r_fused_two_upload_four_step(
            ir,
            elements,
            external_scalar,
            external,
        )? {
            return Ok(program);
        }
        if let Some(program) = Self::double_double_r2r_fused_forced_rader_two_upload(
            ir,
            elements,
            external_scalar,
            external,
        )? {
            return Ok(program);
        }
        if let Some(program) =
            Self::double_double_r2r_fused_forced_rader_two_upload_recursive_output(
                ir,
                elements,
                external_scalar,
                external,
            )?
        {
            return Ok(program);
        }
        if let Some(program) = Self::double_double_r2r_fused_forced_rader_three_upload(
            ir,
            elements,
            external_scalar,
            external,
        )? {
            return Ok(program);
        }
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: external_scalar,
                elements,
                external_layout: Some(external),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: external_scalar,
                elements,
                external_layout: Some(external),
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let mut passes = Vec::new();
        let dispatch = DispatchGeometry {
            x: u32::try_from(ir.batch_group_count()).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double R2R boundary dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        match &ir.algorithm {
            DoubleDoubleR2rAlgorithm::Direct { coefficients } => {
                let coefficients_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: coefficients_id,
                    name: "double_double_r2r_coefficients".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: coefficients.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        coefficients
                            .iter()
                            .copied()
                            .map(|value| ComplexDoubleDouble::new(value, crate::DoubleDouble::ZERO))
                            .collect(),
                    ),
                });
                passes.push(ProgramPass {
                    name: format!("vkfft_dd_r2r_direct_{}", ir.length),
                    dispatch,
                    bindings: vec![
                        binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                        ProgramPassBinding {
                            binding: 2,
                            resource: coefficients_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ],
                });
            }
            DoubleDoubleR2rAlgorithm::FftReduction {
                fft_len,
                fft,
                phases,
            } => {
                let fft_elements =
                    fft_len
                        .checked_mul(ir.batch_count)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double R2R FFT program element count",
                        })?;
                let fft_input = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: fft_input,
                    name: "double_double_r2r_fft_input".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ScalarType::DoubleDouble,
                    elements: fft_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                });
                let fft_output = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: fft_output,
                    name: "double_double_r2r_fft_output".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ScalarType::DoubleDouble,
                    elements: fft_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                });
                let phases_id = if phases.is_empty() {
                    None
                } else {
                    let phases_id = ProgramResourceId(resources.len());
                    resources.push(ProgramResource {
                        id: phases_id,
                        name: "double_double_r2r_phases".to_owned(),
                        kind: ProgramResourceKind::LookupTable,
                        scalar: ScalarType::DoubleDouble,
                        elements: phases.len(),
                        external_layout: None,
                        initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                            phases.clone(),
                        ),
                    });
                    Some(phases_id)
                };
                let child = Self::double_double_one_dim(fft)?;
                if child.scalar != ScalarType::DoubleDouble
                    || child.input_resource()?.elements != fft_elements
                    || child.output_resource()?.elements != fft_elements
                    || child.input_resource()?.scalar != ScalarType::DoubleDouble
                    || child.output_resource()?.scalar != ScalarType::DoubleDouble
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double R2R FFT child must expose full-DD boundaries",
                    ));
                }
                let mut resource_map = vec![None; child.resources.len()];
                for resource in &child.resources {
                    let mapped = match resource.kind {
                        ProgramResourceKind::Input => fft_input,
                        ProgramResourceKind::Output => fft_output,
                        _ => {
                            let new_id = ProgramResourceId(resources.len());
                            let mut cloned = resource.clone();
                            cloned.id = new_id;
                            cloned.name = format!("double_double_r2r_{}", resource.name);
                            cloned.external_layout = None;
                            resources.push(cloned);
                            new_id
                        }
                    };
                    resource_map[resource.id.0] = Some(mapped);
                }
                let mut pre_bindings = vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    ProgramPassBinding {
                        binding: 1,
                        resource: fft_input,
                        role: BufferRole::Output,
                        access: BufferAccess::WriteOnly,
                    },
                ];
                if matches!(
                    ir.effective_transform,
                    crate::R2rTransform::Dct(crate::DctType::III | crate::DctType::IV)
                        | crate::R2rTransform::Dst(crate::DstType::III | crate::DstType::IV)
                ) {
                    pre_bindings.push(ProgramPassBinding {
                        binding: 2,
                        resource: phases_id.ok_or(VkFftError::InvalidKernelIr(
                            "double-double R2R FFT preprocess is missing its phase LUT",
                        ))?,
                        role: BufferRole::LookupTable,
                        access: BufferAccess::ReadOnly,
                    });
                }
                passes.push(ProgramPass {
                    name: match ir.effective_transform {
                        crate::R2rTransform::Dct(crate::DctType::I) => {
                            "vkfft_dd_r2r_fft_dct1_pre".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::I) => {
                            "vkfft_dd_r2r_fft_dst1_pre".to_owned()
                        }
                        crate::R2rTransform::Dct(crate::DctType::II) => {
                            "vkfft_dd_r2r_fft_dct2_pre".to_owned()
                        }
                        crate::R2rTransform::Dct(crate::DctType::III) => {
                            "vkfft_dd_r2r_fft_dct3_pre".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::II) => {
                            "vkfft_dd_r2r_fft_dst2_pre".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::III) => {
                            "vkfft_dd_r2r_fft_dst3_pre".to_owned()
                        }
                        crate::R2rTransform::Dct(crate::DctType::IV) => {
                            "vkfft_dd_r2r_fft_dct4_pre".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::IV) => {
                            "vkfft_dd_r2r_fft_dst4_pre".to_owned()
                        }
                    },
                    dispatch,
                    bindings: pre_bindings,
                });
                for child_pass in &child.passes {
                    let mut pass = child_pass.clone();
                    pass.name = format!("vkfft_dd_r2r_{}", child_pass.name);
                    for child_binding in &mut pass.bindings {
                        child_binding.resource = resource_map
                            .get(child_binding.resource.0)
                            .and_then(|mapped| *mapped)
                            .ok_or(VkFftError::InvalidKernelIr(
                                "double-double R2R child resource mapping is incomplete",
                            ))?;
                    }
                    passes.push(pass);
                }
                let mut post_bindings = vec![
                    ProgramPassBinding {
                        binding: 0,
                        resource: fft_output,
                        role: BufferRole::Input,
                        access: BufferAccess::ReadOnly,
                    },
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                ];
                if matches!(
                    ir.effective_transform,
                    crate::R2rTransform::Dct(crate::DctType::II | crate::DctType::IV)
                        | crate::R2rTransform::Dst(crate::DstType::II | crate::DstType::IV)
                ) {
                    post_bindings.push(ProgramPassBinding {
                        binding: 2,
                        resource: phases_id.ok_or(VkFftError::InvalidKernelIr(
                            "double-double R2R FFT postprocess is missing its phase LUT",
                        ))?,
                        role: BufferRole::LookupTable,
                        access: BufferAccess::ReadOnly,
                    });
                }
                passes.push(ProgramPass {
                    name: match ir.effective_transform {
                        crate::R2rTransform::Dct(crate::DctType::I) => {
                            "vkfft_dd_r2r_fft_dct1_post".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::I) => {
                            "vkfft_dd_r2r_fft_dst1_post".to_owned()
                        }
                        crate::R2rTransform::Dct(crate::DctType::II) => {
                            "vkfft_dd_r2r_fft_dct2_post".to_owned()
                        }
                        crate::R2rTransform::Dct(crate::DctType::III) => {
                            "vkfft_dd_r2r_fft_dct3_post".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::II) => {
                            "vkfft_dd_r2r_fft_dst2_post".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::III) => {
                            "vkfft_dd_r2r_fft_dst3_post".to_owned()
                        }
                        crate::R2rTransform::Dct(crate::DctType::IV) => {
                            "vkfft_dd_r2r_fft_dct4_post".to_owned()
                        }
                        crate::R2rTransform::Dst(crate::DstType::IV) => {
                            "vkfft_dd_r2r_fft_dst4_post".to_owned()
                        }
                    },
                    dispatch,
                    bindings: post_bindings,
                });
            }
            DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize {
                fft_len,
                fft,
                pack_phases,
                extract_phases,
            } => {
                let fft_elements =
                    fft_len
                        .checked_mul(ir.batch_count)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double even-IV half-size program element count",
                        })?;
                let fft_input = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: fft_input,
                    name: "double_double_r2r_even_iv_fft_input".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ScalarType::DoubleDouble,
                    elements: fft_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                });
                let fft_output = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: fft_output,
                    name: "double_double_r2r_even_iv_fft_output".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ScalarType::DoubleDouble,
                    elements: fft_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                });
                let phases_id = ProgramResourceId(resources.len());
                let mut phases = Vec::with_capacity(pack_phases.len() + extract_phases.len());
                phases.extend_from_slice(pack_phases);
                phases.extend_from_slice(extract_phases);
                resources.push(ProgramResource {
                    id: phases_id,
                    name: "double_double_r2r_even_iv_phases".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: phases.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(phases),
                });

                let child = Self::double_double_one_dim(fft)?;
                if child.scalar != ScalarType::DoubleDouble
                    || child.input_resource()?.elements != fft_elements
                    || child.output_resource()?.elements != fft_elements
                    || child.input_resource()?.scalar != ScalarType::DoubleDouble
                    || child.output_resource()?.scalar != ScalarType::DoubleDouble
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double even-IV child must expose full-DD boundaries",
                    ));
                }
                let mut resource_map = vec![None; child.resources.len()];
                for resource in &child.resources {
                    let mapped = match resource.kind {
                        ProgramResourceKind::Input => fft_input,
                        ProgramResourceKind::Output => fft_output,
                        _ => {
                            let new_id = ProgramResourceId(resources.len());
                            let mut cloned = resource.clone();
                            cloned.id = new_id;
                            cloned.name = format!("double_double_r2r_even_iv_{}", resource.name);
                            cloned.external_layout = None;
                            resources.push(cloned);
                            new_id
                        }
                    };
                    resource_map[resource.id.0] = Some(mapped);
                }

                passes.push(ProgramPass {
                    name: "vkfft_dd_r2r_even_iv_pre".to_owned(),
                    dispatch,
                    bindings: vec![
                        binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                        ProgramPassBinding {
                            binding: 1,
                            resource: fft_input,
                            role: BufferRole::Output,
                            access: BufferAccess::WriteOnly,
                        },
                        ProgramPassBinding {
                            binding: 2,
                            resource: phases_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ],
                });
                for child_pass in &child.passes {
                    let mut pass = child_pass.clone();
                    pass.name = format!("vkfft_dd_r2r_even_iv_{}", child_pass.name);
                    for child_binding in &mut pass.bindings {
                        child_binding.resource = resource_map
                            .get(child_binding.resource.0)
                            .and_then(|mapped| *mapped)
                            .ok_or(VkFftError::InvalidKernelIr(
                                "double-double even-IV child resource mapping is incomplete",
                            ))?;
                    }
                    passes.push(pass);
                }
                passes.push(ProgramPass {
                    name: "vkfft_dd_r2r_even_iv_post".to_owned(),
                    dispatch,
                    bindings: vec![
                        ProgramPassBinding {
                            binding: 0,
                            resource: fft_output,
                            role: BufferRole::Input,
                            access: BufferAccess::ReadOnly,
                        },
                        binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                        ProgramPassBinding {
                            binding: 2,
                            resource: phases_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ],
                });
            }
        }
        let program = Self {
            name: format!("vkfft_dd_r2r_program_{}", ir.length),
            scalar: ScalarType::DoubleDouble,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn r2r(ir: &R2rIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.length
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "DCT/DST program element count",
                })?;
        let external = ExternalBufferLayout {
            logical_len: ir.length,
            physical_stride: ir.length,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.external_scalar,
                    elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.external_scalar,
                    elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        builder.flatten_r2r(ir, ProgramResourceId(0), ProgramResourceId(1))?;
        let program = Self {
            name: format!("vkfft_r2r_program_{}", ir.length),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn nd_r2r(ir: &NdR2rIr) -> Result<Self> {
        ir.validate()?;
        let dense_elements =
            ir.tensor_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional DCT/DST program element count",
                })?;
        let input_external_stride = if ir.input_formatted_copy.is_some() {
            ir.input_external_layout.batch_stride
        } else {
            ir.tensor_len
        };
        let output_external_stride = if ir.output_formatted_copy.is_some() {
            ir.output_external_layout.batch_stride
        } else {
            ir.tensor_len
        };
        let input_external = ExternalBufferLayout {
            logical_len: input_external_stride,
            physical_stride: input_external_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let output_external = ExternalBufferLayout {
            logical_len: output_external_stride,
            physical_stride: output_external_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.external_scalar,
                    elements: input_external.physical_elements()?,
                    external_layout: Some(input_external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.external_scalar,
                    elements: output_external.physical_elements()?,
                    external_layout: Some(output_external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        let mut natural_input = if let Some(copy) = &ir.input_formatted_copy {
            let dense = builder.add_scratch("nd_r2r_formatted_input", dense_elements);
            builder.passes.push(two_buffer_pass(
                &copy.name,
                copy.dispatch,
                ProgramResourceId(0).0,
                dense.0,
            ));
            dense
        } else {
            ProgramResourceId(0)
        };
        let formatted_output = if ir.output_formatted_copy.is_some() {
            Some(builder.add_scratch("nd_r2r_formatted_output", dense_elements))
        } else {
            None
        };
        for (axis_index, axis) in ir.axes.iter().enumerate() {
            let packed_input = builder.add_scratch("nd_r2r_axis_input", dense_elements);
            let packed_output = builder.add_scratch("nd_r2r_axis_output", dense_elements);
            let natural_output = if axis_index + 1 == ir.axes.len() {
                formatted_output.unwrap_or(ProgramResourceId(1))
            } else {
                builder.add_scratch("nd_r2r_natural_output", dense_elements)
            };
            builder.passes.push(two_buffer_pass(
                &axis.pack.name,
                axis.pack.dispatch,
                natural_input.0,
                packed_input.0,
            ));
            builder.flatten_r2r(&axis.transform, packed_input, packed_output)?;
            builder.passes.push(two_buffer_pass(
                &axis.scatter.name,
                axis.scatter.dispatch,
                packed_output.0,
                natural_output.0,
            ));
            natural_input = natural_output;
        }
        if let Some(copy) = &ir.output_formatted_copy {
            builder.passes.push(two_buffer_pass(
                &copy.name,
                copy.dispatch,
                formatted_output.expect("formatted R2R output scratch").0,
                ProgramResourceId(1).0,
            ));
        }
        let program = Self {
            name: format!("vkfft_nd_r2r_program_{:?}", ir.dimensions),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn nd_convolution(ir: &NdConvolutionIr) -> Result<Self> {
        ir.validate()?;
        let input_transform_batch_count = ir.batch_count.checked_mul(ir.coordinate_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "multidimensional performConvolution input coordinate-expanded batch count",
            },
        )?;
        let output_transform_batch_count = ir
            .output_batch_count()
            .checked_mul(ir.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional performConvolution output coordinate-expanded batch count",
            })?;
        let input_elements = ir
            .tensor_len
            .checked_mul(input_transform_batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional performConvolution input program element count",
            })?;
        let output_elements = ir
            .tensor_len
            .checked_mul(output_transform_batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional performConvolution output program element count",
            })?;
        let input_external = ExternalBufferLayout {
            logical_len: ir.tensor_len,
            physical_stride: ir.tensor_len,
            batch_count: input_transform_batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let output_external = ExternalBufferLayout {
            logical_len: ir.tensor_len,
            physical_stride: ir.tensor_len,
            batch_count: output_transform_batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut resources = vec![
            ProgramResource {
                id: ProgramResourceId(0),
                name: "input".to_owned(),
                kind: ProgramResourceKind::Input,
                scalar: ir.scalar,
                elements: input_elements,
                external_layout: Some(input_external),
                initialization: ProgramResourceInitialization::ExternalInput,
            },
            ProgramResource {
                id: ProgramResourceId(1),
                name: "output".to_owned(),
                kind: ProgramResourceKind::Output,
                scalar: ir.scalar,
                elements: output_elements,
                external_layout: Some(output_external),
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(2),
                name: "nd_convolution_kernel_spectrum".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ir.scalar,
                elements: ir.kernel_spectrum().len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::Complex64(
                    ir.kernel_spectrum().to_vec(),
                ),
            },
            ProgramResource {
                id: ProgramResourceId(3),
                name: "nd_convolution_forward_spectrum".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ir.scalar,
                elements: input_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
            ProgramResource {
                id: ProgramResourceId(4),
                name: "nd_convolution_multiplied_spectrum".to_owned(),
                kind: ProgramResourceKind::Scratch,
                scalar: ir.scalar,
                elements: output_elements,
                external_layout: None,
                initialization: ProgramResourceInitialization::Zeroed,
            },
        ];
        let mut passes = Vec::new();
        let append_child = |child: &NdFftIr,
                            prefix: &str,
                            input: ProgramResourceId,
                            output: ProgramResourceId,
                            resources: &mut Vec<ProgramResource>,
                            passes: &mut Vec<ProgramPass>|
         -> Result<()> {
            let child = Self::nd_fft(child)?;
            let mut resource_map = vec![None; child.resources.len()];
            for resource in &child.resources {
                let mapped = match resource.kind {
                    ProgramResourceKind::Input => input,
                    ProgramResourceKind::Output => output,
                    _ => {
                        let id = ProgramResourceId(resources.len());
                        let mut cloned = resource.clone();
                        cloned.id = id;
                        cloned.name = format!("{prefix}_{}", resource.name);
                        cloned.external_layout = None;
                        resources.push(cloned);
                        id
                    }
                };
                resource_map[resource.id.0] = Some(mapped);
            }
            for child_pass in &child.passes {
                let mut pass = child_pass.clone();
                pass.name = format!("{prefix}_{}", child_pass.name);
                for binding in &mut pass.bindings {
                    binding.resource = resource_map
                        .get(binding.resource.0)
                        .and_then(|mapped| *mapped)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "multidimensional convolution child resource mapping is incomplete",
                        ))?;
                }
                passes.push(pass);
            }
            Ok(())
        };
        append_child(
            &ir.forward_fft,
            "vkfft_nd_convolution_forward",
            ProgramResourceId(0),
            ProgramResourceId(3),
            &mut resources,
            &mut passes,
        )?;
        passes.push(ProgramPass {
            name: ir.multiply.name.clone(),
            dispatch: ir.multiply.dispatch,
            bindings: vec![
                binding(0, 3, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 4, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, 2, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ],
        });
        append_child(
            &ir.inverse_fft,
            "vkfft_nd_convolution_inverse",
            ProgramResourceId(4),
            ProgramResourceId(1),
            &mut resources,
            &mut passes,
        )?;
        let program = Self {
            name: format!("vkfft_nd_convolution_program_{:?}", ir.dimensions),
            scalar: ir.scalar,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn nd_real_convolution(ir: &NdRealConvolutionIr) -> Result<Self> {
        ir.validate()?;
        let forward = Self::nd_real_fft(&ir.forward_r2c)?;
        let inverse = Self::nd_real_fft(&ir.inverse_c2r)?;

        let child_resource =
            |program: &ProgramIr, kind: ProgramResourceKind| -> Result<ProgramResource> {
                program
                    .resources
                    .iter()
                    .find(|resource| resource.kind == kind)
                    .cloned()
                    .ok_or(VkFftError::InvalidKernelIr(
                        "multidimensional real convolution child is missing an external resource",
                    ))
            };
        let mut input = child_resource(&forward, ProgramResourceKind::Input)?;
        input.id = ProgramResourceId(0);
        input.name = "input".to_owned();
        let mut output = child_resource(&inverse, ProgramResourceKind::Output)?;
        output.id = ProgramResourceId(1);
        output.name = "output".to_owned();
        let mut forward_spectrum = child_resource(&forward, ProgramResourceKind::Output)?;
        forward_spectrum.id = ProgramResourceId(3);
        forward_spectrum.name = "nd_real_convolution_forward_spectrum".to_owned();
        forward_spectrum.kind = ProgramResourceKind::Scratch;
        forward_spectrum.external_layout = None;
        forward_spectrum.initialization = ProgramResourceInitialization::Zeroed;
        let mut multiplied_spectrum = child_resource(&inverse, ProgramResourceKind::Input)?;
        multiplied_spectrum.id = ProgramResourceId(4);
        multiplied_spectrum.name = "nd_real_convolution_multiplied_spectrum".to_owned();
        multiplied_spectrum.kind = ProgramResourceKind::Scratch;
        multiplied_spectrum.external_layout = None;
        multiplied_spectrum.initialization = ProgramResourceInitialization::Zeroed;

        let mut resources = vec![
            input,
            output,
            ProgramResource {
                id: ProgramResourceId(2),
                name: "nd_real_convolution_kernel_spectrum".to_owned(),
                kind: ProgramResourceKind::LookupTable,
                scalar: ir.scalar,
                elements: ir.kernel_spectrum().len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::Complex64(
                    ir.kernel_spectrum().to_vec(),
                ),
            },
            forward_spectrum,
            multiplied_spectrum,
        ];
        let mut passes = Vec::new();
        let append_child = |child: &ProgramIr,
                            prefix: &str,
                            input: ProgramResourceId,
                            output: ProgramResourceId,
                            resources: &mut Vec<ProgramResource>,
                            passes: &mut Vec<ProgramPass>|
         -> Result<()> {
            let mut resource_map = vec![None; child.resources.len()];
            for resource in &child.resources {
                let mapped = match resource.kind {
                    ProgramResourceKind::Input => input,
                    ProgramResourceKind::Output => output,
                    _ => {
                        let id = ProgramResourceId(resources.len());
                        let mut cloned = resource.clone();
                        cloned.id = id;
                        cloned.name = format!("{prefix}_{}", resource.name);
                        cloned.external_layout = None;
                        resources.push(cloned);
                        id
                    }
                };
                resource_map[resource.id.0] = Some(mapped);
            }
            for child_pass in &child.passes {
                let mut pass = child_pass.clone();
                pass.name = format!("{prefix}_{}", child_pass.name);
                for binding in &mut pass.bindings {
                    binding.resource = resource_map
                        .get(binding.resource.0)
                        .and_then(|mapped| *mapped)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "multidimensional real convolution child resource mapping is incomplete",
                        ))?;
                }
                passes.push(pass);
            }
            Ok(())
        };
        append_child(
            &forward,
            "vkfft_nd_real_convolution_forward",
            ProgramResourceId(0),
            ProgramResourceId(3),
            &mut resources,
            &mut passes,
        )?;
        passes.push(ProgramPass {
            name: ir.multiply.name.clone(),
            dispatch: ir.multiply.dispatch,
            bindings: vec![
                binding(0, 3, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 4, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, 2, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ],
        });
        append_child(
            &inverse,
            "vkfft_nd_real_convolution_inverse",
            ProgramResourceId(4),
            ProgramResourceId(1),
            &mut resources,
            &mut passes,
        )?;
        let program = Self {
            name: format!("vkfft_nd_real_convolution_program_{:?}", ir.dimensions),
            scalar: ir.scalar,
            resources,
            passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn convolution(ir: &ConvolutionIr) -> Result<Self> {
        ir.validate()?;
        let input_transform_batch_count = ir.batch_count.checked_mul(ir.coordinate_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "performConvolution program input coordinate-expanded batch count",
            },
        )?;
        let output_transform_batch_count = ir
            .output_batch_count()
            .checked_mul(ir.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution program output coordinate-expanded batch count",
            })?;
        let input_elements = ir
            .sequence_len
            .checked_mul(input_transform_batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution program input element count",
            })?;
        let output_elements = ir
            .sequence_len
            .checked_mul(output_transform_batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution program output element count",
            })?;
        let input_external = ExternalBufferLayout {
            logical_len: ir.sequence_len,
            physical_stride: ir.sequence_len,
            batch_count: input_transform_batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let output_external = ExternalBufferLayout {
            logical_len: ir.sequence_len,
            physical_stride: ir.sequence_len,
            batch_count: output_transform_batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.external_scalar,
                    elements: input_elements,
                    external_layout: Some(input_external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.external_scalar,
                    elements: output_elements,
                    external_layout: Some(output_external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        let kernel = builder.add_lut("convolution_kernel_spectrum", ir.kernel_spectrum().to_vec());
        if let (Some(input_copy), Some(output_copy)) =
            (&ir.input_storage_copy, &ir.output_storage_copy)
        {
            let input_compute =
                builder.add_scratch("convolution_caller_input_compute", input_elements);
            let output_compute =
                builder.add_scratch("convolution_caller_output_compute", output_elements);
            let forward_spectrum =
                builder.add_scratch("convolution_forward_spectrum", input_elements);
            let multiplied_spectrum =
                builder.add_scratch("convolution_multiplied_spectrum", output_elements);
            builder.passes.push(two_buffer_pass(
                &input_copy.name,
                input_copy.dispatch,
                0,
                input_compute.0,
            ));
            builder.flatten_one_dim(&ir.forward_fft, input_compute, forward_spectrum)?;
            builder.passes.push(ProgramPass {
                name: ir.multiply.name.clone(),
                dispatch: ir.multiply.dispatch,
                bindings: vec![
                    binding(
                        0,
                        forward_spectrum.0,
                        BufferRole::Input,
                        BufferAccess::ReadOnly,
                    ),
                    binding(
                        1,
                        multiplied_spectrum.0,
                        BufferRole::Output,
                        BufferAccess::WriteOnly,
                    ),
                    binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                ],
            });
            builder.flatten_one_dim(&ir.inverse_fft, multiplied_spectrum, output_compute)?;
            builder.passes.push(two_buffer_pass(
                &output_copy.name,
                output_copy.dispatch,
                output_compute.0,
                1,
            ));
            let program = Self {
                name: format!("vkfft_convolution_program_{}", ir.sequence_len),
                scalar: ir.scalar,
                resources: builder.resources,
                passes: builder.passes,
            };
            program.validate()?;
            return Ok(program);
        }
        if let Some(step) = &ir.fused_stockham_step {
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                ],
            });
        } else if let Some(step) = &ir.fused_multi_kernel_stockham_step {
            let stockham_twiddles = if step.requires_twiddle_lut()? {
                Some(builder.add_stockham_root_lut(step.sequence_len)?)
            } else {
                None
            };
            let mut bindings = vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ];
            if let Some(twiddles) = stockham_twiddles {
                bindings.push(binding(
                    3,
                    twiddles.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings,
            });
        } else if let Some(step) = &ir.fused_matrix_stockham_step {
            let stockham_twiddles = if step.requires_twiddle_lut()? {
                Some(builder.add_stockham_root_lut(step.sequence_len)?)
            } else {
                None
            };
            let mut bindings = vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ];
            if let Some(twiddles) = stockham_twiddles {
                bindings.push(binding(
                    3,
                    twiddles.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings,
            });
        } else if let Some(step) = &ir.fused_direct_rader_step {
            let rader_twiddles = builder.add_lut(
                "convolution_direct_rader_twiddles",
                step.forward.table.twiddles_by_generator_power.clone(),
            );
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    binding(
                        3,
                        rader_twiddles.0,
                        BufferRole::TwiddleLookupTable,
                        BufferAccess::ReadOnly,
                    ),
                ],
            });
        } else if let Some(step) = &ir.fused_direct_rader_multi_kernel_step {
            let rader_twiddles = builder.add_lut(
                "convolution_direct_rader_multi_kernel_twiddles",
                step.forward.table.twiddles_by_generator_power.clone(),
            );
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings: vec![
                    binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    binding(
                        3,
                        rader_twiddles.0,
                        BufferRole::TwiddleLookupTable,
                        BufferAccess::ReadOnly,
                    ),
                ],
            });
        } else if let Some(step) = &ir.fused_fft_rader_step {
            let forward_rader_spectrum = builder.add_lut(
                "convolution_fft_rader_forward_spectrum",
                step.forward.kernel_spectrum()?.to_vec(),
            );
            let inverse_rader_spectrum = builder.add_lut(
                "convolution_fft_rader_inverse_spectrum",
                step.inverse.kernel_spectrum()?.to_vec(),
            );
            let stockham_twiddles = if step.requires_twiddle_lut()? {
                Some(builder.add_stockham_root_lut(step.convolution_len)?)
            } else {
                None
            };
            let mut bindings = vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                binding(
                    3,
                    forward_rader_spectrum.0,
                    BufferRole::LookupTable,
                    BufferAccess::ReadOnly,
                ),
                binding(
                    4,
                    inverse_rader_spectrum.0,
                    BufferRole::LookupTable,
                    BufferAccess::ReadOnly,
                ),
            ];
            if let Some(twiddles) = stockham_twiddles {
                bindings.push(binding(
                    5,
                    twiddles.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings,
            });
        } else if let Some(step) = &ir.fused_fft_rader_multi_kernel_step {
            let forward_rader_spectrum = builder.add_lut(
                "convolution_fft_rader_multi_kernel_forward_spectrum",
                step.forward.kernel_spectrum()?.to_vec(),
            );
            let inverse_rader_spectrum = builder.add_lut(
                "convolution_fft_rader_multi_kernel_inverse_spectrum",
                step.inverse.kernel_spectrum()?.to_vec(),
            );
            let stockham_twiddles = if step.requires_twiddle_lut()? {
                Some(builder.add_stockham_root_lut(step.convolution_len)?)
            } else {
                None
            };
            let mut bindings = vec![
                binding(0, 0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                binding(
                    3,
                    forward_rader_spectrum.0,
                    BufferRole::LookupTable,
                    BufferAccess::ReadOnly,
                ),
                binding(
                    4,
                    inverse_rader_spectrum.0,
                    BufferRole::LookupTable,
                    BufferAccess::ReadOnly,
                ),
            ];
            if let Some(twiddles) = stockham_twiddles {
                bindings.push(binding(
                    5,
                    twiddles.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: step.name.clone(),
                dispatch: step.dispatch,
                bindings,
            });
        } else if let Some(two_upload) = &ir.fused_two_upload_multi_kernel_stockham {
            let forward_mid = builder.add_scratch(
                "convolution_two_upload_multi_kernel_forward_mid",
                input_elements,
            );
            let inverse_mid = builder.add_scratch(
                "convolution_two_upload_multi_kernel_inverse_mid",
                output_elements,
            );
            builder.push_stockham_pass(
                &two_upload.forward_high,
                ProgramResourceId(0),
                forward_mid,
                None,
                None,
            )?;
            let special_twiddle = match two_upload.special_twiddle_lut_len()? {
                Some(len) => Some(builder.add_stockham_root_lut(len)?),
                None => None,
            };
            let mut special_bindings = vec![
                binding(0, forward_mid.0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(
                    1,
                    inverse_mid.0,
                    BufferRole::Output,
                    BufferAccess::WriteOnly,
                ),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ];
            if let Some(twiddle) = special_twiddle {
                special_bindings.push(binding(
                    3,
                    twiddle.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: two_upload.name.clone(),
                dispatch: two_upload.special_dispatch,
                bindings: special_bindings,
            });
            builder.push_stockham_pass(
                &two_upload.inverse_high,
                inverse_mid,
                ProgramResourceId(1),
                None,
                None,
            )?;
        } else if let Some(two_upload) = &ir.fused_two_upload_stockham {
            let forward_mid =
                builder.add_scratch("convolution_two_upload_forward_mid", input_elements);
            let inverse_mid =
                builder.add_scratch("convolution_two_upload_inverse_mid", input_elements);
            builder.push_stockham_pass(
                &two_upload.forward_high,
                ProgramResourceId(0),
                forward_mid,
                None,
                None,
            )?;
            let special_twiddle = match two_upload.special_twiddle_lut_len()? {
                Some(len) => Some(builder.add_stockham_root_lut(len)?),
                None => None,
            };
            let mut special_bindings = vec![
                binding(0, forward_mid.0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(
                    1,
                    inverse_mid.0,
                    BufferRole::Output,
                    BufferAccess::WriteOnly,
                ),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ];
            if let Some(twiddle) = special_twiddle {
                special_bindings.push(binding(
                    3,
                    twiddle.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: two_upload.name.clone(),
                dispatch: two_upload.special_dispatch,
                bindings: special_bindings,
            });
            builder.push_stockham_pass(
                &two_upload.inverse_high,
                inverse_mid,
                ProgramResourceId(1),
                None,
                None,
            )?;
        } else if let Some(three_upload) = &ir.fused_three_upload_multi_kernel_stockham {
            let scratch_a = builder.add_scratch(
                "convolution_three_upload_multi_kernel_scratch_a",
                output_elements,
            );
            let scratch_b = builder.add_scratch(
                "convolution_three_upload_multi_kernel_scratch_b",
                output_elements,
            );
            builder.push_stockham_pass(
                &three_upload.forward_high,
                ProgramResourceId(0),
                scratch_a,
                None,
                None,
            )?;
            builder.push_stockham_pass(
                &three_upload.forward_middle,
                scratch_a,
                scratch_b,
                None,
                None,
            )?;
            builder.passes.push(ProgramPass {
                name: three_upload.name.clone(),
                dispatch: three_upload.special_dispatch,
                bindings: vec![
                    binding(0, scratch_b.0, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, scratch_a.0, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                ],
            });
            builder.push_stockham_pass(
                &three_upload.inverse_middle,
                scratch_a,
                scratch_b,
                None,
                None,
            )?;
            builder.push_stockham_pass(
                &three_upload.inverse_high,
                scratch_b,
                ProgramResourceId(1),
                None,
                None,
            )?;
        } else if let Some(three_upload) = &ir.fused_three_upload_stockham {
            let scratch_a =
                builder.add_scratch("convolution_three_upload_scratch_a", input_elements);
            let scratch_b =
                builder.add_scratch("convolution_three_upload_scratch_b", input_elements);
            builder.push_stockham_pass(
                &three_upload.forward_high,
                ProgramResourceId(0),
                scratch_a,
                None,
                None,
            )?;
            builder.push_stockham_pass(
                &three_upload.forward_middle,
                scratch_a,
                scratch_b,
                None,
                None,
            )?;
            let special_twiddle = match three_upload.special_twiddle_lut_len()? {
                Some(len) => Some(builder.add_stockham_root_lut(len)?),
                None => None,
            };
            let mut special_bindings = vec![
                binding(0, scratch_b.0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, scratch_a.0, BufferRole::Output, BufferAccess::WriteOnly),
                binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
            ];
            if let Some(twiddle) = special_twiddle {
                special_bindings.push(binding(
                    3,
                    twiddle.0,
                    BufferRole::TwiddleLookupTable,
                    BufferAccess::ReadOnly,
                ));
            }
            builder.passes.push(ProgramPass {
                name: three_upload.name.clone(),
                dispatch: three_upload.special_dispatch,
                bindings: special_bindings,
            });
            builder.push_stockham_pass(
                &three_upload.inverse_middle,
                scratch_a,
                scratch_b,
                None,
                None,
            )?;
            builder.push_stockham_pass(
                &three_upload.inverse_high,
                scratch_b,
                ProgramResourceId(1),
                None,
                None,
            )?;
        } else {
            let forward_spectrum =
                builder.add_scratch("convolution_forward_spectrum", input_elements);
            builder.flatten_one_dim(&ir.forward_fft, ProgramResourceId(0), forward_spectrum)?;
            if let Some(fused_inverse) = &ir.fused_inverse_stockham {
                builder.push_stockham_pass(
                    fused_inverse,
                    forward_spectrum,
                    ProgramResourceId(1),
                    Some(kernel),
                    None,
                )?;
            } else {
                let multiplied_spectrum =
                    builder.add_scratch("convolution_multiplied_spectrum", output_elements);
                builder.passes.push(ProgramPass {
                    name: ir.multiply.name.clone(),
                    dispatch: ir.multiply.dispatch,
                    bindings: vec![
                        binding(
                            0,
                            forward_spectrum.0,
                            BufferRole::Input,
                            BufferAccess::ReadOnly,
                        ),
                        binding(
                            1,
                            multiplied_spectrum.0,
                            BufferRole::Output,
                            BufferAccess::WriteOnly,
                        ),
                        binding(2, kernel.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    ],
                });
                builder.flatten_one_dim(
                    &ir.inverse_fft,
                    multiplied_spectrum,
                    ProgramResourceId(1),
                )?;
            }
        }
        let program = Self {
            name: format!("vkfft_perform_convolution_program_{}", ir.sequence_len),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn bluestein(pipeline: &BluesteinPipelineIr) -> Result<Self> {
        pipeline.validate()?;
        let logical_elements = pipeline
            .logical_len
            .checked_mul(pipeline.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Bluestein program logical element count",
            })?;
        let convolution_elements = pipeline
            .convolution_len
            .checked_mul(pipeline.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Bluestein program convolution element count",
            })?;
        let external = ExternalBufferLayout {
            logical_len: pipeline.logical_len,
            physical_stride: pipeline.logical_len,
            batch_count: pipeline.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let lut = pipeline.kernel_spectrum()?.to_vec();
        let mut builder = RecursiveProgramBuilder {
            scalar: pipeline.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: pipeline.external_storage_scalar(),
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: pipeline.external_storage_scalar(),
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(2),
                    name: "kernel_spectrum".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: pipeline.scalar,
                    elements: pipeline.convolution_len,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Complex64(lut),
                },
                ProgramResource {
                    id: ProgramResourceId(3),
                    name: "scratch_a".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: pipeline.scalar,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(4),
                    name: "scratch_b".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: pipeline.scalar,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: vec![two_buffer_pass(
                &pipeline.preprocess.name,
                pipeline.preprocess.dispatch,
                0,
                3,
            )],
            serial: 0,
        };
        builder.flatten_recursive_ir(
            &pipeline.forward_fft,
            ProgramResourceId(3),
            ProgramResourceId(4),
        )?;
        let convolution_output =
            if let Some(fused_inverse) = pipeline.fused_inverse_stockham_kernel()? {
                builder.push_stockham_pass(
                    &fused_inverse,
                    ProgramResourceId(4),
                    ProgramResourceId(3),
                    Some(ProgramResourceId(2)),
                    None,
                )?;
                ProgramResourceId(3)
            } else if pipeline.has_fused_recursive_inverse() {
                builder.flatten_node_with_boundary_resources(
                    &pipeline.inverse_fft.root,
                    ProgramResourceId(4),
                    ProgramResourceId(3),
                    Some(ProgramResourceId(2)),
                    None,
                )?;
                ProgramResourceId(3)
            } else {
                builder.passes.push(ProgramPass {
                    name: pipeline.multiply.name.clone(),
                    dispatch: pipeline.multiply.dispatch,
                    bindings: vec![
                        binding(0, 4, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, 3, BufferRole::Output, BufferAccess::WriteOnly),
                        binding(2, 2, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    ],
                });
                builder.flatten_recursive_ir(
                    &pipeline.inverse_fft,
                    ProgramResourceId(3),
                    ProgramResourceId(4),
                )?;
                ProgramResourceId(4)
            };
        builder.passes.push(two_buffer_pass(
            &pipeline.postprocess.name,
            pipeline.postprocess.dispatch,
            convolution_output.0,
            1,
        ));
        let program = Self {
            name: format!("vkfft_bluestein_program_{}", pipeline.logical_len),
            scalar: pipeline.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn rader_fft(pipeline: &RaderFftPipelineIr) -> Result<Self> {
        pipeline.validate()?;
        let logical_elements = pipeline.prime.checked_mul(pipeline.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "FFT Rader program logical element count",
            },
        )?;
        let convolution_elements = pipeline
            .convolution_len
            .checked_mul(pipeline.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "FFT Rader program convolution element count",
            })?;
        let external = ExternalBufferLayout {
            logical_len: pipeline.prime,
            physical_stride: pipeline.prime,
            batch_count: pipeline.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: pipeline.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: pipeline.input_storage_scalar,
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "scratch_a".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: pipeline.scalar,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(2),
                    name: "scratch_b".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: pipeline.scalar,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(3),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: pipeline.output_storage_scalar,
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(4),
                    name: "kernel_spectrum".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: pipeline.scalar,
                    elements: pipeline.convolution_len,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Complex64(
                        pipeline.kernel_spectrum()?.to_vec(),
                    ),
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        let forward_input = match pipeline.input_strategy {
            RaderFftInputStrategy::GatherReversePass => {
                builder.passes.push(two_buffer_pass(
                    &pipeline.gather.name,
                    pipeline.gather.dispatch,
                    0,
                    1,
                ));
                ProgramResourceId(1)
            }
            RaderFftInputStrategy::GeneratorOrderStockham
            | RaderFftInputStrategy::GeneratorOrderRecursive => ProgramResourceId(0),
        };
        builder.flatten_one_dim(&pipeline.forward_fft, forward_input, ProgramResourceId(2))?;
        if let Some(fused_inverse) = pipeline.fused_inverse_rader_kernel()? {
            builder.push_stockham_pass(
                &fused_inverse,
                ProgramResourceId(2),
                ProgramResourceId(3),
                Some(ProgramResourceId(4)),
                Some(ProgramResourceId(0)),
            )?;
        } else if pipeline.has_fused_recursive_inverse() {
            let inverse = pipeline
                .inverse_recursive()
                .ok_or(VkFftError::InvalidKernelIr(
                    "fused recursive Rader program lost its recursive inverse child",
                ))?;
            builder.flatten_node_with_boundary_resources(
                &inverse.root,
                ProgramResourceId(2),
                ProgramResourceId(3),
                Some(ProgramResourceId(4)),
                Some(ProgramResourceId(0)),
            )?;
        } else {
            builder.passes.push(ProgramPass {
                name: pipeline.multiply.name.clone(),
                dispatch: pipeline.multiply.dispatch,
                bindings: vec![
                    binding(0, 2, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 1, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, 4, BufferRole::LookupTable, BufferAccess::ReadOnly),
                ],
            });
            builder.flatten_one_dim(
                &pipeline.inverse_fft,
                ProgramResourceId(1),
                ProgramResourceId(2),
            )?;
            builder.passes.push(ProgramPass {
                name: pipeline.scatter.name.clone(),
                dispatch: pipeline.scatter.dispatch,
                bindings: vec![
                    binding(0, 2, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 3, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, 0, BufferRole::Auxiliary, BufferAccess::ReadOnly),
                ],
            });
        }
        let program = Self {
            name: format!("vkfft_rader_fft_program_{}", pipeline.prime),
            scalar: pipeline.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn mixed_rader_stockham(ir: &MixedRaderStockhamIr) -> Result<Self> {
        ir.validate()?;
        match &ir.prime_stage {
            MixedPrimeRaderIr::FftConvolution(rader) => Self::mixed_rader_stockham_fft(ir, rader),
            MixedPrimeRaderIr::Direct(direct) => Self::mixed_rader_stockham_direct(ir, direct),
        }
    }

    fn mixed_rader_stockham_fft(
        ir: &MixedRaderStockhamIr,
        rader: &RaderFftPipelineIr,
    ) -> Result<Self> {
        let logical_elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "mixed program logical element count",
                })?;
        let convolution_elements = rader.convolution_len.checked_mul(rader.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "mixed program Rader convolution element count",
            },
        )?;
        let external = ExternalBufferLayout {
            logical_len: ir.logical_len,
            physical_stride: ir.logical_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "prime_input".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(2),
                    name: "rader_scratch_a".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(3),
                    name: "rader_scratch_b".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: convolution_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(4),
                    name: "prime_output".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(5),
                    name: "rader_kernel_spectrum".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ir.scalar,
                    elements: rader.convolution_len,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Complex64(
                        rader.kernel_spectrum()?.to_vec(),
                    ),
                },
                ProgramResource {
                    id: ProgramResourceId(6),
                    name: "stockham_input".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(7),
                    name: "stockham_output".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(8),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: vec![two_buffer_pass(
                &ir.pack_prime.name,
                ir.pack_prime.dispatch,
                0,
                1,
            )],
            serial: 0,
        };
        let forward_input = match rader.input_strategy {
            RaderFftInputStrategy::GatherReversePass => {
                builder.passes.push(two_buffer_pass(
                    &rader.gather.name,
                    rader.gather.dispatch,
                    1,
                    2,
                ));
                ProgramResourceId(2)
            }
            RaderFftInputStrategy::GeneratorOrderStockham
            | RaderFftInputStrategy::GeneratorOrderRecursive => ProgramResourceId(1),
        };
        builder.flatten_one_dim(&rader.forward_fft, forward_input, ProgramResourceId(3))?;
        if let Some(fused_inverse) = rader.fused_inverse_rader_kernel()? {
            builder.push_stockham_pass(
                &fused_inverse,
                ProgramResourceId(3),
                ProgramResourceId(4),
                Some(ProgramResourceId(5)),
                Some(ProgramResourceId(1)),
            )?;
        } else if rader.has_fused_recursive_inverse() {
            let inverse = rader
                .inverse_recursive()
                .ok_or(VkFftError::InvalidKernelIr(
                    "mixed fused recursive Rader program lost its recursive inverse child",
                ))?;
            builder.flatten_node_with_boundary_resources(
                &inverse.root,
                ProgramResourceId(3),
                ProgramResourceId(4),
                Some(ProgramResourceId(5)),
                Some(ProgramResourceId(1)),
            )?;
        } else {
            builder.passes.push(ProgramPass {
                name: rader.multiply.name.clone(),
                dispatch: rader.multiply.dispatch,
                bindings: vec![
                    binding(0, 3, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 2, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, 5, BufferRole::LookupTable, BufferAccess::ReadOnly),
                ],
            });
            builder.flatten_one_dim(
                &rader.inverse_fft,
                ProgramResourceId(2),
                ProgramResourceId(3),
            )?;
            builder.passes.push(ProgramPass {
                name: rader.scatter.name.clone(),
                dispatch: rader.scatter.dispatch,
                bindings: vec![
                    binding(0, 3, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, 4, BufferRole::Output, BufferAccess::WriteOnly),
                    binding(2, 1, BufferRole::Auxiliary, BufferAccess::ReadOnly),
                ],
            });
        }
        builder.passes.push(two_buffer_pass(
            &ir.twiddle_transpose.name,
            ir.twiddle_transpose.dispatch,
            4,
            6,
        ));
        builder.passes.push(two_buffer_pass(
            &ir.stockham_stage.name,
            ir.stockham_stage.dispatch,
            6,
            7,
        ));
        builder.passes.push(two_buffer_pass(
            &ir.scatter_output.name,
            ir.scatter_output.dispatch,
            7,
            8,
        ));
        let program = Self {
            name: format!("vkfft_mixed_rader_stockham_program_{}", ir.logical_len),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    fn mixed_rader_stockham_direct(
        ir: &MixedRaderStockhamIr,
        direct: &RaderDirectIr,
    ) -> Result<Self> {
        let logical_elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "mixed direct-Rader program logical element count",
                })?;
        let external = ExternalBufferLayout {
            logical_len: ir.logical_len,
            physical_stride: ir.logical_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let program = Self {
            name: format!(
                "vkfft_mixed_direct_rader_stockham_program_{}",
                ir.logical_len
            ),
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "prime_input".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(2),
                    name: "prime_output".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(3),
                    name: "rader_roots".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ir.scalar,
                    elements: direct.twiddle_lut().len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Complex64(
                        direct.twiddle_lut().to_vec(),
                    ),
                },
                ProgramResource {
                    id: ProgramResourceId(4),
                    name: "stockham_input".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(5),
                    name: "stockham_output".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(6),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.scalar,
                    elements: logical_elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: vec![
                two_buffer_pass(&ir.pack_prime.name, ir.pack_prime.dispatch, 0, 1),
                ProgramPass {
                    name: direct.name.clone(),
                    dispatch: direct.dispatch,
                    bindings: vec![
                        binding(0, 1, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, 2, BufferRole::Output, BufferAccess::WriteOnly),
                        binding(2, 3, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    ],
                },
                two_buffer_pass(
                    &ir.twiddle_transpose.name,
                    ir.twiddle_transpose.dispatch,
                    2,
                    4,
                ),
                two_buffer_pass(&ir.stockham_stage.name, ir.stockham_stage.dispatch, 4, 5),
                two_buffer_pass(&ir.scatter_output.name, ir.scatter_output.dispatch, 5, 6),
            ],
        };
        program.validate()?;
        Ok(program)
    }

    pub fn recursive_fft(ir: &RecursiveFftIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive FFT program element count",
                })?;
        let external = ExternalBufferLayout {
            logical_len: ir.logical_len,
            physical_stride: ir.logical_len,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let external_scalar = ir.external_storage_scalar();
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: external_scalar,
                    elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: external_scalar,
                    elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        builder.flatten_recursive_ir(ir, ProgramResourceId(0), ProgramResourceId(1))?;
        let program = Self {
            name: format!("vkfft_recursive_fft_program_{}", ir.logical_len),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn one_dim_fft(ir: &OneDimFftIr) -> Result<Self> {
        ir.validate()?;
        if let OneDimFftIr::Recursive(recursive) = ir
            && ir.zero_pad_pass().is_none()
        {
            return Self::recursive_fft(recursive);
        }
        let elements = ir.logical_len().checked_mul(ir.batch_count()).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "generic one-dimensional FFT program element count",
            },
        )?;
        let external = ExternalBufferLayout {
            logical_len: ir.logical_len(),
            physical_stride: ir.logical_len(),
            batch_count: ir.batch_count(),
            element_shape: ProgramElementShape::Complex,
        };
        let external_scalar = ir.external_storage_scalar();
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar(),
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: external_scalar,
                    elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: external_scalar,
                    elements,
                    external_layout: Some(external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        builder.flatten_one_dim(ir, ProgramResourceId(0), ProgramResourceId(1))?;
        let program = Self {
            name: format!("vkfft_1d_fft_program_{}", ir.logical_len()),
            scalar: ir.scalar(),
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn nd_real_fft(ir: &NdRealFftIr) -> Result<Self> {
        ir.validate()?;
        let full_elements = ir.full_tensor_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "multidimensional real program full element count",
            },
        )?;
        let compact_elements = ir.compact_tensor_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "multidimensional real program compact element count",
            },
        )?;
        let input_tensor_len = ir.input_tensor_len();
        let output_tensor_len = ir.output_tensor_len();
        let input_layout = ExternalBufferLayout {
            logical_len: if ir.input_formatted_copy.is_some() {
                ir.input_external_layout.batch_stride
            } else {
                input_tensor_len
            },
            physical_stride: if ir.input_formatted_copy.is_some() {
                ir.input_external_layout.batch_stride
            } else {
                input_tensor_len
            },
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let output_layout = ExternalBufferLayout {
            logical_len: if ir.output_formatted_copy.is_some() {
                ir.output_external_layout.batch_stride
            } else {
                output_tensor_len
            },
            physical_stride: if ir.output_formatted_copy.is_some() {
                ir.output_external_layout.batch_stride
            } else {
                output_tensor_len
            },
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let input_elements = input_layout.physical_elements()?;
        let output_elements = output_layout.physical_elements()?;
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: if ir.input_boundary_compute_storage {
                        ir.scalar
                    } else {
                        ir.external_scalar
                    },
                    elements: input_elements,
                    external_layout: Some(input_layout),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: if ir.output_boundary_compute_storage {
                        ir.scalar
                    } else {
                        ir.external_scalar
                    },
                    elements: output_elements,
                    external_layout: Some(output_layout),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };

        let dense_input = if let Some(copy) = &ir.input_formatted_copy {
            let elements = input_tensor_len.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multidimensional real formatted dense input size",
                },
            )?;
            let dense = builder.add_scratch("nd_real_formatted_input", elements);
            builder.passes.push(two_buffer_pass(
                &copy.name,
                copy.dispatch,
                ProgramResourceId(0).0,
                dense.0,
            ));
            dense
        } else {
            ProgramResourceId(0)
        };
        let dense_output = if ir.output_formatted_copy.is_some() {
            let elements = output_tensor_len.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multidimensional real formatted dense output size",
                },
            )?;
            Some(builder.add_scratch("nd_real_formatted_output", elements))
        } else {
            None
        };

        match ir.kind {
            RealFftKind::RealToComplex => {
                let real_input = if let Some(zero_pad) = &ir.zero_pad_pass {
                    let boundary = builder.add_scratch("nd_real_zero_padded_input", full_elements);
                    builder.push_nd_zero_pad_pass(zero_pad, dense_input, boundary)?;
                    boundary
                } else {
                    dense_input
                };
                let final_output = dense_output.unwrap_or(ProgramResourceId(1));
                let mut natural = if ir.complex_axes.is_empty() {
                    final_output
                } else {
                    builder.add_scratch("nd_real_compact_natural", compact_elements)
                };
                builder.flatten_real_fft(&ir.real_axis, real_input, natural)?;
                for (axis_index, axis) in ir.complex_axes.iter().enumerate() {
                    let packed = builder.add_scratch("nd_real_axis_input", compact_elements);
                    let transformed = builder.add_scratch("nd_real_axis_output", compact_elements);
                    let next = if axis_index + 1 == ir.complex_axes.len() {
                        final_output
                    } else {
                        builder.add_scratch("nd_real_axis_natural", compact_elements)
                    };
                    builder.passes.push(two_buffer_pass(
                        &axis.pack.name,
                        axis.pack.dispatch,
                        natural.0,
                        packed.0,
                    ));
                    builder.flatten_one_dim(&axis.transform, packed, transformed)?;
                    builder.passes.push(two_buffer_pass(
                        &axis.scatter.name,
                        axis.scatter.dispatch,
                        transformed.0,
                        next.0,
                    ));
                    natural = next;
                }
                if let Some(copy) = &ir.output_formatted_copy {
                    builder.passes.push(two_buffer_pass(
                        &copy.name,
                        copy.dispatch,
                        final_output.0,
                        ProgramResourceId(1).0,
                    ));
                }
            }
            RealFftKind::ComplexToReal => {
                let mut natural = dense_input;
                for axis in &ir.complex_axes {
                    let packed = builder.add_scratch("nd_real_axis_input", compact_elements);
                    let transformed = builder.add_scratch("nd_real_axis_output", compact_elements);
                    let next = builder.add_scratch("nd_real_axis_natural", compact_elements);
                    builder.passes.push(two_buffer_pass(
                        &axis.pack.name,
                        axis.pack.dispatch,
                        natural.0,
                        packed.0,
                    ));
                    builder.flatten_one_dim(&axis.transform, packed, transformed)?;
                    builder.passes.push(two_buffer_pass(
                        &axis.scatter.name,
                        axis.scatter.dispatch,
                        transformed.0,
                        next.0,
                    ));
                    natural = next;
                }
                let final_output = dense_output.unwrap_or(ProgramResourceId(1));
                let real_output = if ir.zero_pad_pass.is_some() {
                    builder.add_scratch("nd_real_unpadded_output", full_elements)
                } else {
                    final_output
                };
                builder.flatten_real_fft(&ir.real_axis, natural, real_output)?;
                let dense_final = if let Some(zero_pad) = &ir.zero_pad_pass {
                    builder.push_nd_zero_pad_pass(zero_pad, real_output, final_output)?;
                    final_output
                } else {
                    real_output
                };
                if let Some(copy) = &ir.output_formatted_copy {
                    builder.passes.push(two_buffer_pass(
                        &copy.name,
                        copy.dispatch,
                        dense_final.0,
                        ProgramResourceId(1).0,
                    ));
                }
            }
        }
        let program = Self {
            name: format!("vkfft_nd_real_program_{:?}_{:?}", ir.dimensions, ir.kind),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn nd_fft(ir: &NdFftIr) -> Result<Self> {
        ir.validate()?;
        let elements =
            ir.tensor_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional FFT program element count",
                })?;
        let input_external_logical_len = if ir.has_formatted_input_tensor_strides()? {
            ir.input_buffer_batch_stride
        } else {
            ir.tensor_len
        };
        let output_external_logical_len = if ir.has_formatted_output_tensor_strides()? {
            ir.output_buffer_batch_stride
        } else {
            ir.tensor_len
        };
        let input_external = ExternalBufferLayout {
            logical_len: input_external_logical_len,
            physical_stride: ir.input_buffer_batch_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let output_external = ExternalBufferLayout {
            logical_len: output_external_logical_len,
            physical_stride: ir.output_buffer_batch_stride,
            batch_count: ir.batch_count,
            element_shape: ProgramElementShape::Complex,
        };
        let input_elements = input_external.physical_elements()?;
        let output_elements = output_external.physical_elements()?;
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.external_scalar,
                    elements: input_elements,
                    external_layout: Some(input_external),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.external_scalar,
                    elements: output_elements,
                    external_layout: Some(output_external),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        let mut natural_input = ProgramResourceId(0);
        if let Some(copy) = &ir.input_formatted_copy {
            let dense = builder.add_scratch("nd_formatted_input", elements);
            builder.passes.push(two_buffer_pass(
                &copy.name,
                copy.dispatch,
                natural_input.0,
                dense.0,
            ));
            natural_input = dense;
        }
        if let Some(pass) = &ir.zero_pad_pass
            && pass.operation.is_input_boundary()
        {
            let boundary = builder.add_scratch("nd_zero_pad_boundary", elements);
            builder.push_nd_zero_pad_pass(pass, natural_input, boundary)?;
            natural_input = boundary;
        }
        for (axis_index, axis) in ir.axes.iter().enumerate() {
            let packed_input = builder.add_scratch("nd_axis_input", elements);
            let packed_output = builder.add_scratch("nd_axis_output", elements);
            let natural_output = if axis_index + 1 == ir.axes.len() {
                if ir
                    .zero_pad_pass
                    .as_ref()
                    .is_some_and(|pass| pass.operation.is_output_boundary())
                    || ir.output_formatted_copy.is_some()
                {
                    builder.add_scratch("nd_external_boundary", elements)
                } else {
                    ProgramResourceId(1)
                }
            } else {
                builder.add_scratch("nd_natural_output", elements)
            };
            builder.passes.push(two_buffer_pass(
                &axis.pack.name,
                axis.pack.dispatch,
                natural_input.0,
                packed_input.0,
            ));
            builder.flatten_one_dim(&axis.transform, packed_input, packed_output)?;
            builder.passes.push(two_buffer_pass(
                &axis.scatter.name,
                axis.scatter.dispatch,
                packed_output.0,
                natural_output.0,
            ));
            natural_input = natural_output;
        }
        if let Some(pass) = &ir.zero_pad_pass
            && pass.operation.is_output_boundary()
        {
            let target = if ir.output_formatted_copy.is_some() {
                builder.add_scratch("nd_formatted_output", elements)
            } else {
                ProgramResourceId(1)
            };
            builder.push_nd_zero_pad_pass(pass, natural_input, target)?;
            natural_input = target;
        }
        if let Some(copy) = &ir.output_formatted_copy {
            builder.passes.push(two_buffer_pass(
                &copy.name,
                copy.dispatch,
                natural_input.0,
                ProgramResourceId(1).0,
            ));
        }
        let program = Self {
            name: format!("vkfft_nd_fft_program_{:?}", ir.dimensions),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    pub fn real_fft(ir: &RealFftIr) -> Result<Self> {
        ir.validate()?;
        let full_elements =
            ir.length
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "real FFT program full element count",
                })?;
        let half_elements = ir.half_spectrum_len.checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "real FFT program half-spectrum element count",
            },
        )?;
        let (input_elements, output_elements, input_layout, output_layout) = match ir.kind {
            RealFftKind::RealToComplex => (
                full_elements,
                half_elements,
                ExternalBufferLayout {
                    logical_len: ir.length,
                    physical_stride: ir.length,
                    batch_count: ir.batch_count,
                    element_shape: ProgramElementShape::Complex,
                },
                ExternalBufferLayout {
                    logical_len: ir.half_spectrum_len,
                    physical_stride: ir.half_spectrum_len,
                    batch_count: ir.batch_count,
                    element_shape: ProgramElementShape::Complex,
                },
            ),
            RealFftKind::ComplexToReal => (
                half_elements,
                full_elements,
                ExternalBufferLayout {
                    logical_len: ir.half_spectrum_len,
                    physical_stride: ir.half_spectrum_len,
                    batch_count: ir.batch_count,
                    element_shape: ProgramElementShape::Complex,
                },
                ExternalBufferLayout {
                    logical_len: ir.length,
                    physical_stride: ir.length,
                    batch_count: ir.batch_count,
                    element_shape: ProgramElementShape::Complex,
                },
            ),
        };
        let mut builder = RecursiveProgramBuilder {
            scalar: ir.scalar,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ir.input_storage_scalar,
                    elements: input_elements,
                    external_layout: Some(input_layout),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ir.output_storage_scalar,
                    elements: output_elements,
                    external_layout: Some(output_layout),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: Vec::new(),
            serial: 0,
        };
        let external_input = ProgramResourceId(0);
        let external_output = ProgramResourceId(1);
        let transform_input = if ir.kind == RealFftKind::RealToComplex {
            if let Some(zero_pad) = &ir.zero_pad_pass {
                let scratch = builder.add_scratch("real_zero_padded_input", full_elements);
                builder.passes.push(two_buffer_pass(
                    &zero_pad.name,
                    zero_pad.dispatch,
                    external_input.0,
                    scratch.0,
                ));
                scratch
            } else {
                external_input
            }
        } else {
            external_input
        };
        let transform_output =
            if ir.kind == RealFftKind::ComplexToReal && ir.zero_pad_pass.is_some() {
                builder.add_scratch("real_unpadded_output", full_elements)
            } else {
                external_output
            };
        builder.flatten_real_fft(ir, transform_input, transform_output)?;
        if ir.kind == RealFftKind::ComplexToReal
            && let Some(zero_pad) = &ir.zero_pad_pass
        {
            builder.passes.push(two_buffer_pass(
                &zero_pad.name,
                zero_pad.dispatch,
                transform_output.0,
                external_output.0,
            ));
        }
        let program = Self {
            name: format!(
                "vkfft_real_fft_program_{}_{}",
                ir.length,
                match ir.kind {
                    RealFftKind::RealToComplex => "r2c",
                    RealFftKind::ComplexToReal => "c2r",
                }
            ),
            scalar: ir.scalar,
            resources: builder.resources,
            passes: builder.passes,
        };
        program.validate()?;
        Ok(program)
    }

    /// Build a backend-neutral physical allocation plan for the logical resources.
    /// External input/output stay dedicated. Identical immutable LUTs share one
    /// allocation, while scratch resources can reuse a slot after their previous
    /// lifetime ends when their first access is `WriteOnly` (the full-overwrite
    /// contract used by the generated kernels in this crate).
    pub fn memory_plan(&self) -> Result<ProgramMemoryPlan> {
        self.validate()?;
        let mut resource_allocations = vec![ProgramAllocationId(usize::MAX); self.resources.len()];
        let mut allocations = Vec::<ProgramAllocation>::new();

        let add_dedicated =
            |resource: &ProgramResource,
             kind: ProgramAllocationKind,
             allocations: &mut Vec<ProgramAllocation>,
             resource_allocations: &mut Vec<ProgramAllocationId>| {
                let id = ProgramAllocationId(allocations.len());
                allocations.push(ProgramAllocation {
                    id,
                    kind,
                    scalar: resource.scalar,
                    elements: resource.elements,
                    resources: vec![resource.id],
                });
                resource_allocations[resource.id.0] = id;
            };

        for resource in &self.resources {
            match resource.kind {
                ProgramResourceKind::Input => add_dedicated(
                    resource,
                    ProgramAllocationKind::ExternalInput,
                    &mut allocations,
                    &mut resource_allocations,
                ),
                ProgramResourceKind::Output => add_dedicated(
                    resource,
                    ProgramAllocationKind::ExternalOutput,
                    &mut allocations,
                    &mut resource_allocations,
                ),
                ProgramResourceKind::Scratch | ProgramResourceKind::LookupTable => {}
            }
        }

        for resource in self
            .resources
            .iter()
            .filter(|resource| resource.kind == ProgramResourceKind::LookupTable)
        {
            let shared = self.resources[..resource.id.0].iter().find(|candidate| {
                candidate.kind == ProgramResourceKind::LookupTable
                    && candidate.scalar == resource.scalar
                    && candidate.elements == resource.elements
                    && candidate.initialization == resource.initialization
            });
            if let Some(shared) = shared {
                let allocation_id = resource_allocations[shared.id.0];
                if allocation_id.0 == usize::MAX {
                    return Err(VkFftError::InvalidKernelIr(
                        "program LUT deduplication encountered an unassigned resource",
                    ));
                }
                resource_allocations[resource.id.0] = allocation_id;
                allocations[allocation_id.0].resources.push(resource.id);
            } else {
                add_dedicated(
                    resource,
                    ProgramAllocationKind::LookupTable,
                    &mut allocations,
                    &mut resource_allocations,
                );
            }
        }

        #[derive(Clone, Copy)]
        struct ScratchLifetime {
            resource: ProgramResourceId,
            first: usize,
            last: usize,
            first_access: BufferAccess,
        }

        let mut scratch_lifetimes = Vec::<ScratchLifetime>::new();
        for resource in self
            .resources
            .iter()
            .filter(|resource| resource.kind == ProgramResourceKind::Scratch)
        {
            let mut first = None::<(usize, BufferAccess)>;
            let mut last = None::<usize>;
            for (pass_index, pass) in self.passes.iter().enumerate() {
                for binding in pass
                    .bindings
                    .iter()
                    .filter(|binding| binding.resource == resource.id)
                {
                    if first.is_none() {
                        first = Some((pass_index, binding.access));
                    }
                    last = Some(pass_index);
                }
            }
            if let (Some((first, first_access)), Some(last)) = (first, last) {
                scratch_lifetimes.push(ScratchLifetime {
                    resource: resource.id,
                    first,
                    last,
                    first_access,
                });
            } else {
                add_dedicated(
                    resource,
                    ProgramAllocationKind::Scratch,
                    &mut allocations,
                    &mut resource_allocations,
                );
            }
        }
        scratch_lifetimes.sort_by_key(|lifetime| (lifetime.first, lifetime.last));

        let mut scratch_slots = Vec::<(ProgramAllocationId, usize)>::new();
        for lifetime in scratch_lifetimes {
            let resource = &self.resources[lifetime.resource.0];
            let can_alias = lifetime.first_access == BufferAccess::WriteOnly
                && matches!(
                    resource.initialization,
                    ProgramResourceInitialization::Zeroed
                );
            let reusable = if can_alias {
                scratch_slots
                    .iter()
                    .enumerate()
                    .filter(|(_, (allocation_id, last_pass))| {
                        if *last_pass >= lifetime.first {
                            return false;
                        }
                        allocations[allocation_id.0]
                            .resources
                            .first()
                            .map(|resource_id| self.resources[resource_id.0].element_shape())
                            == Some(resource.element_shape())
                    })
                    .min_by_key(|(_, (allocation_id, _))| {
                        let allocation = &allocations[allocation_id.0];
                        allocation.elements.max(resource.elements)
                    })
                    .map(|(index, _)| index)
            } else {
                None
            };

            if let Some(slot_index) = reusable {
                let (allocation_id, last_pass) = &mut scratch_slots[slot_index];
                let allocation = &mut allocations[allocation_id.0];
                allocation.elements = allocation.elements.max(resource.elements);
                allocation.resources.push(resource.id);
                resource_allocations[resource.id.0] = *allocation_id;
                *last_pass = lifetime.last;
            } else {
                let id = ProgramAllocationId(allocations.len());
                allocations.push(ProgramAllocation {
                    id,
                    kind: ProgramAllocationKind::Scratch,
                    scalar: resource.scalar,
                    elements: resource.elements,
                    resources: vec![resource.id],
                });
                resource_allocations[resource.id.0] = id;
                scratch_slots.push((id, lifetime.last));
            }
        }

        if resource_allocations.iter().any(|id| id.0 == usize::MAX) {
            return Err(VkFftError::InvalidKernelIr(
                "program memory planner left a logical resource unassigned",
            ));
        }
        Ok(ProgramMemoryPlan {
            allocations,
            resource_allocations,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.resources.is_empty() || self.passes.is_empty() {
            return Err(VkFftError::InvalidKernelIr(
                "program requires at least one resource and one pass",
            ));
        }
        let mut input_count = 0usize;
        let mut output_count = 0usize;
        for (index, resource) in self.resources.iter().enumerate() {
            let external_mixed_scalar = matches!(
                resource.kind,
                ProgramResourceKind::Input | ProgramResourceKind::Output
            ) && matches!(
                (self.scalar, resource.scalar),
                (ScalarType::F64, ScalarType::F32)
                    | (ScalarType::F32, ScalarType::F16)
                    | (ScalarType::DoubleDouble, ScalarType::F64)
            );
            let boundary_mixed_scratch = resource.kind == ProgramResourceKind::Scratch
                && self.scalar == ScalarType::DoubleDouble
                && resource.scalar == ScalarType::F64;
            if resource.id.0 != index
                || resource.elements == 0
                || (resource.scalar != self.scalar
                    && !external_mixed_scalar
                    && !boundary_mixed_scratch)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "program resources must have contiguous IDs, non-zero sizes, and a compute-compatible scalar type",
                ));
            }
            match resource.kind {
                ProgramResourceKind::Input => input_count += 1,
                ProgramResourceKind::Output => output_count += 1,
                ProgramResourceKind::Scratch | ProgramResourceKind::LookupTable => {}
            }
            if let Some(layout) = resource.external_layout {
                layout.validate()?;
                if layout.physical_elements()? != resource.elements {
                    return Err(VkFftError::InvalidKernelIr(
                        "program external resource layout does not cover its physical allocation",
                    ));
                }
            }
            match &resource.initialization {
                ProgramResourceInitialization::ExternalInput => {
                    if resource.kind != ProgramResourceKind::Input
                        || resource.external_layout.is_none()
                    {
                        return Err(VkFftError::InvalidKernelIr(
                            "program external input initialization requires an external input resource",
                        ));
                    }
                }
                ProgramResourceInitialization::Zeroed => {}
                ProgramResourceInitialization::Complex64(values) => {
                    if resource.kind != ProgramResourceKind::LookupTable
                        || values.len() != resource.elements
                        || values
                            .iter()
                            .any(|value| !value.re.is_finite() || !value.im.is_finite())
                    {
                        return Err(VkFftError::InvalidKernelIr(
                            "program LUT initialization does not match the resource",
                        ));
                    }
                }
                ProgramResourceInitialization::ComplexDoubleDouble(values) => {
                    if resource.kind != ProgramResourceKind::LookupTable
                        || resource.scalar != ScalarType::DoubleDouble
                        || values.len() != resource.elements
                        || values.iter().any(|value| {
                            !value.re.hi.is_finite()
                                || !value.re.lo.is_finite()
                                || !value.im.hi.is_finite()
                                || !value.im.lo.is_finite()
                        })
                    {
                        return Err(VkFftError::InvalidKernelIr(
                            "program double-double LUT initialization does not match the resource",
                        ));
                    }
                }
                ProgramResourceInitialization::StockhamUnitRoots { len } => {
                    if resource.kind != ProgramResourceKind::LookupTable
                        || *len == 0
                        || *len != resource.elements
                    {
                        return Err(VkFftError::InvalidKernelIr(
                            "program Stockham unit-root initialization does not match the resource",
                        ));
                    }
                }
            }
        }
        if input_count != 1 || output_count != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "program currently requires exactly one external input and one output resource",
            ));
        }
        let input_scalar = self.input_resource()?.scalar;
        let output_scalar = self.output_resource()?.scalar;
        let supports_external_scalar = |external_scalar| {
            external_scalar == self.scalar
                || matches!(
                    (self.scalar, external_scalar),
                    (ScalarType::F64, ScalarType::F32)
                        | (ScalarType::F32, ScalarType::F16)
                        | (ScalarType::DoubleDouble, ScalarType::F64)
                )
        };
        if !supports_external_scalar(input_scalar) || !supports_external_scalar(output_scalar) {
            return Err(VkFftError::InvalidKernelIr(
                "program external storage scalar is inconsistent with compute precision",
            ));
        }

        let input_id = self
            .resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Input)
            .map(|resource| resource.id)
            .expect("validated input resource");
        let output_id = self
            .resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Output)
            .map(|resource| resource.id)
            .expect("validated output resource");
        let mut reads_external_input = false;
        let mut writes_external_output = false;
        for pass in &self.passes {
            if pass.name.is_empty()
                || pass.dispatch.x == 0
                || pass.dispatch.y == 0
                || pass.dispatch.z == 0
                || pass.bindings.len() < 2
            {
                return Err(VkFftError::InvalidKernelIr(
                    "program pass metadata is incomplete",
                ));
            }
            let mut seen_bindings = Vec::with_capacity(pass.bindings.len());
            for pass_binding in &pass.bindings {
                if seen_bindings.contains(&pass_binding.binding) {
                    return Err(VkFftError::InvalidKernelIr(
                        "program pass has duplicate descriptor bindings",
                    ));
                }
                seen_bindings.push(pass_binding.binding);
                let resource = self.resource(pass_binding.resource)?;
                if matches!(
                    pass_binding.role,
                    BufferRole::LookupTable | BufferRole::TwiddleLookupTable
                ) && pass_binding.access != BufferAccess::ReadOnly
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "program immutable LUT bindings must be read-only",
                    ));
                }
                if pass_binding.role == BufferRole::Auxiliary
                    && !(pass_binding.access == BufferAccess::ReadOnly
                        || (pass_binding.access == BufferAccess::ReadWrite
                            && resource.kind == ProgramResourceKind::Scratch))
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "program auxiliary bindings must be read-only or read-write scratch",
                    ));
                }
                if pass_binding.resource == input_id
                    && pass_binding.access == BufferAccess::ReadOnly
                {
                    reads_external_input = true;
                }
                if pass_binding.resource == output_id
                    && matches!(
                        pass_binding.access,
                        BufferAccess::WriteOnly | BufferAccess::ReadWrite
                    )
                {
                    writes_external_output = true;
                }
                if resource.kind == ProgramResourceKind::LookupTable
                    && pass_binding.access != BufferAccess::ReadOnly
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "program LUT resources are immutable",
                    ));
                }
            }
        }
        if !reads_external_input || !writes_external_output {
            return Err(VkFftError::InvalidKernelIr(
                "program must read its external input and write its external output",
            ));
        }
        Ok(())
    }

    pub fn resource(&self, id: ProgramResourceId) -> Result<&ProgramResource> {
        self.resources
            .get(id.0)
            .filter(|resource| resource.id == id)
            .ok_or(VkFftError::InvalidKernelIr(
                "program pass references an unknown resource",
            ))
    }

    pub fn input_resource(&self) -> Result<&ProgramResource> {
        self.resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Input)
            .ok_or(VkFftError::InvalidKernelIr("program has no input resource"))
    }

    pub fn output_resource(&self) -> Result<&ProgramResource> {
        self.resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Output)
            .ok_or(VkFftError::InvalidKernelIr(
                "program has no output resource",
            ))
    }
}

fn add_double_double_recursive_scratch(
    resources: &mut Vec<ProgramResource>,
    serial: &mut usize,
    stem: &str,
    elements: usize,
) -> ProgramResourceId {
    *serial += 1;
    let id = ProgramResourceId(resources.len());
    resources.push(ProgramResource {
        id,
        name: format!("{stem}_{}", *serial),
        kind: ProgramResourceKind::Scratch,
        scalar: ScalarType::DoubleDouble,
        elements,
        external_layout: None,
        initialization: ProgramResourceInitialization::Zeroed,
    });
    id
}

fn append_double_double_recursive_child(
    child: &ProgramIr,
    prefix: &str,
    input: ProgramResourceId,
    output: ProgramResourceId,
    resources: &mut Vec<ProgramResource>,
    passes: &mut Vec<ProgramPass>,
    allow_output_storage_override: bool,
) -> Result<()> {
    let parent_input_scalar = resources
        .get(input.0)
        .ok_or(VkFftError::InvalidKernelIr(
            "double-double recursive child input resource is missing",
        ))?
        .scalar;
    let parent_output_scalar = resources
        .get(output.0)
        .ok_or(VkFftError::InvalidKernelIr(
            "double-double recursive child output resource is missing",
        ))?
        .scalar;
    let child_output_scalar = child.output_resource()?.scalar;
    let output_matches = child_output_scalar == parent_output_scalar
        || (allow_output_storage_override
            && child_output_scalar == ScalarType::DoubleDouble
            && matches!(
                parent_output_scalar,
                ScalarType::DoubleDouble | ScalarType::F64
            ));
    if child.scalar != ScalarType::DoubleDouble
        || child.input_resource()?.scalar != parent_input_scalar
        || !output_matches
    {
        return Err(VkFftError::InvalidKernelIr(
            "double-double recursive leaf ProgramIr boundary storage does not match parent resources",
        ));
    }
    let mut resource_map = vec![None; child.resources.len()];
    for resource in &child.resources {
        let mapped = match resource.kind {
            ProgramResourceKind::Input => input,
            ProgramResourceKind::Output => output,
            _ => {
                let id = ProgramResourceId(resources.len());
                let mut cloned = resource.clone();
                cloned.id = id;
                cloned.name = format!("{prefix}_{}", resource.name);
                cloned.external_layout = None;
                resources.push(cloned);
                id
            }
        };
        resource_map[resource.id.0] = Some(mapped);
    }
    for child_pass in &child.passes {
        let mut pass = child_pass.clone();
        pass.name = format!("{prefix}_{}", child_pass.name);
        for binding in &mut pass.bindings {
            binding.resource = resource_map
                .get(binding.resource.0)
                .and_then(|mapped| *mapped)
                .ok_or(VkFftError::InvalidKernelIr(
                    "double-double recursive child resource mapping is incomplete",
                ))?;
        }
        passes.push(pass);
    }
    Ok(())
}

fn flatten_double_double_recursive_node(
    node: &DoubleDoubleRecursiveFftNodeIr,
    input: ProgramResourceId,
    output: ProgramResourceId,
    resources: &mut Vec<ProgramResource>,
    passes: &mut Vec<ProgramPass>,
    serial: &mut usize,
) -> Result<()> {
    node.validate()?;
    match node {
        DoubleDoubleRecursiveFftNodeIr::Stockham(ir) => {
            let child = ProgramIr::double_double_stockham(ir)?;
            *serial += 1;
            append_double_double_recursive_child(
                &child,
                &format!("vkfft_dd_recursive_leaf_{}", *serial),
                input,
                output,
                resources,
                passes,
                false,
            )?;
        }
        DoubleDoubleRecursiveFftNodeIr::DirectRader(ir) => {
            let child = ProgramIr::double_double_direct_rader(ir)?;
            *serial += 1;
            append_double_double_recursive_child(
                &child,
                &format!("vkfft_dd_recursive_leaf_{}", *serial),
                input,
                output,
                resources,
                passes,
                false,
            )?;
        }
        DoubleDoubleRecursiveFftNodeIr::FftRader(ir) => {
            let child = ProgramIr::double_double_fft_rader(ir)?;
            *serial += 1;
            append_double_double_recursive_child(
                &child,
                &format!("vkfft_dd_recursive_leaf_{}", *serial),
                input,
                output,
                resources,
                passes,
                false,
            )?;
        }
        DoubleDoubleRecursiveFftNodeIr::Bluestein(ir) => {
            let child = ProgramIr::double_double_bluestein(ir)?;
            *serial += 1;
            append_double_double_recursive_child(
                &child,
                &format!("vkfft_dd_recursive_leaf_{}", *serial),
                input,
                output,
                resources,
                passes,
                false,
            )?;
        }
        DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => {
            if let Some(fused) = ir.fused_small_direct_rader_stockham()? {
                let direct_lut_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: direct_lut_id,
                    name: "double_double_composite_direct_rader_roots".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: fused.direct.table.twiddles_by_generator_power.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        fused.direct.table.twiddles_by_generator_power.clone(),
                    ),
                });
                let parent_roots_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: parent_roots_id,
                    name: "double_double_composite_parent_roots".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: ir.twiddles.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        ir.twiddles.clone(),
                    ),
                });
                passes.push(ProgramPass {
                    name: fused.name(),
                    dispatch: ir.pack_right.dispatch,
                    bindings: vec![
                        binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                        ProgramPassBinding {
                            binding: 2,
                            resource: direct_lut_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 3,
                            resource: parent_roots_id,
                            role: BufferRole::TwiddleLookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ],
                });
                return Ok(());
            }
            if let Some(fused) = ir.fused_small_fft_rader_stockham()? {
                let spectrum_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: spectrum_id,
                    name: "double_double_composite_fft_rader_spectrum".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: fused.rader.kernel_spectrum.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        fused.rader.kernel_spectrum.clone(),
                    ),
                });
                let parent_roots_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: parent_roots_id,
                    name: "double_double_composite_fft_rader_parent_roots".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: ir.twiddles.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        ir.twiddles.clone(),
                    ),
                });
                passes.push(ProgramPass {
                    name: fused.name(),
                    dispatch: ir.pack_right.dispatch,
                    bindings: vec![
                        binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                        ProgramPassBinding {
                            binding: 2,
                            resource: spectrum_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 3,
                            resource: parent_roots_id,
                            role: BufferRole::TwiddleLookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ],
                });
                return Ok(());
            }
            if let Some(fused) = ir.fused_dual_fft_rader_supported()? {
                let right_spectrum_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: right_spectrum_id,
                    name: "double_double_composite_dual_fft_rader_right_spectrum".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: fused.right.kernel_spectrum.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        fused.right.kernel_spectrum.clone(),
                    ),
                });
                let left_spectrum_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: left_spectrum_id,
                    name: "double_double_composite_dual_fft_rader_left_spectrum".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: fused.left.kernel_spectrum.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        fused.left.kernel_spectrum.clone(),
                    ),
                });
                let parent_roots_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: parent_roots_id,
                    name: "double_double_composite_dual_fft_rader_parent_roots".to_owned(),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: ir.twiddles.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                        ir.twiddles.clone(),
                    ),
                });
                passes.push(ProgramPass {
                    name: fused.name(),
                    dispatch: ir.pack_right.dispatch,
                    bindings: vec![
                        binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                        ProgramPassBinding {
                            binding: 2,
                            resource: right_spectrum_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 3,
                            resource: left_spectrum_id,
                            role: BufferRole::LookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 4,
                            resource: parent_roots_id,
                            role: BufferRole::TwiddleLookupTable,
                            access: BufferAccess::ReadOnly,
                        },
                    ],
                });
                return Ok(());
            }
            let elements = ir.logical_len.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double recursive Cooley-Tukey program element count",
                },
            )?;
            let right_input = add_double_double_recursive_scratch(
                resources,
                serial,
                "vkfft_dd_recursive_right_input",
                elements,
            );
            let right_output = add_double_double_recursive_scratch(
                resources,
                serial,
                "vkfft_dd_recursive_right_output",
                elements,
            );
            let left_input = add_double_double_recursive_scratch(
                resources,
                serial,
                "vkfft_dd_recursive_left_input",
                elements,
            );
            let left_output = add_double_double_recursive_scratch(
                resources,
                serial,
                "vkfft_dd_recursive_left_output",
                elements,
            );
            passes.push(two_buffer_pass(
                &ir.pack_right.name,
                ir.pack_right.dispatch,
                input.0,
                right_input.0,
            ));
            flatten_double_double_recursive_node(
                &ir.right,
                right_input,
                right_output,
                resources,
                passes,
                serial,
            )?;
            *serial += 1;
            let twiddle_id = ProgramResourceId(resources.len());
            resources.push(ProgramResource {
                id: twiddle_id,
                name: format!("vkfft_dd_recursive_twiddles_{}", *serial),
                kind: ProgramResourceKind::LookupTable,
                scalar: ScalarType::DoubleDouble,
                elements: ir.twiddles.len(),
                external_layout: None,
                initialization: ProgramResourceInitialization::ComplexDoubleDouble(
                    ir.twiddles.clone(),
                ),
            });
            passes.push(ProgramPass {
                name: ir.twiddle_transpose.name.clone(),
                dispatch: ir.twiddle_transpose.dispatch,
                bindings: vec![
                    binding(0, right_output.0, BufferRole::Input, BufferAccess::ReadOnly),
                    binding(1, left_input.0, BufferRole::Output, BufferAccess::WriteOnly),
                    ProgramPassBinding {
                        binding: 2,
                        resource: twiddle_id,
                        role: BufferRole::TwiddleLookupTable,
                        access: BufferAccess::ReadOnly,
                    },
                ],
            });
            flatten_double_double_recursive_node(
                &ir.left,
                left_input,
                left_output,
                resources,
                passes,
                serial,
            )?;
            let mut scatter_bindings = vec![
                binding(0, left_output.0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
            ];
            let mapped_twiddle_period = match ir.scatter_output.output_modifier {
                DoubleDoubleCooleyTukeyOutputModifier::FourStepRight(mapping) => {
                    Some(mapping.logical_len)
                }
                DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload2(mapping) => {
                    Some(mapping.logical_len)
                }
                DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload1(mapping) => {
                    let [a, b, _] = mapping.axis_split;
                    Some(a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double recursive three-upload-1 twiddle period",
                    })?)
                }
                _ => None,
            };
            if let Some(period) = mapped_twiddle_period {
                let values = crate::lut::unit_root_table_double_double(period, ir.direction)?;
                let twiddle_id = ProgramResourceId(resources.len());
                resources.push(ProgramResource {
                    id: twiddle_id,
                    name: format!("vkfft_dd_recursive_four_step_twiddles_{}", *serial),
                    kind: ProgramResourceKind::LookupTable,
                    scalar: ScalarType::DoubleDouble,
                    elements: values.len(),
                    external_layout: None,
                    initialization: ProgramResourceInitialization::ComplexDoubleDouble(values),
                });
                scatter_bindings.push(ProgramPassBinding {
                    binding: 2,
                    resource: twiddle_id,
                    role: BufferRole::TwiddleLookupTable,
                    access: BufferAccess::ReadOnly,
                });
            }
            passes.push(ProgramPass {
                name: ir.scatter_output.name.clone(),
                dispatch: ir.scatter_output.dispatch,
                bindings: scatter_bindings,
            });
        }
    }
    Ok(())
}

struct RecursiveProgramBuilder {
    scalar: ScalarType,
    resources: Vec<ProgramResource>,
    passes: Vec<ProgramPass>,
    serial: usize,
}

impl RecursiveProgramBuilder {
    fn add_scratch(&mut self, stem: &str, elements: usize) -> ProgramResourceId {
        let id = ProgramResourceId(self.resources.len());
        self.serial += 1;
        self.resources.push(ProgramResource {
            id,
            name: format!("{stem}_{}", self.serial),
            kind: ProgramResourceKind::Scratch,
            scalar: self.scalar,
            elements,
            external_layout: None,
            initialization: ProgramResourceInitialization::Zeroed,
        });
        id
    }

    fn add_lut(&mut self, stem: &str, values: Vec<Complex64>) -> ProgramResourceId {
        if let Some(existing) = self.resources.iter().find(|resource| {
            resource.kind == ProgramResourceKind::LookupTable
                && resource.scalar == self.scalar
                && matches!(
                    &resource.initialization,
                    ProgramResourceInitialization::Complex64(existing_values)
                        if existing_values == &values
                )
        }) {
            return existing.id;
        }
        let id = ProgramResourceId(self.resources.len());
        self.serial += 1;
        self.resources.push(ProgramResource {
            id,
            name: format!("{stem}_{}", self.serial),
            kind: ProgramResourceKind::LookupTable,
            scalar: self.scalar,
            elements: values.len(),
            external_layout: None,
            initialization: ProgramResourceInitialization::Complex64(values),
        });
        id
    }

    fn add_stockham_root_lut(&mut self, len: usize) -> Result<ProgramResourceId> {
        if len == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham unit-root LUT requires a non-zero period",
            ));
        }
        let initialization = ProgramResourceInitialization::StockhamUnitRoots { len };
        if let Some(existing) = self.resources.iter().find(|resource| {
            resource.kind == ProgramResourceKind::LookupTable
                && resource.scalar == self.scalar
                && resource.elements == len
                && resource.initialization == initialization
        }) {
            return Ok(existing.id);
        }
        let id = ProgramResourceId(self.resources.len());
        self.serial += 1;
        self.resources.push(ProgramResource {
            id,
            name: format!("stockham_unit_roots_{}", self.serial),
            kind: ProgramResourceKind::LookupTable,
            scalar: self.scalar,
            elements: len,
            external_layout: None,
            initialization,
        });
        Ok(id)
    }

    fn stockham_pass_bindings(
        &mut self,
        kernel: &KernelIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
        lookup: Option<ProgramResourceId>,
        auxiliary: Option<ProgramResourceId>,
    ) -> Result<Vec<ProgramPassBinding>> {
        let twiddle = match kernel.twiddle_lut_len() {
            Some(len) => Some(self.add_stockham_root_lut(len)?),
            None => None,
        };
        kernel
            .bindings
            .iter()
            .map(|descriptor| {
                let resource = match descriptor.role {
                    BufferRole::Input => input,
                    BufferRole::Output => output,
                    BufferRole::LookupTable => lookup.ok_or(VkFftError::InvalidKernelIr(
                        "Stockham pass requires a lookup-table resource",
                    ))?,
                    BufferRole::TwiddleLookupTable => {
                        twiddle.ok_or(VkFftError::InvalidKernelIr(
                            "Stockham pass requires a twiddle-table resource",
                        ))?
                    }
                    BufferRole::Auxiliary => auxiliary.ok_or(VkFftError::InvalidKernelIr(
                        "Stockham pass requires an auxiliary resource",
                    ))?,
                };
                Ok(binding(
                    descriptor.binding,
                    resource.0,
                    descriptor.role,
                    descriptor.access,
                ))
            })
            .collect()
    }

    fn push_stockham_pass(
        &mut self,
        kernel: &KernelIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
        lookup: Option<ProgramResourceId>,
        auxiliary: Option<ProgramResourceId>,
    ) -> Result<()> {
        let bindings = self.stockham_pass_bindings(kernel, input, output, lookup, auxiliary)?;
        self.passes.push(ProgramPass {
            name: kernel.name.clone(),
            dispatch: kernel.dispatch,
            bindings,
        });
        Ok(())
    }

    fn flatten_bluestein(
        &mut self,
        pipeline: &BluesteinPipelineIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        pipeline.validate()?;
        let convolution_elements = pipeline
            .convolution_len
            .checked_mul(pipeline.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "generic program Bluestein convolution element count",
            })?;
        let scratch_a = self.add_scratch("bluestein_scratch_a", convolution_elements);
        let scratch_b = self.add_scratch("bluestein_scratch_b", convolution_elements);
        let lut = self.add_lut(
            "bluestein_kernel_spectrum",
            pipeline.kernel_spectrum()?.to_vec(),
        );
        self.passes.push(two_buffer_pass(
            &pipeline.preprocess.name,
            pipeline.preprocess.dispatch,
            input.0,
            scratch_a.0,
        ));
        self.flatten_recursive_ir(&pipeline.forward_fft, scratch_a, scratch_b)?;
        let convolution_output =
            if let Some(fused_inverse) = pipeline.fused_inverse_stockham_kernel()? {
                self.push_stockham_pass(&fused_inverse, scratch_b, scratch_a, Some(lut), None)?;
                scratch_a
            } else if pipeline.has_fused_recursive_inverse() {
                self.flatten_node_with_boundary_resources(
                    &pipeline.inverse_fft.root,
                    scratch_b,
                    scratch_a,
                    Some(lut),
                    None,
                )?;
                scratch_a
            } else {
                self.passes.push(ProgramPass {
                    name: pipeline.multiply.name.clone(),
                    dispatch: pipeline.multiply.dispatch,
                    bindings: vec![
                        binding(0, scratch_b.0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, scratch_a.0, BufferRole::Output, BufferAccess::WriteOnly),
                        binding(2, lut.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    ],
                });
                self.flatten_recursive_ir(&pipeline.inverse_fft, scratch_a, scratch_b)?;
                scratch_b
            };
        self.passes.push(two_buffer_pass(
            &pipeline.postprocess.name,
            pipeline.postprocess.dispatch,
            convolution_output.0,
            output.0,
        ));
        Ok(())
    }

    fn push_zero_pad_pass(
        &mut self,
        pass: &crate::ZeroPadPassIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        pass.validate()?;
        self.passes.push(ProgramPass {
            name: pass.name.clone(),
            dispatch: pass.dispatch,
            bindings: vec![
                binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
            ],
        });
        Ok(())
    }

    fn push_nd_zero_pad_pass(
        &mut self,
        pass: &crate::NdZeroPadPassIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        pass.validate()?;
        self.passes.push(ProgramPass {
            name: pass.name.clone(),
            dispatch: pass.dispatch,
            bindings: vec![
                binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
            ],
        });
        Ok(())
    }

    fn flatten_recursive_ir(
        &mut self,
        ir: &RecursiveFftIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        if let Some(uploads) = ir.four_step_stockham_upload_kernels()? {
            let elements = ir.logical_len.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "recursive Four-step program element count",
                },
            )?;
            let scratch = self.add_scratch("four_step_exchange", elements);
            match uploads.as_slice() {
                [first, second] => {
                    self.push_stockham_pass(first, input, scratch, None, None)?;
                    self.push_stockham_pass(second, scratch, output, None, None)?;
                }
                [first, second, third] => {
                    let output_scalar = self
                        .resources
                        .get(output.0)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "recursive Four-step output resource is missing",
                        ))?
                        .scalar;
                    if output_scalar == ir.scalar {
                        self.push_stockham_pass(first, input, output, None, None)?;
                        self.push_stockham_pass(second, output, scratch, None, None)?;
                        self.push_stockham_pass(third, scratch, output, None, None)?;
                    } else {
                        let scratch_b = self.add_scratch("four_step_exchange_b", elements);
                        self.push_stockham_pass(first, input, scratch, None, None)?;
                        self.push_stockham_pass(second, scratch, scratch_b, None, None)?;
                        self.push_stockham_pass(third, scratch_b, output, None, None)?;
                    }
                }
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "Four-step program currently requires exactly two or three kernels",
                    ));
                }
            }
            Ok(())
        } else if let Some(uploads) = ir.four_step_rader_upload_nodes()? {
            let elements = ir.logical_len.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "recursive Rader Four-step program element count",
                },
            )?;
            match uploads.as_slice() {
                [first, second] => {
                    let scratch = self.add_scratch("rader_four_step_exchange", elements);
                    self.flatten_node(first, input, scratch)?;
                    self.flatten_node(second, scratch, output)
                }
                [first, second, third] => {
                    let output_scalar = self
                        .resources
                        .get(output.0)
                        .ok_or(VkFftError::InvalidKernelIr(
                            "recursive Rader Four-step output resource is missing",
                        ))?
                        .scalar;
                    if output_scalar == ir.scalar {
                        let scratch = self.add_scratch("rader_four_step_exchange", elements);
                        self.flatten_node(first, input, output)?;
                        self.flatten_node(second, output, scratch)?;
                        self.flatten_node(third, scratch, output)
                    } else {
                        let scratch_a = self.add_scratch("rader_four_step_exchange_a", elements);
                        let scratch_b = self.add_scratch("rader_four_step_exchange_b", elements);
                        self.flatten_node(first, input, scratch_a)?;
                        self.flatten_node(second, scratch_a, scratch_b)?;
                        self.flatten_node(third, scratch_b, output)
                    }
                }
                _ => Err(VkFftError::InvalidKernelIr(
                    "Rader Four-step program currently requires exactly two or three upload nodes",
                )),
            }
        } else {
            self.flatten_node(&ir.root, input, output)
        }
    }

    fn flatten_one_dim(
        &mut self,
        ir: &OneDimFftIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        let Some(zero_pad) = ir.zero_pad_pass() else {
            return match ir {
                OneDimFftIr::Recursive(recursive) => {
                    self.flatten_recursive_ir(recursive, input, output)
                }
                OneDimFftIr::Bluestein(pipeline) => self.flatten_bluestein(pipeline, input, output),
            };
        };
        let elements = ir.logical_len().checked_mul(ir.batch_count()).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "zero-padded one-dimensional program element count",
            },
        )?;
        let boundary = self.add_scratch("zero_pad_boundary", elements);
        if zero_pad.operation.is_input_boundary() {
            self.push_zero_pad_pass(zero_pad, input, boundary)?;
            match ir {
                OneDimFftIr::Recursive(recursive) => {
                    self.flatten_recursive_ir(recursive, boundary, output)
                }
                OneDimFftIr::Bluestein(pipeline) => {
                    self.flatten_bluestein(pipeline, boundary, output)
                }
            }
        } else {
            match ir {
                OneDimFftIr::Recursive(recursive) => {
                    self.flatten_recursive_ir(recursive, input, boundary)?
                }
                OneDimFftIr::Bluestein(pipeline) => {
                    self.flatten_bluestein(pipeline, input, boundary)?
                }
            }
            self.push_zero_pad_pass(zero_pad, boundary, output)
        }
    }

    fn flatten_r2r(
        &mut self,
        ir: &R2rIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        ir.validate()?;
        if let Some(reduction) = ir.fft_reduction.as_deref() {
            let fft_elements = reduction.fft_len.checked_mul(reduction.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "R2R FFT reduction program element count",
                },
            )?;
            let fft_input = self.add_scratch("r2r_fft_input", fft_elements);
            let fft_output = self.add_scratch("r2r_fft_output", fft_elements);
            self.passes.push(two_buffer_pass(
                &reduction.preprocess.name,
                reduction.preprocess.dispatch,
                input.0,
                fft_input.0,
            ));
            self.flatten_one_dim(&reduction.fft, fft_input, fft_output)?;
            self.passes.push(two_buffer_pass(
                &reduction.postprocess.name,
                reduction.postprocess.dispatch,
                fft_output.0,
                output.0,
            ));
        } else {
            self.passes.push(two_buffer_pass(
                &format!("vkfft_r2r_{:?}_{:?}", ir.effective_transform, ir.direction),
                ir.dispatch,
                input.0,
                output.0,
            ));
        }
        Ok(())
    }

    fn flatten_real_fft(
        &mut self,
        ir: &RealFftIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        ir.validate()?;
        if let Some(fused) = ir.fused_even_input_stockham_kernel()? {
            let transform_elements = ir.transform_len().checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "fused real FFT internal transform element count",
                },
            )?;
            let fuses_postprocess = matches!(
                fused.output_modifier,
                crate::StockhamOutputModifier::RealEvenPostprocess(_)
                    | crate::StockhamOutputModifier::RealEvenUnpack(_)
            );
            let transform_output = if ir.postprocess.is_some() && !fuses_postprocess {
                self.add_scratch("real_transform", transform_elements)
            } else {
                output
            };
            let auxiliary = if matches!(
                fused.output_modifier,
                crate::StockhamOutputModifier::RealEvenPostprocess(_)
            ) {
                Some(self.add_scratch("real_postprocess_scratch", transform_elements))
            } else {
                None
            };
            self.push_stockham_pass(&fused, input, transform_output, None, auxiliary)?;
            if !fuses_postprocess && let Some(postprocess) = &ir.postprocess {
                self.passes.push(two_buffer_pass(
                    &postprocess.name,
                    postprocess.dispatch,
                    transform_output.0,
                    output.0,
                ));
            }
            return Ok(());
        }
        let transform_elements = ir.transform_len().checked_mul(ir.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "real FFT program internal transform element count",
            },
        )?;
        if let Some(fused_recursive) = ir.fused_even_recursive_ir()? {
            self.flatten_node_with_boundary_resources(
                &fused_recursive.root,
                input,
                output,
                None,
                None,
            )?;
            return Ok(());
        }
        if let Some(fused_bluestein) = ir.fused_even_bluestein_ir()? {
            self.flatten_bluestein(&fused_bluestein, input, output)?;
            return Ok(());
        }

        let transform_input = if let Some(preprocess) = &ir.preprocess {
            let scratch = self.add_scratch("real_preprocess", transform_elements);
            self.passes.push(two_buffer_pass(
                &preprocess.name,
                preprocess.dispatch,
                input.0,
                scratch.0,
            ));
            scratch
        } else {
            input
        };
        let transform_output = if ir.postprocess.is_some() {
            self.add_scratch("real_transform", transform_elements)
        } else {
            output
        };
        self.flatten_one_dim(&ir.transform, transform_input, transform_output)?;
        if let Some(postprocess) = &ir.postprocess {
            self.passes.push(two_buffer_pass(
                &postprocess.name,
                postprocess.dispatch,
                transform_output.0,
                output.0,
            ));
        }
        Ok(())
    }

    fn flatten_node_with_boundary_resources(
        &mut self,
        node: &RecursiveFftNodeIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
        lookup: Option<ProgramResourceId>,
        auxiliary: Option<ProgramResourceId>,
    ) -> Result<()> {
        node.validate()?;
        let RecursiveFftNodeIr::CooleyTukey(ir) = node else {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive boundary resources require a Cooley-Tukey root",
            ));
        };
        let elements =
            ir.logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive program fused Cooley-Tukey element count",
                })?;
        let right_input = self.add_scratch("recursive_right_input", elements);
        let right_output = self.add_scratch("recursive_right_output", elements);
        let left_input = self.add_scratch("recursive_left_input", elements);
        let left_output = self.add_scratch("recursive_left_output", elements);

        let mut pack_bindings = vec![
            binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(
                1,
                right_input.0,
                BufferRole::Output,
                BufferAccess::WriteOnly,
            ),
        ];
        if ir.pack_right.input_modifier == CooleyTukeyInputModifier::MultiplyLookupTable {
            let lookup = lookup.ok_or(VkFftError::InvalidKernelIr(
                "recursive fused Cooley-Tukey pack requires a lookup resource",
            ))?;
            pack_bindings.push(binding(
                2,
                lookup.0,
                BufferRole::LookupTable,
                BufferAccess::ReadOnly,
            ));
        }
        self.passes.push(ProgramPass {
            name: ir.pack_right.name.clone(),
            dispatch: ir.pack_right.dispatch,
            bindings: pack_bindings,
        });
        self.flatten_node(&ir.right, right_input, right_output)?;
        self.passes.push(two_buffer_pass(
            &ir.twiddle_transpose.name,
            ir.twiddle_transpose.dispatch,
            right_output.0,
            left_input.0,
        ));
        self.flatten_node(&ir.left, left_input, left_output)?;

        let mut scatter_bindings = vec![
            binding(0, left_output.0, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
        ];
        if matches!(
            ir.scatter_output.output_modifier,
            CooleyTukeyOutputModifier::RaderScatter(_)
        ) {
            let auxiliary = auxiliary.ok_or(VkFftError::InvalidKernelIr(
                "recursive fused Cooley-Tukey scatter requires an auxiliary resource",
            ))?;
            scatter_bindings.push(binding(
                2,
                auxiliary.0,
                BufferRole::Auxiliary,
                BufferAccess::ReadOnly,
            ));
        }
        self.passes.push(ProgramPass {
            name: ir.scatter_output.name.clone(),
            dispatch: ir.scatter_output.dispatch,
            bindings: scatter_bindings,
        });
        Ok(())
    }

    fn flatten_node(
        &mut self,
        node: &RecursiveFftNodeIr,
        input: ProgramResourceId,
        output: ProgramResourceId,
    ) -> Result<()> {
        node.validate()?;
        match node {
            RecursiveFftNodeIr::Stockham(kernel) => {
                self.push_stockham_pass(kernel, input, output, None, None)?;
            }
            RecursiveFftNodeIr::DirectRader(direct) => {
                let lut = self.add_lut("rader_roots", direct.twiddle_lut().to_vec());
                self.passes.push(ProgramPass {
                    name: direct.name.clone(),
                    dispatch: direct.dispatch,
                    bindings: vec![
                        binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                        binding(2, lut.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                    ],
                });
            }
            RecursiveFftNodeIr::FftRader(rader) => {
                let convolution_elements = rader
                    .convolution_len
                    .checked_mul(rader.batch_count)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "recursive program Rader convolution element count",
                    })?;
                let scratch_a = self.add_scratch("rader_scratch_a", convolution_elements);
                let scratch_b = self.add_scratch("rader_scratch_b", convolution_elements);
                let lut = self.add_lut("rader_kernel_spectrum", rader.kernel_spectrum()?.to_vec());
                let forward_input = match rader.input_strategy {
                    RaderFftInputStrategy::GatherReversePass => {
                        self.passes.push(two_buffer_pass(
                            &rader.gather.name,
                            rader.gather.dispatch,
                            input.0,
                            scratch_a.0,
                        ));
                        scratch_a
                    }
                    RaderFftInputStrategy::GeneratorOrderStockham
                    | RaderFftInputStrategy::GeneratorOrderRecursive => input,
                };
                self.flatten_one_dim(&rader.forward_fft, forward_input, scratch_b)?;
                if let Some(fused_inverse) = rader.fused_inverse_rader_kernel()? {
                    self.push_stockham_pass(
                        &fused_inverse,
                        scratch_b,
                        output,
                        Some(lut),
                        Some(input),
                    )?;
                } else if rader.has_fused_recursive_inverse() {
                    let inverse = rader
                        .inverse_recursive()
                        .ok_or(VkFftError::InvalidKernelIr(
                            "recursive fused Rader program lost its recursive inverse child",
                        ))?;
                    self.flatten_node_with_boundary_resources(
                        &inverse.root,
                        scratch_b,
                        output,
                        Some(lut),
                        Some(input),
                    )?;
                } else {
                    self.passes.push(ProgramPass {
                        name: rader.multiply.name.clone(),
                        dispatch: rader.multiply.dispatch,
                        bindings: vec![
                            binding(0, scratch_b.0, BufferRole::Input, BufferAccess::ReadOnly),
                            binding(1, scratch_a.0, BufferRole::Output, BufferAccess::WriteOnly),
                            binding(2, lut.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                        ],
                    });
                    self.flatten_one_dim(&rader.inverse_fft, scratch_a, scratch_b)?;
                    self.passes.push(ProgramPass {
                        name: rader.scatter.name.clone(),
                        dispatch: rader.scatter.dispatch,
                        bindings: vec![
                            binding(0, scratch_b.0, BufferRole::Input, BufferAccess::ReadOnly),
                            binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                            binding(2, input.0, BufferRole::Auxiliary, BufferAccess::ReadOnly),
                        ],
                    });
                }
            }
            RecursiveFftNodeIr::CooleyTukey(ir) => {
                if let Some(fused) = ir.fused_small_direct_rader_stockham()? {
                    let lut = self.add_lut(
                        "composite_direct_rader_roots",
                        fused.direct.twiddle_lut().to_vec(),
                    );
                    self.passes.push(ProgramPass {
                        name: fused.name(),
                        dispatch: ir.pack_right.dispatch,
                        bindings: vec![
                            binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                            binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                            binding(2, lut.0, BufferRole::LookupTable, BufferAccess::ReadOnly),
                        ],
                    });
                    return Ok(());
                }
                if let Some(fused) = ir.fused_small_fft_rader_stockham()? {
                    let spectrum = self.add_lut(
                        "composite_fft_rader_spectrum",
                        fused.rader.kernel_spectrum()?.to_vec(),
                    );
                    let mut bindings = vec![
                        binding(0, input.0, BufferRole::Input, BufferAccess::ReadOnly),
                        binding(1, output.0, BufferRole::Output, BufferAccess::WriteOnly),
                        binding(
                            2,
                            spectrum.0,
                            BufferRole::LookupTable,
                            BufferAccess::ReadOnly,
                        ),
                    ];
                    if let Some(len) = fused.forward.twiddle_lut_len() {
                        let twiddle = self.add_stockham_root_lut(len)?;
                        bindings.push(binding(
                            3,
                            twiddle.0,
                            BufferRole::TwiddleLookupTable,
                            BufferAccess::ReadOnly,
                        ));
                    }
                    self.passes.push(ProgramPass {
                        name: fused.name(),
                        dispatch: ir.pack_right.dispatch,
                        bindings,
                    });
                    return Ok(());
                }
                if let Some((mapped_right, mapped_left)) = ir.fused_fft_rader_cooley_boundaries()? {
                    let elements = ir.logical_len.checked_mul(ir.batch_count).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "recursive program fused Cooley-right element count",
                        },
                    )?;
                    let right_output = self.add_scratch("recursive_right_output", elements);
                    let mapped_right = RecursiveFftNodeIr::FftRader(Box::new(mapped_right));
                    self.flatten_node(&mapped_right, input, right_output)?;
                    self.push_stockham_pass(&mapped_left, right_output, output, None, None)?;
                    return Ok(());
                }
                if let Some(mapped_left) = ir.fused_fft_rader_left_stockham()? {
                    let elements = ir.logical_len.checked_mul(ir.batch_count).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "recursive program fused Cooley-left element count",
                        },
                    )?;
                    let right_input = self.add_scratch("recursive_right_input", elements);
                    let right_output = self.add_scratch("recursive_right_output", elements);
                    self.passes.push(two_buffer_pass(
                        &ir.pack_right.name,
                        ir.pack_right.dispatch,
                        input.0,
                        right_input.0,
                    ));
                    self.flatten_node(&ir.right, right_input, right_output)?;
                    self.push_stockham_pass(&mapped_left, right_output, output, None, None)?;
                    return Ok(());
                }
                if ir.pack_right.input_modifier == CooleyTukeyInputModifier::MultiplyLookupTable
                    || !matches!(
                        ir.scatter_output.output_modifier,
                        CooleyTukeyOutputModifier::None
                            | CooleyTukeyOutputModifier::FourStepRight(_)
                            | CooleyTukeyOutputModifier::FourStepLeft(_)
                            | CooleyTukeyOutputModifier::FourStepThreeUpload2(_)
                            | CooleyTukeyOutputModifier::FourStepThreeUpload1(_)
                            | CooleyTukeyOutputModifier::FourStepThreeUpload0(_)
                    )
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "modified recursive Cooley-Tukey boundaries require explicit resources",
                    ));
                }
                let elements = ir.logical_len.checked_mul(ir.batch_count).ok_or(
                    VkFftError::ArithmeticOverflow {
                        operation: "recursive program Cooley-Tukey element count",
                    },
                )?;
                let right_input = self.add_scratch("recursive_right_input", elements);
                let right_output = self.add_scratch("recursive_right_output", elements);
                let left_input = self.add_scratch("recursive_left_input", elements);
                let left_output = self.add_scratch("recursive_left_output", elements);
                self.passes.push(two_buffer_pass(
                    &ir.pack_right.name,
                    ir.pack_right.dispatch,
                    input.0,
                    right_input.0,
                ));
                self.flatten_node(&ir.right, right_input, right_output)?;
                self.passes.push(two_buffer_pass(
                    &ir.twiddle_transpose.name,
                    ir.twiddle_transpose.dispatch,
                    right_output.0,
                    left_input.0,
                ));
                self.flatten_node(&ir.left, left_input, left_output)?;
                self.passes.push(two_buffer_pass(
                    &ir.scatter_output.name,
                    ir.scatter_output.dispatch,
                    left_output.0,
                    output.0,
                ));
            }
        }
        Ok(())
    }
}

fn double_double_stockham_input_lookup_eligible(ir: &DoubleDoubleStockhamIr) -> bool {
    ir.sequence_len > 64
        && ir.external_storage == crate::PrecisionStorage::DoubleDouble
        && ir.zero_pad_pass.is_none()
        && !ir.stages.is_empty()
}

fn attach_double_double_stockham_input_lookup(
    pass: &mut ProgramPass,
    lookup: ProgramResourceId,
) -> Result<()> {
    if pass.bindings.iter().any(|binding| binding.binding == 3) {
        return Err(VkFftError::InvalidKernelIr(
            "fused DD Stockham input lookup binding 3 is already occupied",
        ));
    }
    pass.bindings.push(ProgramPassBinding {
        binding: 3,
        resource: lookup,
        role: BufferRole::LookupTable,
        access: BufferAccess::ReadOnly,
    });
    pass.name.push_str("_input_mul_lut");
    Ok(())
}

fn binding(
    binding: u32,
    resource: usize,
    role: BufferRole,
    access: BufferAccess,
) -> ProgramPassBinding {
    ProgramPassBinding {
        binding,
        resource: ProgramResourceId(resource),
        role,
        access,
    }
}

fn two_buffer_pass(
    name: &str,
    dispatch: DispatchGeometry,
    input: usize,
    output: usize,
) -> ProgramPass {
    ProgramPass {
        name: name.to_owned(),
        dispatch,
        bindings: vec![
            binding(0, input, BufferRole::Input, BufferAccess::ReadOnly),
            binding(1, output, BufferRole::Output, BufferAccess::WriteOnly),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        Backend, DeviceProfile, Direction, GpuVendor, Precision, TransformKind, ZeroPaddingDomain,
    };
    use crate::{FftConfig, FftPlan};

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    #[test]
    fn double_double_even_real_program_uses_half_size_child_and_root_lut() {
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for transform in [TransformKind::RealToComplex, TransformKind::ComplexToReal] {
                let plan = FftPlan::build(
                    FftConfig::new(vec![16])
                        .with_batch_count(2)
                        .with_transform(transform)
                        .with_precision(precision)
                        .with_inverse_normalization(transform == TransformKind::ComplexToReal),
                )
                .unwrap();
                let ir = crate::DoubleDoubleRealFftIr::build(&plan).unwrap();
                assert!(ir.even_half_size);
                assert_eq!(ir.transform.sequence_len(), 8);
                assert_eq!(ir.even_roots.len(), 9);
                let program = ProgramIr::double_double_real(&ir).unwrap();
                assert_eq!(program.resources[2].elements, 16);
                assert_eq!(program.resources[3].elements, 16);
                let roots = program
                    .resources
                    .iter()
                    .find(|resource| resource.name == "double_double_real_even_roots")
                    .expect("even DD real ProgramIr must own a root LUT");
                assert_eq!(roots.scalar, ScalarType::DoubleDouble);
                assert_eq!(roots.elements, 9);
                let ProgramResourceInitialization::ComplexDoubleDouble(values) =
                    &roots.initialization
                else {
                    panic!("even DD real roots must preserve DD words");
                };
                assert_eq!(values, &ir.even_roots);
                let first = program.passes.first().unwrap();
                let last = program.passes.last().unwrap();
                match transform {
                    TransformKind::RealToComplex => {
                        assert_eq!(first.bindings.len(), 2);
                        assert_eq!(last.bindings.len(), 3);
                        assert!(last.bindings.iter().any(|binding| {
                            binding.binding == 2
                                && binding.resource == roots.id
                                && binding.role == BufferRole::LookupTable
                        }));
                    }
                    TransformKind::ComplexToReal => {
                        assert_eq!(first.bindings.len(), 3);
                        assert_eq!(last.bindings.len(), 2);
                        assert!(first.bindings.iter().any(|binding| {
                            binding.binding == 2
                                && binding.resource == roots.id
                                && binding.role == BufferRole::LookupTable
                        }));
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn double_double_real_program_separates_real_compact_and_full_dd_resources() {
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let external_scalar = if precision == Precision::DoubleDouble {
                ScalarType::DoubleDouble
            } else {
                ScalarType::F64
            };
            for (transform, kind, input_len, output_len) in [
                (
                    TransformKind::RealToComplex,
                    crate::RealFftKind::RealToComplex,
                    15usize,
                    8usize,
                ),
                (
                    TransformKind::ComplexToReal,
                    crate::RealFftKind::ComplexToReal,
                    8usize,
                    15usize,
                ),
            ] {
                let plan = FftPlan::build(
                    FftConfig::new(vec![15])
                        .with_batch_count(2)
                        .with_transform(transform)
                        .with_precision(precision)
                        .with_inverse_normalization(transform == TransformKind::ComplexToReal),
                )
                .unwrap();
                let ir = crate::DoubleDoubleRealFftIr::build(&plan).unwrap();
                assert_eq!(ir.kind, kind);
                let child = ProgramIr::double_double_one_dim(&ir.transform).unwrap();
                let program = ProgramIr::double_double_real(&ir).unwrap();
                let input = program.input_resource().unwrap();
                let output = program.output_resource().unwrap();
                assert_eq!(input.scalar, external_scalar);
                assert_eq!(output.scalar, external_scalar);
                assert_eq!(input.external_layout.unwrap().logical_len, input_len);
                assert_eq!(output.external_layout.unwrap().logical_len, output_len);
                assert_eq!(program.resources[2].scalar, ScalarType::DoubleDouble);
                assert_eq!(program.resources[2].elements, 30);
                assert_eq!(program.resources[3].scalar, ScalarType::DoubleDouble);
                assert_eq!(program.resources[3].elements, 30);
                assert_eq!(program.passes.len(), child.passes.len() + 2);
                assert_eq!(
                    program.passes.first().unwrap().bindings[0].resource,
                    ProgramResourceId(0)
                );
                assert_eq!(
                    program.passes.first().unwrap().bindings[1].resource,
                    ProgramResourceId(2)
                );
                assert_eq!(
                    program.passes.last().unwrap().bindings[0].resource,
                    ProgramResourceId(3)
                );
                assert_eq!(
                    program.passes.last().unwrap().bindings[1].resource,
                    ProgramResourceId(1)
                );
                for child_resource in child.resources.iter().filter(|resource| {
                    !matches!(
                        resource.kind,
                        ProgramResourceKind::Input | ProgramResourceKind::Output
                    )
                }) {
                    let name = format!("double_double_real_{}", child_resource.name);
                    let mapped = program
                        .resources
                        .iter()
                        .find(|resource| resource.name == name)
                        .expect("DD real child resource must be cloned into parent");
                    assert_eq!(mapped.scalar, ScalarType::DoubleDouble);
                    assert_eq!(mapped.initialization, child_resource.initialization);
                    assert!(mapped.external_layout.is_none());
                }
            }
        }
    }

    #[test]
    fn double_double_real_padding_stays_fused_in_true_real_boundaries() {
        for length in [15usize, 16usize] {
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                let external_scalar = if precision == Precision::DoubleDouble {
                    ScalarType::DoubleDouble
                } else {
                    ScalarType::F64
                };
                for transform in [TransformKind::RealToComplex, TransformKind::ComplexToReal] {
                    let config = FftConfig::new(vec![length])
                        .with_batch_count(7)
                        .with_transform(transform)
                        .with_precision(precision)
                        .with_inverse_normalization(transform == TransformKind::ComplexToReal)
                        .with_grouped_batch(0, 3)
                        .unwrap()
                        .with_zero_padding(0, 3, 7)
                        .unwrap();
                    let ir = crate::DoubleDoubleRealFftIr::build(&FftPlan::build(config).unwrap())
                        .unwrap();
                    assert!(ir.has_spatial_zero_padding());
                    assert_eq!(ir.grouped_batch, 3);
                    assert_eq!(ir.transform.grouped_batch(), 3);
                    assert_eq!(ir.batch_group_count(), 3);
                    let child = ProgramIr::double_double_one_dim(&ir.transform).unwrap();
                    let program = ProgramIr::double_double_real(&ir).unwrap();
                    assert!(program.passes.iter().all(|pass| pass.dispatch.x == 3));
                    assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
                    assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
                    assert_eq!(program.passes.len(), child.passes.len() + 2);
                    assert!(
                        !program
                            .passes
                            .iter()
                            .any(|pass| pass.name.contains("zero_pad"))
                    );
                    assert_eq!(
                        program.passes.first().unwrap().bindings[0].resource,
                        ProgramResourceId(0)
                    );
                    assert_eq!(
                        program.passes.last().unwrap().bindings[1].resource,
                        ProgramResourceId(1)
                    );
                    assert!(
                        program
                            .resources
                            .iter()
                            .skip(2)
                            .all(|resource| { resource.scalar == ScalarType::DoubleDouble })
                    );
                    assert_eq!(ir.even_half_size, length.is_multiple_of(2));
                    if ir.even_half_size {
                        assert_eq!(ir.transform.sequence_len(), length / 2);
                    } else {
                        assert_eq!(ir.transform.sequence_len(), length);
                    }
                }
            }
        }
    }

    #[test]
    fn double_double_nd_real_program_keeps_scalar_full_and_complex_compact_storage() {
        for last_len in [7usize, 8usize] {
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                let external_scalar = if precision == Precision::DoubleDouble {
                    ScalarType::DoubleDouble
                } else {
                    ScalarType::F64
                };
                for transform in [TransformKind::RealToComplex, TransformKind::ComplexToReal] {
                    let plan = FftPlan::build(
                        FftConfig::new(vec![3, last_len])
                            .with_batch_count(7)
                            .with_precision(precision)
                            .with_transform(transform)
                            .with_inverse_normalization(transform == TransformKind::ComplexToReal)
                            .with_grouped_batch(0, 3)
                            .unwrap()
                            .with_grouped_batch(1, 3)
                            .unwrap(),
                    )
                    .unwrap();
                    let ir = crate::DoubleDoubleNdRealFftIr::build(&plan).unwrap();
                    let program = ProgramIr::double_double_nd_real(&ir).unwrap();
                    let compact_last = last_len / 2 + 1;
                    assert_eq!(ir.real_grouped_batch, 3);
                    assert_eq!(ir.real_axis.grouped_batch, 3);
                    assert_eq!(ir.batch_group_count(), 3);
                    assert_eq!(ir.complex_axes.len(), 1);
                    assert_eq!(ir.complex_axes[0].grouped_batch, 3);
                    assert_eq!(ir.complex_axes[0].transform.grouped_batch(), 3);
                    assert!(program.passes.iter().any(|pass| pass.dispatch.x == 3));
                    assert!(program.passes.iter().any(|pass| pass.dispatch.x == 7));
                    let outer_groups = (7 * compact_last).div_ceil(3);
                    assert!(
                        program
                            .passes
                            .iter()
                            .any(|pass| pass.dispatch.x == outer_groups as u32)
                    );
                    let input = program.input_resource().unwrap();
                    let output = program.output_resource().unwrap();
                    assert_eq!(input.scalar, external_scalar);
                    assert_eq!(output.scalar, external_scalar);
                    assert_eq!(
                        input.element_shape(),
                        if transform == TransformKind::RealToComplex {
                            ProgramElementShape::Scalar
                        } else {
                            ProgramElementShape::Complex
                        }
                    );
                    assert_eq!(
                        output.element_shape(),
                        if transform == TransformKind::RealToComplex {
                            ProgramElementShape::Complex
                        } else {
                            ProgramElementShape::Scalar
                        }
                    );
                    assert_eq!(program.resources[2].scalar, ScalarType::DoubleDouble);
                    assert_eq!(
                        program.resources[2].element_shape(),
                        ProgramElementShape::Scalar
                    );
                    assert_eq!(program.resources[2].element_bytes(), 16);
                    for resource in program.resources.iter().skip(3) {
                        assert_eq!(resource.scalar, ScalarType::DoubleDouble);
                    }
                    assert!(
                        program
                            .passes
                            .first()
                            .unwrap()
                            .name
                            .contains("external_input_promote")
                    );
                    assert!(
                        program
                            .passes
                            .last()
                            .unwrap()
                            .name
                            .contains("external_output_finalize")
                    );
                    assert!(
                        program
                            .passes
                            .iter()
                            .any(|pass| pass.name.contains("last_axis_vkfft_dd_real_"))
                    );
                    assert!(
                        program
                            .passes
                            .iter()
                            .any(|pass| pass.name.contains("axis_0_pack"))
                    );
                    let memory = program.memory_plan().unwrap();
                    for allocation in &memory.allocations {
                        let Some(first) = allocation.resources.first().copied() else {
                            continue;
                        };
                        let shape = program.resources[first.0].element_shape();
                        assert!(
                            allocation
                                .resources
                                .iter()
                                .all(|resource| program.resources[resource.0].element_shape()
                                    == shape)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn double_double_nd_real_padding_stays_fused_in_full_real_boundary() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            for transform in [TransformKind::RealToComplex, TransformKind::ComplexToReal] {
                let config = FftConfig::new(vec![3, 8])
                    .with_batch_count(7)
                    .with_precision(precision)
                    .with_transform(transform)
                    .with_inverse_normalization(transform == TransformKind::ComplexToReal)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap()
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_zero_padding(1, 2, 4)
                    .unwrap();
                let ir = crate::DoubleDoubleNdRealFftIr::build(&FftPlan::build(config).unwrap())
                    .unwrap();
                assert!(ir.has_spatial_zero_padding());
                let program = ProgramIr::double_double_nd_real(&ir).unwrap();
                assert_eq!(ir.real_grouped_batch, 3);
                assert_eq!(ir.real_axis.grouped_batch, 3);
                assert_eq!(ir.complex_axes[0].transform.grouped_batch(), 3);
                assert_eq!(ir.batch_group_count(), 3);
                assert!(program.passes.iter().any(|pass| pass.dispatch.x == 3));
                assert!(program.passes.iter().any(|pass| pass.dispatch.x == 7));
                assert!(program.passes.iter().any(|pass| pass.dispatch.x == 12));
                assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
                assert!(
                    program
                        .passes
                        .first()
                        .unwrap()
                        .name
                        .contains("external_input_promote")
                );
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("external_output_finalize")
                );
                assert!(
                    !program
                        .passes
                        .iter()
                        .any(|pass| pass.name.contains("zero_pad"))
                );
                assert_eq!(
                    program.resources[2].element_shape(),
                    ProgramElementShape::Scalar
                );
                assert_eq!(program.resources[2].element_bytes(), 16);
                assert!(program.resources.iter().skip(2).all(|resource| {
                    resource.scalar == ScalarType::DoubleDouble
                        && resource.kind != ProgramResourceKind::Input
                        && resource.kind != ProgramResourceKind::Output
                }));
            }
        }
    }

    #[test]
    fn double_double_nd_padding_stays_fused_in_external_pack_scatter() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let config = FftConfig::new(vec![3, 4])
                    .with_batch_count(2)
                    .with_precision(precision)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_zero_padding(1, 1, 3)
                    .unwrap();
                let ir =
                    crate::DoubleDoubleNdFftIr::build(&FftPlan::build(config).unwrap(), direction)
                        .unwrap();
                assert!(ir.has_spatial_zero_padding());
                let program = ProgramIr::double_double_nd(&ir).unwrap();
                assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
                assert!(
                    !program
                        .passes
                        .iter()
                        .any(|pass| pass.name.contains("zero_pad"))
                );
                assert_eq!(
                    program.passes.first().unwrap().bindings[0].resource,
                    ProgramResourceId(0)
                );
                assert_eq!(
                    program.passes.last().unwrap().bindings[1].resource,
                    ProgramResourceId(1)
                );
                assert!(
                    program
                        .resources
                        .iter()
                        .skip(2)
                        .all(|resource| { resource.scalar == ScalarType::DoubleDouble })
                );
            }
        }
    }

    #[test]
    fn double_double_nd_program_maps_fft_rader_child_resources_and_boundaries() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan =
                FftPlan::build(FftConfig::new(vec![3, 17]).with_precision(precision)).unwrap();
            let ir = crate::DoubleDoubleNdFftIr::build(&plan, Direction::Forward).unwrap();
            let crate::DoubleDoubleOneDimIr::FftRader(rader) = &ir.axes[0].transform else {
                panic!("DD ND [3,17] fastest axis must use FFT Rader");
            };
            assert_eq!(rader.prime, 17);
            let child = ProgramIr::double_double_one_dim(&ir.axes[0].transform).unwrap();
            let program = ProgramIr::double_double_nd(&ir).unwrap();
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            assert!(
                program
                    .resources
                    .iter()
                    .skip(2)
                    .all(|resource| { resource.scalar == ScalarType::DoubleDouble })
            );
            for child_resource in child.resources.iter().filter(|resource| {
                !matches!(
                    resource.kind,
                    ProgramResourceKind::Input | ProgramResourceKind::Output
                )
            }) {
                let name = format!("double_double_nd_axis_1_{}", child_resource.name);
                let mapped = program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .expect("FFT-Rader child resource must be cloned into ND parent");
                assert_eq!(mapped.scalar, ScalarType::DoubleDouble);
                assert_eq!(mapped.elements, child_resource.elements);
                assert_eq!(mapped.initialization, child_resource.initialization);
                assert!(mapped.external_layout.is_none());
            }
            let gather = program
                .passes
                .iter()
                .find(|pass| pass.name.contains("axis_1_") && pass.name.contains("_gather"))
                .expect("ND parent must include FFT-Rader gather pass");
            assert_eq!(gather.bindings[0].resource, ProgramResourceId(4));
            let rader_scatter = program
                .passes
                .iter()
                .find(|pass| pass.name.contains("axis_1_") && pass.name.contains("_scatter"))
                .expect("ND parent must include FFT-Rader scatter pass");
            assert_eq!(rader_scatter.bindings[0].resource, ProgramResourceId(4));
            assert_eq!(rader_scatter.bindings[1].resource, ProgramResourceId(5));
            assert!(rader_scatter.bindings.iter().any(|binding| {
                binding.role == BufferRole::Auxiliary
                    && binding.resource != ProgramResourceId(4)
                    && binding.resource != ProgramResourceId(5)
            }));
            assert_eq!(
                program.passes.first().unwrap().bindings[0].resource,
                ProgramResourceId(0)
            );
            assert_eq!(
                program.passes.last().unwrap().bindings[1].resource,
                ProgramResourceId(1)
            );
        }
    }

    #[test]
    fn double_double_nd_program_keeps_only_tensor_boundaries_external() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![3, 4])
                    .with_batch_count(2)
                    .with_precision(precision),
            )
            .unwrap();
            let ir = crate::DoubleDoubleNdFftIr::build(&plan, Direction::Forward).unwrap();
            let program = ProgramIr::double_double_nd(&ir).unwrap();
            assert_eq!(program.scalar, ScalarType::DoubleDouble);
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            let child_programs = ir
                .axes
                .iter()
                .map(|axis| ProgramIr::double_double_one_dim(&axis.transform).unwrap())
                .collect::<Vec<_>>();
            let expected_internal_resources = child_programs
                .iter()
                .map(|child| {
                    child
                        .resources
                        .iter()
                        .filter(|resource| {
                            !matches!(
                                resource.kind,
                                ProgramResourceKind::Input | ProgramResourceKind::Output
                            )
                        })
                        .count()
                })
                .sum::<usize>();
            assert_eq!(program.resources.len(), 6 + expected_internal_resources);
            for resource in &program.resources[2..] {
                assert_eq!(resource.scalar, ScalarType::DoubleDouble);
            }
            let expected_passes = child_programs
                .iter()
                .map(|child| child.passes.len() + 2)
                .sum::<usize>();
            assert_eq!(program.passes.len(), expected_passes);
            assert_eq!(
                program.passes.first().unwrap().bindings[0].resource,
                ProgramResourceId(0)
            );
            assert_eq!(
                program.passes.last().unwrap().bindings[1].resource,
                ProgramResourceId(1)
            );
            for (axis, child) in ir.axes.iter().zip(&child_programs) {
                for child_resource in child.resources.iter().filter(|resource| {
                    !matches!(
                        resource.kind,
                        ProgramResourceKind::Input | ProgramResourceKind::Output
                    )
                }) {
                    let expected_name = format!(
                        "double_double_nd_axis_{}_{}",
                        axis.axis, child_resource.name
                    );
                    let cloned = program
                        .resources
                        .iter()
                        .find(|resource| resource.name == expected_name)
                        .expect("DD ND child internal resource must be cloned into parent");
                    assert_eq!(cloned.kind, child_resource.kind);
                    assert_eq!(cloned.scalar, child_resource.scalar);
                    assert_eq!(cloned.elements, child_resource.elements);
                    assert_eq!(cloned.initialization, child_resource.initialization);
                    assert!(cloned.external_layout.is_none());
                }
            }
        }
    }

    #[test]
    fn double_double_stockham_large_program_uses_dd_ping_pong_stages() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![1024])
                    .with_batch_count(2)
                    .with_precision(precision),
            )
            .unwrap();
            let ir = crate::DoubleDoubleStockhamIr::build(&plan, Direction::Forward).unwrap();
            assert!(ir.stages.len() >= 2);
            let program = ProgramIr::double_double_stockham(&ir).unwrap();
            assert_eq!(program.resources.len(), 5);
            assert_eq!(program.passes.len(), ir.stages.len() + 2);
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.resources[2].scalar, ScalarType::DoubleDouble);
            assert_eq!(program.resources[3].scalar, ScalarType::DoubleDouble);
            assert_eq!(program.resources[4].scalar, ScalarType::DoubleDouble);
            assert_eq!(program.passes.first().unwrap().bindings.len(), 2);
            assert_eq!(program.passes.last().unwrap().bindings.len(), 2);
            for pass in &program.passes[1..program.passes.len() - 1] {
                assert_eq!(pass.bindings.len(), 3);
                assert_eq!(pass.bindings[2].role, BufferRole::TwiddleLookupTable);
            }
            let ProgramResourceInitialization::ComplexDoubleDouble(values) =
                &program.resources[4].initialization
            else {
                panic!("large DD Stockham twiddle resource must preserve DD words");
            };
            assert_eq!(values, &ir.twiddles.packed_values());
        }
    }

    #[test]
    fn double_double_stockham_program_separates_compute_and_external_storage() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![15])
                    .with_batch_count(2)
                    .with_precision(precision),
            )
            .unwrap();
            let ir = crate::DoubleDoubleStockhamIr::build(&plan, Direction::Forward).unwrap();
            let program = ProgramIr::double_double_stockham(&ir).unwrap();
            assert_eq!(program.scalar, ScalarType::DoubleDouble);
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.passes.len(), 1);
            assert_eq!(program.passes[0].bindings.len(), 3);
            assert_eq!(
                program.passes[0].bindings[2].role,
                BufferRole::TwiddleLookupTable
            );

            let lut = program
                .resources
                .iter()
                .find(|resource| resource.kind == ProgramResourceKind::LookupTable)
                .unwrap();
            assert_eq!(lut.scalar, ScalarType::DoubleDouble);
            let ProgramResourceInitialization::ComplexDoubleDouble(values) = &lut.initialization
            else {
                panic!("double-double twiddle resource must retain DD words");
            };
            assert_eq!(values.len(), lut.elements);
            assert_eq!(values, &ir.twiddles.packed_values());

            let memory = program.memory_plan().unwrap();
            let input_allocation = memory.allocation_for(ProgramResourceId(0)).unwrap();
            let output_allocation = memory.allocation_for(ProgramResourceId(1)).unwrap();
            let lut_allocation = memory.allocation_for(lut.id).unwrap();
            assert_eq!(
                memory.allocations[input_allocation.0].scalar,
                external_scalar
            );
            assert_eq!(
                memory.allocations[output_allocation.0].scalar,
                external_scalar
            );
            assert_eq!(
                memory.allocations[lut_allocation.0].scalar,
                ScalarType::DoubleDouble
            );
            assert_eq!(ScalarType::DoubleDouble.complex_bytes(), 32);
            assert_eq!(
                memory.allocations[lut_allocation.0].elements
                    * ScalarType::DoubleDouble.complex_bytes(),
                lut.elements * 32
            );
            let expected_external_bytes = if external_scalar == ScalarType::DoubleDouble {
                15 * 2 * 32
            } else {
                15 * 2 * 16
            };
            assert_eq!(
                memory.allocations[input_allocation.0].elements * external_scalar.complex_bytes(),
                expected_external_bytes
            );
        }
    }

    #[test]
    fn double_double_direct_rader_program_keeps_lut_at_compute_precision() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![47])
                    .with_batch_count(2)
                    .with_precision(precision),
            )
            .unwrap();
            let ir = crate::DoubleDoubleDirectRaderIr::build(&plan, Direction::Forward).unwrap();
            let program = ProgramIr::double_double_direct_rader(&ir).unwrap();
            assert_eq!(program.scalar, ScalarType::DoubleDouble);
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.resources.len(), 3);
            assert_eq!(program.passes.len(), 1);
            assert_eq!(program.passes[0].bindings.len(), 3);
            assert_eq!(program.passes[0].bindings[2].role, BufferRole::LookupTable);
            let lut = &program.resources[2];
            assert_eq!(lut.kind, ProgramResourceKind::LookupTable);
            assert_eq!(lut.scalar, ScalarType::DoubleDouble);
            assert_eq!(lut.elements, 46);
            let ProgramResourceInitialization::ComplexDoubleDouble(values) = &lut.initialization
            else {
                panic!("DD direct-Rader roots must keep four-word initialization");
            };
            assert_eq!(values, &ir.table.twiddles_by_generator_power);
            let memory = program.memory_plan().unwrap();
            assert_eq!(
                memory.allocations[memory.allocation_for(lut.id).unwrap().0].scalar,
                ScalarType::DoubleDouble
            );
        }
    }

    #[test]
    fn double_double_bluestein_program_keeps_chirp_and_convolution_at_dd_precision() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.max_rader_fft_prime = 100;
            let plan = FftPlan::build(
                FftConfig::new(vec![103])
                    .with_batch_count(7)
                    .with_precision(precision)
                    .with_tuning(tuning)
                    .with_grouped_batch(0, 3)
                    .unwrap(),
            )
            .unwrap();
            let ir = crate::DoubleDoubleBluesteinIr::build(&plan, Direction::Forward).unwrap();
            assert_eq!(ir.convolution_len, 210);
            assert_eq!(ir.batch_count, 7);
            assert_eq!(ir.grouped_batch, 3);
            assert_eq!(ir.batch_group_count(), 3);
            assert_eq!(ir.forward_fft.grouped_batch(), 3);
            assert_eq!(ir.inverse_fft.grouped_batch(), 3);
            let program = ProgramIr::double_double_bluestein(&ir).unwrap();
            assert_eq!(program.scalar, ScalarType::DoubleDouble);
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            let parent_scratch = [
                "double_double_bluestein_forward_input",
                "double_double_bluestein_forward_output",
                "double_double_bluestein_inverse_output",
            ]
            .map(|name| {
                program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap_or_else(|| panic!("missing fused DD Bluestein scratch {name}"))
            });
            for resource in parent_scratch {
                assert_eq!(resource.kind, ProgramResourceKind::Scratch);
                assert_eq!(resource.scalar, ScalarType::DoubleDouble);
                assert_eq!(resource.elements, 210 * 7);
            }
            assert!(
                !program
                    .resources
                    .iter()
                    .any(|resource| { resource.name == "double_double_bluestein_inverse_input" })
            );
            let forward_input = program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_bluestein_forward_input")
                .unwrap();
            let forward_output = program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_bluestein_forward_output")
                .unwrap();
            let inverse_output = program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_bluestein_inverse_output")
                .unwrap();
            let chirp = program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_bluestein_chirp")
                .unwrap();
            let kernel = program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_bluestein_kernel_spectrum")
                .unwrap();
            assert!(program.resources.len() > 8);
            assert!(program.passes.len() > 3);
            assert!(program.passes.iter().all(|pass| pass.dispatch.x == 3));
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| !pass.name.ends_with("_multiply"))
            );
            let inverse_stage0 = program
                .passes
                .iter()
                .find(|pass| {
                    pass.name.contains("vkfft_dd_bluestein_inverse")
                        && pass.name.contains("stage_0")
                })
                .expect("DD Bluestein inverse Stockham stage0");
            assert!(inverse_stage0.name.ends_with("_input_mul_lut"));
            assert!(inverse_stage0.bindings.iter().any(|binding| {
                binding.binding == 3
                    && binding.role == BufferRole::LookupTable
                    && binding.access == BufferAccess::ReadOnly
            }));
            let preprocess = &program.passes[0];
            assert_eq!(preprocess.bindings.len(), 3);
            assert_eq!(preprocess.bindings[0].resource, ProgramResourceId(0));
            assert_eq!(preprocess.bindings[1].resource, forward_input.id);
            assert_eq!(preprocess.bindings[2].resource, chirp.id);
            assert_eq!(preprocess.bindings[2].role, BufferRole::LookupTable);
            let postprocess = program.passes.last().unwrap();
            assert_eq!(postprocess.bindings[0].role, BufferRole::Input);
            assert_eq!(postprocess.bindings[0].resource, inverse_output.id);
            assert_eq!(postprocess.bindings[1].resource, ProgramResourceId(1));
            assert_eq!(postprocess.bindings[2].resource, chirp.id);
            for resource in program
                .resources
                .iter()
                .filter(|resource| resource.kind == ProgramResourceKind::LookupTable)
            {
                assert_eq!(resource.scalar, ScalarType::DoubleDouble);
                let ProgramResourceInitialization::ComplexDoubleDouble(values) =
                    &resource.initialization
                else {
                    panic!("DD Bluestein immutable tables must retain all four words");
                };
                assert_eq!(values.len(), resource.elements);
                assert!(
                    values
                        .iter()
                        .any(|value| value.re.lo != 0.0 || value.im.lo != 0.0)
                );
            }
            assert_eq!(chirp.elements, 103);
            assert_eq!(kernel.elements, 210);
            let memory = program.memory_plan().unwrap();
            let parent_scratch_allocations =
                [forward_input.id, forward_output.id, inverse_output.id]
                    .map(|resource| memory.allocation_for(resource).unwrap());
            assert!(
                parent_scratch_allocations
                    .iter()
                    .all(|allocation| *allocation == parent_scratch_allocations[0])
            );
            let scratch_allocations = memory
                .allocations
                .iter()
                .filter(|allocation| allocation.kind == ProgramAllocationKind::Scratch)
                .collect::<Vec<_>>();
            assert_eq!(scratch_allocations.len(), 4);
            assert!(scratch_allocations.iter().all(|allocation| {
                allocation.scalar == ScalarType::DoubleDouble && allocation.elements == 210 * 7
            }));
            for resource in &program.resources[2..] {
                assert_eq!(
                    memory.allocations[memory.allocation_for(resource.id).unwrap().0].scalar,
                    ScalarType::DoubleDouble
                );
            }
        }
    }

    #[test]
    fn double_double_fft_rader_program_keeps_all_convolution_tables_at_dd_precision() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![257])
                    .with_batch_count(2)
                    .with_precision(precision),
            )
            .unwrap();
            let ir = crate::DoubleDoubleFftRaderIr::build(&plan, Direction::Forward).unwrap();
            let program = ProgramIr::double_double_fft_rader(&ir).unwrap();
            assert_eq!(program.scalar, ScalarType::DoubleDouble);
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            let child_resources = |child: &crate::DoubleDoubleBluesteinConvolutionIr| match child {
                crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child) => {
                    ProgramIr::double_double_stockham(child)
                        .unwrap()
                        .resources
                        .len()
                        - 2
                }
                crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) => {
                    ProgramIr::double_double_recursive(child)
                        .unwrap()
                        .resources
                        .len()
                        - 2
                }
                crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) => {
                    ProgramIr::double_double_bluestein(child)
                        .unwrap()
                        .resources
                        .len()
                        - 2
                }
            };
            assert_eq!(
                program.resources.len(),
                5 + child_resources(&ir.forward_fft) + child_resources(&ir.inverse_fft)
            );
            assert_eq!(program.resources[2].kind, ProgramResourceKind::Scratch);
            assert_eq!(program.resources[3].kind, ProgramResourceKind::Scratch);
            assert_eq!(program.resources[2].elements, 256 * 2);
            assert_eq!(program.resources[3].elements, 256 * 2);
            let child_passes = |child: &crate::DoubleDoubleBluesteinConvolutionIr| match child {
                crate::DoubleDoubleBluesteinConvolutionIr::Stockham(child) => {
                    ProgramIr::double_double_stockham_internal(child)
                        .unwrap()
                        .passes
                        .len()
                }
                crate::DoubleDoubleBluesteinConvolutionIr::Recursive(child) => {
                    ProgramIr::double_double_recursive(child)
                        .unwrap()
                        .passes
                        .len()
                }
                crate::DoubleDoubleBluesteinConvolutionIr::Bluestein(child) => {
                    ProgramIr::double_double_bluestein(child)
                        .unwrap()
                        .passes
                        .len()
                }
            };
            let expected_passes = 2 + child_passes(&ir.forward_fft) + child_passes(&ir.inverse_fft);
            assert_eq!(program.passes.len(), expected_passes);
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| !pass.name.ends_with("_multiply"))
            );
            assert_eq!(program.passes[0].bindings.len(), 2);
            assert_eq!(program.passes[0].bindings[0].resource, ProgramResourceId(0));
            assert_eq!(program.passes[0].bindings[1].resource, ProgramResourceId(2));
            let kernel_resource = program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_rader_kernel_spectrum")
                .expect("DD FFT-Rader kernel spectrum resource");
            assert_eq!(kernel_resource.elements, 256);
            let inverse_stage0 = program
                .passes
                .iter()
                .find(|pass| {
                    pass.name.contains("vkfft_dd_fft_rader_inverse")
                        && pass.name.contains("stage_0")
                })
                .expect("DD FFT-Rader inverse Stockham stage0");
            assert!(inverse_stage0.name.ends_with("_input_mul_lut"));
            assert!(inverse_stage0.bindings.iter().any(|binding| {
                binding.binding == 3
                    && binding.resource == kernel_resource.id
                    && binding.role == BufferRole::LookupTable
                    && binding.access == BufferAccess::ReadOnly
            }));
            assert!(program.passes.iter().any(|pass| {
                pass.bindings.iter().any(|binding| {
                    binding.resource == kernel_resource.id
                        && binding.role == BufferRole::LookupTable
                })
            }));
            assert!(
                program
                    .passes
                    .iter()
                    .filter(|pass| {
                        pass.bindings
                            .iter()
                            .any(|binding| binding.role == BufferRole::TwiddleLookupTable)
                    })
                    .count()
                    >= 2
            );
            let scatter = program.passes.last().unwrap();
            assert_eq!(scatter.bindings.len(), 3);
            assert_eq!(scatter.bindings[0].resource, ProgramResourceId(0));
            assert_eq!(scatter.bindings[1].resource, ProgramResourceId(1));
            assert_eq!(scatter.bindings[2].role, BufferRole::Auxiliary);
            assert!(matches!(scatter.bindings[2].resource.0, 2 | 3));
            for resource in program
                .resources
                .iter()
                .filter(|resource| resource.kind == ProgramResourceKind::LookupTable)
            {
                assert_eq!(resource.scalar, ScalarType::DoubleDouble);
                let ProgramResourceInitialization::ComplexDoubleDouble(values) =
                    &resource.initialization
                else {
                    panic!("DD FFT-Rader immutable tables must keep four-word initialization");
                };
                assert_eq!(values.len(), resource.elements);
                assert!(
                    values
                        .iter()
                        .any(|value| value.re.lo != 0.0 || value.im.lo != 0.0)
                );
            }
            let memory = program.memory_plan().unwrap();
            for resource in &program.resources[2..] {
                assert_eq!(
                    memory.allocations[memory.allocation_for(resource.id).unwrap().0].scalar,
                    ScalarType::DoubleDouble
                );
            }
        }
    }

    #[test]
    fn bluestein_program_describes_ping_pong_and_lut_resources() {
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        let plan = FftPlan::build(
            FftConfig::new(vec![103])
                .with_batch_count(2)
                .with_tuning(bluestein_tuning),
        )
        .unwrap();
        let pipeline = BluesteinPipelineIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::bluestein(&pipeline).unwrap();
        assert_eq!(program.resources.len(), 5);
        assert_eq!(program.passes.len(), 4);
        assert_eq!(
            program.input_resource().unwrap().elements,
            pipeline.logical_len * 2
        );
        assert_eq!(
            program.output_resource().unwrap().elements,
            pipeline.logical_len * 2
        );
        assert_eq!(program.resources[2].kind, ProgramResourceKind::LookupTable);
        assert_eq!(program.resources[3].elements, pipeline.convolution_len * 2);
        assert_eq!(program.resources[4].elements, pipeline.convolution_len * 2);
        assert_eq!(program.passes[2].bindings[2].role, BufferRole::LookupTable);
        assert_eq!(program.passes[2].bindings[2].resource, ProgramResourceId(2));
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == pipeline.multiply.name)
        );
        assert_eq!(program.passes[0].bindings[0].resource, ProgramResourceId(0));
        assert_eq!(program.passes[0].bindings[1].resource, ProgramResourceId(3));
        assert_eq!(program.passes[3].bindings[0].resource, ProgramResourceId(3));
        assert_eq!(program.passes[3].bindings[1].resource, ProgramResourceId(1));
    }

    #[test]
    fn grouped_bluestein_program_keeps_parent_tail_and_single_kernel_lut() {
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let config = FftConfig::new(vec![103])
            .with_batch_count(7)
            .with_grouped_batch(0, 3)
            .unwrap()
            .with_precision(Precision::F16StorageF32Compute)
            .with_tuning(tuning)
            .with_zero_padding(0, 19, 31)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = crate::OneDimFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let crate::OneDimFftIr::Bluestein(pipeline) = &ir else {
            panic!("forced p103 grouped plan should use Bluestein");
        };
        assert_eq!(pipeline.grouped_batch, 3);
        let program = ProgramIr::one_dim_fft(&ir).unwrap();
        assert!(
            program.passes.iter().all(|pass| pass.dispatch.x == 3),
            "grouped Bluestein dispatches: {:?}",
            program
                .passes
                .iter()
                .map(|pass| (pass.name.as_str(), pass.dispatch.x))
                .collect::<Vec<_>>()
        );
        let kernel_luts = program
            .resources
            .iter()
            .filter(|resource| {
                resource.kind == ProgramResourceKind::LookupTable
                    && resource.name.contains("bluestein_kernel_spectrum")
            })
            .collect::<Vec<_>>();
        assert_eq!(kernel_luts.len(), 1);
        assert_eq!(kernel_luts[0].elements, pipeline.convolution_len);
        assert_eq!(program.input_resource().unwrap().elements, 103 * 7);
        assert_eq!(program.output_resource().unwrap().elements, 103 * 7);
        assert!(
            program
                .passes
                .first()
                .unwrap()
                .name
                .contains("zero_pad_forward_input")
        );
        program.validate().unwrap();
    }

    #[test]
    fn multidimensional_real_and_r2r_programs_use_axis_composition() {
        let r2c_plan = FftPlan::build(
            FftConfig::new(vec![3, 8]).with_transform(crate::TransformKind::RealToComplex),
        )
        .unwrap();
        let r2c = crate::NdRealFftIr::build(&r2c_plan, device()).unwrap();
        let r2c_program = ProgramIr::nd_real_fft(&r2c).unwrap();
        assert_eq!(r2c_program.input_resource().unwrap().elements, 24);
        assert_eq!(r2c_program.output_resource().unwrap().elements, 15);
        assert_eq!(r2c_program.passes.len(), 4);

        let r2r_plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_transform(crate::TransformKind::Dct(crate::DctType::II)),
        )
        .unwrap();
        let r2r = crate::NdR2rIr::build(&r2r_plan, Direction::Forward, device()).unwrap();
        let r2r_program = ProgramIr::nd_r2r(&r2r).unwrap();
        assert_eq!(r2r_program.input_resource().unwrap().elements, 12);
        assert_eq!(r2r_program.output_resource().unwrap().elements, 12);
        assert_eq!(r2r_program.passes.len(), 10);
        assert!(r2r_program.passes[0].name.contains("nd_r2r_pack_axis_1"));
    }

    #[test]
    fn r2r_fft_reduction_programs_cover_all_i_through_iv_shapes() {
        for (transform, label, length, fft_elements) in [
            (
                crate::TransformKind::Dct(crate::DctType::I),
                "dct1",
                9usize,
                32usize,
            ),
            (
                crate::TransformKind::Dst(crate::DstType::I),
                "dst1",
                9usize,
                40usize,
            ),
            (
                crate::TransformKind::Dct(crate::DctType::II),
                "dct2",
                9usize,
                18usize,
            ),
            (
                crate::TransformKind::Dct(crate::DctType::III),
                "dct3",
                9usize,
                18usize,
            ),
            (
                crate::TransformKind::Dst(crate::DstType::II),
                "dst2",
                9usize,
                18usize,
            ),
            (
                crate::TransformKind::Dst(crate::DstType::III),
                "dst3",
                9usize,
                18usize,
            ),
            (
                crate::TransformKind::Dct(crate::DctType::IV),
                "dct4_even",
                8usize,
                8usize,
            ),
            (
                crate::TransformKind::Dst(crate::DstType::IV),
                "dst4_even",
                8usize,
                8usize,
            ),
            (
                crate::TransformKind::Dct(crate::DctType::IV),
                "dct4_odd",
                9usize,
                36usize,
            ),
            (
                crate::TransformKind::Dst(crate::DstType::IV),
                "dst4_odd",
                9usize,
                36usize,
            ),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(2)
                    .with_transform(transform),
            )
            .unwrap();
            let ir = crate::R2rIr::build(&plan, Direction::Forward, device()).unwrap();
            assert!(ir.fft_reduction.is_some());
            let program = ProgramIr::r2r(&ir).unwrap();
            assert_eq!(program.resources.len(), 4);
            assert_eq!(program.passes.len(), 3);
            assert_eq!(program.input_resource().unwrap().elements, 2 * length);
            assert_eq!(program.output_resource().unwrap().elements, 2 * length);
            assert_eq!(program.resources[2].elements, fft_elements);
            assert_eq!(program.resources[3].elements, fft_elements);
            assert!(
                program
                    .passes
                    .first()
                    .unwrap()
                    .name
                    .contains(&format!("r2r_fft_{label}_pre"))
            );
            assert!(
                program
                    .passes
                    .last()
                    .unwrap()
                    .name
                    .contains(&format!("r2r_fft_{label}_post"))
            );
        }
    }

    #[test]
    fn one_dim_zero_padding_wraps_forward_and_inverse_at_external_boundaries() {
        let config = FftConfig::new(vec![64])
            .with_zero_padding(0, 32, 64)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let forward = OneDimFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let forward_program = ProgramIr::one_dim_fft(&forward).unwrap();
        assert!(
            forward_program
                .passes
                .first()
                .unwrap()
                .name
                .contains("zero_pad_forward_input")
        );
        assert_eq!(
            forward_program.passes.first().unwrap().bindings[0].resource,
            ProgramResourceId(0)
        );
        assert_eq!(
            forward_program
                .resource(forward_program.passes.first().unwrap().bindings[1].resource)
                .unwrap()
                .kind,
            ProgramResourceKind::Scratch
        );

        let config = FftConfig::new(vec![64])
            .with_inverse_normalization(true)
            .with_zero_padding(0, 32, 64)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let inverse = OneDimFftIr::build(&plan, Direction::Inverse, device()).unwrap();
        let inverse_program = ProgramIr::one_dim_fft(&inverse).unwrap();
        assert!(
            inverse_program
                .passes
                .last()
                .unwrap()
                .name
                .contains("zero_pad_inverse_output")
        );
        assert_eq!(
            inverse_program.passes.last().unwrap().bindings[1].resource,
            ProgramResourceId(1)
        );
        assert_eq!(
            inverse_program
                .resource(inverse_program.passes.last().unwrap().bindings[0].resource)
                .unwrap()
                .kind,
            ProgramResourceKind::Scratch
        );
    }

    #[test]
    fn frequency_zero_padding_reverses_mixed_storage_program_boundary_order() {
        let build = |direction| {
            let config = FftConfig::new(vec![64])
                .with_batch_count(2)
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_zero_padding(0, 16, 32)
                .unwrap()
                .with_zero_padding_domain(ZeroPaddingDomain::Frequency);
            let plan = FftPlan::build(config).unwrap();
            let ir = OneDimFftIr::build(&plan, direction, device()).unwrap();
            ProgramIr::one_dim_fft(&ir).unwrap()
        };

        let forward = build(Direction::Forward);
        let forward_pad = forward.passes.last().unwrap();
        assert!(forward_pad.name.contains("zero_pad_forward_output"));
        assert_eq!(forward_pad.bindings[1].resource, ProgramResourceId(1));
        assert_eq!(
            forward
                .resource(forward_pad.bindings[0].resource)
                .unwrap()
                .kind,
            ProgramResourceKind::Scratch
        );
        assert_eq!(
            forward
                .resource(forward_pad.bindings[0].resource)
                .unwrap()
                .scalar,
            ScalarType::F32
        );
        assert_eq!(forward.output_resource().unwrap().scalar, ScalarType::F16);

        let inverse = build(Direction::Inverse);
        let inverse_pad = inverse.passes.first().unwrap();
        assert!(inverse_pad.name.contains("zero_pad_inverse_input"));
        assert_eq!(inverse_pad.bindings[0].resource, ProgramResourceId(0));
        assert_eq!(inverse.input_resource().unwrap().scalar, ScalarType::F16);
        assert_eq!(
            inverse
                .resource(inverse_pad.bindings[1].resource)
                .unwrap()
                .kind,
            ProgramResourceKind::Scratch
        );
        assert_eq!(
            inverse
                .resource(inverse_pad.bindings[1].resource)
                .unwrap()
                .scalar,
            ScalarType::F32
        );
    }

    #[test]
    fn generic_one_dim_program_covers_bluestein_with_compact_external_buffers() {
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        let plan = FftPlan::build(
            FftConfig::new(vec![103])
                .with_batch_count(2)
                .with_tuning(bluestein_tuning),
        )
        .unwrap();
        let ir = crate::OneDimFftIr::build(&plan, Direction::Forward, device()).unwrap();
        assert!(matches!(ir, crate::OneDimFftIr::Bluestein(_)));
        let program = ProgramIr::one_dim_fft(&ir).unwrap();
        assert_eq!(program.input_resource().unwrap().elements, 206);
        assert_eq!(program.output_resource().unwrap().elements, 206);
        assert_eq!(program.passes.len(), 4);
        assert_eq!(program.passes[2].bindings.len(), 3);
        assert_eq!(program.passes[2].bindings[2].role, BufferRole::LookupTable);
        assert!(
            program
                .resources
                .iter()
                .any(|resource| resource.kind == ProgramResourceKind::LookupTable)
        );
        let memory = program.memory_plan().unwrap();
        assert!(memory.allocations.len() <= program.resources.len());
    }

    #[test]
    fn mixed_program_flattens_outer_and_nested_fft_rader_passes() {
        for length in [34usize, 514] {
            let plan = FftPlan::build(FftConfig::new(vec![length]).with_batch_count(2)).unwrap();
            let ir =
                crate::MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
            let program = ProgramIr::mixed_rader_stockham(&ir).unwrap();
            assert_eq!(program.resources.len(), 9);
            assert_eq!(program.passes.len(), 6);
            assert_eq!(program.input_resource().unwrap().elements, 2 * length);
            assert_eq!(program.output_resource().unwrap().elements, 2 * length);
            assert_eq!(program.resources[5].kind, ProgramResourceKind::LookupTable);
            assert_eq!(program.passes[2].bindings[2].role, BufferRole::LookupTable);
            assert_eq!(program.passes[2].bindings[2].resource, ProgramResourceId(5));
            assert_eq!(program.passes[2].bindings[3].role, BufferRole::Auxiliary);
            assert_eq!(program.passes[2].bindings[3].resource, ProgramResourceId(1));
            if length == 514 {
                // The fused generator-order forward FFT sees four logical 256-point
                // containers; both internal FFTs physically launch two grouped workgroups.
                assert_eq!(program.passes[1].dispatch.x, 2);
                assert_eq!(program.passes[2].dispatch.x, 2);
            }
        }
    }

    #[test]
    fn p257_rader_transpose_fuses_generator_gather_into_forward_stockham() {
        let plan = FftPlan::build(FftConfig::new(vec![2056])).unwrap();
        let ir = crate::MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let MixedPrimeRaderIr::FftConvolution(rader) = &ir.prime_stage else {
            panic!("2056 should use FFT-convolution Rader");
        };
        assert_eq!(
            rader.input_strategy,
            RaderFftInputStrategy::GeneratorOrderStockham
        );
        let program = ProgramIr::mixed_rader_stockham(&ir).unwrap();
        assert_eq!(program.resources.len(), 9);
        assert_eq!(program.passes.len(), 6);
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == rader.gather.name)
        );
        assert_eq!(program.passes[1].bindings[0].resource, ProgramResourceId(1));
        assert_eq!(program.passes[1].bindings[0].role, BufferRole::Input);
        assert_eq!(program.passes[2].bindings[2].role, BufferRole::LookupTable);
        assert_eq!(program.passes[2].bindings[2].resource, ProgramResourceId(5));
        assert_eq!(program.passes[2].bindings[3].role, BufferRole::Auxiliary);
        assert_eq!(program.passes[2].bindings[3].resource, ProgramResourceId(1));
        assert_eq!(program.passes[2].bindings[1].resource, ProgramResourceId(4));
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == rader.scatter.name)
        );
    }

    #[test]
    fn mixed_program_flattens_direct_rader_stage() {
        let plan = FftPlan::build(FftConfig::new(vec![94]).with_batch_count(2)).unwrap();
        let ir = crate::MixedRaderStockhamIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::mixed_rader_stockham(&ir).unwrap();
        assert_eq!(program.resources.len(), 7);
        assert_eq!(program.passes.len(), 5);
        assert_eq!(program.resources[3].kind, ProgramResourceKind::LookupTable);
        assert_eq!(program.passes[1].bindings[2].resource, ProgramResourceId(3));
    }

    #[test]
    fn fft_rader_program_preserves_prime_input_for_scatter() {
        let plan = FftPlan::build(FftConfig::new(vec![17]).with_batch_count(2)).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::rader_fft(&pipeline).unwrap();
        assert_eq!(program.resources.len(), 5);
        assert_eq!(program.passes.len(), 2);
        assert_eq!(
            pipeline.input_strategy,
            RaderFftInputStrategy::GeneratorOrderStockham
        );
        let fused_inverse = &program.passes[1];
        assert_eq!(fused_inverse.bindings[2].role, BufferRole::LookupTable);
        assert_eq!(fused_inverse.bindings[2].resource, ProgramResourceId(4));
        assert_eq!(fused_inverse.bindings[3].role, BufferRole::Auxiliary);
        assert_eq!(fused_inverse.bindings[3].resource, ProgramResourceId(0));
        assert_eq!(fused_inverse.bindings[1].resource, ProgramResourceId(3));
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == pipeline.scatter.name)
        );
        assert_eq!(program.resources[4].kind, ProgramResourceKind::LookupTable);
    }

    #[test]
    fn recursive_program_flattens_repeated_and_multi_prime_rader_trees() {
        for (length, expected_passes, expected_luts) in
            [(289usize, 7usize, 1usize), (323, 7, 2), (578, 8, 1)]
        {
            let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
            let ir = crate::RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
            let program = ProgramIr::recursive_fft(&ir).unwrap();
            assert_eq!(program.passes.len(), expected_passes);
            assert_eq!(program.input_resource().unwrap().elements, length);
            assert_eq!(program.output_resource().unwrap().elements, length);
            assert_eq!(
                program
                    .resources
                    .iter()
                    .filter(|resource| resource.kind == ProgramResourceKind::LookupTable)
                    .count(),
                expected_luts
            );
            assert!(program.passes.iter().any(|pass| {
                pass.bindings
                    .iter()
                    .any(|binding| binding.role == BufferRole::Auxiliary)
            }));
            if length == 578 {
                assert!(
                    !program
                        .passes
                        .iter()
                        .any(|pass| pass.name == "vkfft_recursive_twiddle_34_forward")
                );
                assert!(
                    !program
                        .passes
                        .iter()
                        .any(|pass| pass.name.contains("recursive_scatter_34"))
                );
                assert!(
                    !program
                        .passes
                        .iter()
                        .any(|pass| pass.name.contains("recursive_pack_34"))
                );
            }
        }
    }

    #[test]
    fn real_fft_program_uses_compact_external_layouts() {
        let length = 17usize;
        let batch_count = 2usize;
        let r2c_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::RealToComplex),
        )
        .unwrap();
        let r2c = crate::RealFftIr::build(&r2c_plan, device()).unwrap();
        let r2c_program = ProgramIr::real_fft(&r2c).unwrap();
        assert_eq!(
            r2c_program.input_resource().unwrap().elements,
            length * batch_count
        );
        assert_eq!(
            r2c_program.output_resource().unwrap().elements,
            (length / 2 + 1) * batch_count
        );
        assert!(r2c_program.passes.last().unwrap().name.contains("real_r2c"));

        let c2r_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let c2r = crate::RealFftIr::build(&c2r_plan, device()).unwrap();
        let c2r_program = ProgramIr::real_fft(&c2r).unwrap();
        assert_eq!(
            c2r_program.input_resource().unwrap().elements,
            (length / 2 + 1) * batch_count
        );
        assert_eq!(
            c2r_program.output_resource().unwrap().elements,
            length * batch_count
        );
        assert!(
            c2r_program
                .passes
                .first()
                .unwrap()
                .name
                .contains("real_c2r")
        );
    }

    #[test]
    fn real_zero_padding_wraps_only_external_spatial_boundaries() {
        let r2c_plan = FftPlan::build(
            FftConfig::new(vec![34])
                .with_transform(crate::TransformKind::RealToComplex)
                .with_zero_padding(0, 17, 34)
                .unwrap(),
        )
        .unwrap();
        let r2c = crate::RealFftIr::build(&r2c_plan, device()).unwrap();
        let r2c_program = ProgramIr::real_fft(&r2c).unwrap();
        assert!(
            r2c_program
                .passes
                .first()
                .unwrap()
                .name
                .contains("zero_pad_forward_input")
        );
        assert_eq!(
            r2c_program.passes.first().unwrap().bindings[0].resource,
            ProgramResourceId(0)
        );
        assert_eq!(
            r2c_program
                .resource(r2c_program.passes.first().unwrap().bindings[1].resource)
                .unwrap()
                .kind,
            ProgramResourceKind::Scratch
        );

        let c2r_plan = FftPlan::build(
            FftConfig::new(vec![34])
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_zero_padding(0, 17, 34)
                .unwrap(),
        )
        .unwrap();
        let c2r = crate::RealFftIr::build(&c2r_plan, device()).unwrap();
        let c2r_program = ProgramIr::real_fft(&c2r).unwrap();
        assert!(
            c2r_program
                .passes
                .last()
                .unwrap()
                .name
                .contains("zero_pad_inverse_output")
        );
        assert_eq!(
            c2r_program.passes.last().unwrap().bindings[1].resource,
            ProgramResourceId(1)
        );

        let nd_plan = FftPlan::build(
            FftConfig::new(vec![3, 8])
                .with_transform(crate::TransformKind::RealToComplex)
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .with_zero_padding(1, 6, 8)
                .unwrap(),
        )
        .unwrap();
        let nd = crate::NdRealFftIr::build(&nd_plan, device()).unwrap();
        assert!(nd.real_axis.zero_pad_pass.is_none());
        let nd_program = ProgramIr::nd_real_fft(&nd).unwrap();
        assert!(
            nd_program
                .passes
                .first()
                .unwrap()
                .name
                .contains("nd_zero_pad_forward_input")
        );
        assert_eq!(
            nd_program.passes.first().unwrap().bindings[0].resource,
            ProgramResourceId(0)
        );
    }

    #[test]
    fn even_real_stockham_program_fuses_io_passes() {
        let length = 16usize;
        let batch_count = 2usize;
        let r2c_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::RealToComplex),
        )
        .unwrap();
        let r2c = crate::RealFftIr::build(&r2c_plan, device()).unwrap();
        let fused_r2c = r2c
            .fused_even_input_stockham_kernel()
            .unwrap()
            .expect("16-point R2C should have a single half-size Stockham root");
        assert!(matches!(
            fused_r2c.io_mapping,
            crate::StockhamIoMapping::RealEvenPack(_)
        ));
        assert!(matches!(
            fused_r2c.output_modifier,
            crate::StockhamOutputModifier::RealEvenPostprocess(_)
        ));
        let r2c_program = ProgramIr::real_fft(&r2c).unwrap();
        assert_eq!(r2c_program.passes.len(), 1);
        assert!(r2c_program.passes[0].name.contains("stockham"));
        assert_eq!(r2c_program.passes[0].bindings.len(), 3);
        assert_eq!(
            r2c_program.passes[0].bindings[2].role,
            BufferRole::Auxiliary
        );
        assert_eq!(
            r2c_program.passes[0].bindings[2].access,
            BufferAccess::ReadWrite
        );
        assert_eq!(
            r2c_program
                .resource(r2c_program.passes[0].bindings[2].resource)
                .unwrap()
                .kind,
            ProgramResourceKind::Scratch
        );

        let c2r_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(crate::TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let c2r = crate::RealFftIr::build(&c2r_plan, device()).unwrap();
        let fused_c2r = c2r
            .fused_even_input_stockham_kernel()
            .unwrap()
            .expect("16-point C2R should have a single half-size Stockham root");
        assert!(matches!(
            fused_c2r.io_mapping,
            crate::StockhamIoMapping::RealEvenInversePreprocess(_)
        ));
        assert!(matches!(
            fused_c2r.output_modifier,
            crate::StockhamOutputModifier::RealEvenUnpack(_)
        ));
        let c2r_program = ProgramIr::real_fft(&c2r).unwrap();
        assert_eq!(c2r_program.passes.len(), 1);
        assert_eq!(
            c2r_program.passes[0].bindings[0].resource,
            ProgramResourceId(0)
        );
        assert_eq!(
            c2r_program.passes[0].bindings[1].resource,
            ProgramResourceId(1)
        );
    }

    #[test]
    fn recursive_even_real_program_fuses_root_boundaries() {
        let length = 578usize;
        for transform in [
            crate::TransformKind::RealToComplex,
            crate::TransformKind::ComplexToReal,
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_transform(transform)
                    .with_inverse_normalization(transform == crate::TransformKind::ComplexToReal),
            )
            .unwrap();
            let ir = crate::RealFftIr::build(&plan, device()).unwrap();
            assert!(ir.fused_even_input_stockham_kernel().unwrap().is_none());
            assert!(ir.fused_even_recursive_ir().unwrap().is_some());
            let program = ProgramIr::real_fft(&ir).unwrap();
            assert!(
                program
                    .resources
                    .iter()
                    .all(|resource| !resource.name.contains("real_recursive_transform"))
            );
            assert!(program.passes.iter().all(|pass| {
                !pass.name.contains("pack_even_odd")
                    && !pass.name.contains("preprocess_even_half")
                    && !pass.name.contains("postprocess_even_half")
                    && !pass.name.contains("unpack_even_odd")
            }));
            match transform {
                crate::TransformKind::RealToComplex => {
                    assert!(
                        program
                            .passes
                            .first()
                            .unwrap()
                            .name
                            .contains("real_even_pack")
                    );
                    assert!(
                        program
                            .passes
                            .last()
                            .unwrap()
                            .name
                            .contains("real_even_postprocess")
                    );
                }
                crate::TransformKind::ComplexToReal => {
                    assert!(
                        program
                            .passes
                            .first()
                            .unwrap()
                            .name
                            .contains("real_even_inverse_preprocess")
                    );
                    assert!(
                        program
                            .passes
                            .last()
                            .unwrap()
                            .name
                            .contains("real_even_unpack")
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn bluestein_even_real_program_fuses_outer_boundaries() {
        let length = 206usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        for transform in [
            crate::TransformKind::RealToComplex,
            crate::TransformKind::ComplexToReal,
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_transform(transform)
                    .with_inverse_normalization(transform == crate::TransformKind::ComplexToReal)
                    .with_tuning(tuning),
            )
            .unwrap();
            let ir = crate::RealFftIr::build(&plan, device()).unwrap();
            assert!(ir.fused_even_input_stockham_kernel().unwrap().is_none());
            assert!(ir.fused_even_recursive_ir().unwrap().is_none());
            assert!(ir.fused_even_bluestein_ir().unwrap().is_some());
            let program = ProgramIr::real_fft(&ir).unwrap();
            assert!(program.passes.iter().all(|pass| {
                !pass.name.contains("pack_even_odd")
                    && !pass.name.contains("preprocess_even_half")
                    && !pass.name.contains("postprocess_even_half")
                    && !pass.name.contains("unpack_even_odd")
            }));
            assert!(program.resources.iter().all(|resource| {
                !resource.name.contains("real_preprocess")
                    && !resource.name.contains("real_transform")
            }));
            match transform {
                crate::TransformKind::RealToComplex => {
                    assert!(
                        program
                            .passes
                            .first()
                            .unwrap()
                            .name
                            .contains("real_even_pack")
                    );
                    assert!(
                        program
                            .passes
                            .last()
                            .unwrap()
                            .name
                            .contains("real_even_postprocess")
                    );
                }
                crate::TransformKind::ComplexToReal => {
                    assert!(
                        program
                            .passes
                            .first()
                            .unwrap()
                            .name
                            .contains("real_even_inverse_preprocess")
                    );
                    assert!(
                        program
                            .passes
                            .last()
                            .unwrap()
                            .name
                            .contains("real_even_unpack")
                    );
                }
                _ => unreachable!(),
            }
            assert!(
                program
                    .resources
                    .iter()
                    .any(|resource| resource.name.contains("bluestein_scratch"))
            );
        }
    }

    #[test]
    fn multidimensional_program_flattens_axis_pack_transform_scatter() {
        let plan = FftPlan::build(FftConfig::new(vec![3, 4]).with_batch_count(2)).unwrap();
        let ir = crate::NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::nd_fft(&ir).unwrap();
        assert_eq!(program.passes.len(), 6);
        assert_eq!(program.input_resource().unwrap().elements, 24);
        assert_eq!(program.output_resource().unwrap().elements, 24);
        assert!(program.passes[0].name.contains("nd_pack_axis_1"));
        assert!(program.passes[2].name.contains("nd_scatter_axis_1"));
        assert!(program.passes[3].name.contains("nd_pack_axis_0"));
        assert_eq!(program.passes[4].dispatch.x, 2);
        let outer = ir
            .axes
            .iter()
            .find(|axis| axis.axis == 0)
            .expect("outer ND axis");
        let crate::OneDimFftIr::Recursive(outer_recursive) = &outer.transform else {
            panic!("outer ND Stockham axis unexpectedly selected Bluestein");
        };
        let crate::RecursiveFftNodeIr::Stockham(outer_kernel) = &outer_recursive.root else {
            panic!("outer ND Stockham axis unexpectedly selected Rader");
        };
        assert_eq!(
            [outer_kernel.workgroup_size.x, outer_kernel.workgroup_size.y],
            [4, 1]
        );
        assert_eq!(outer_kernel.dispatch.x, 2);
        let memory = program.memory_plan().unwrap();
        assert!(memory.allocations.len() < program.resources.len());
    }

    #[test]
    fn omit_dimension_removes_only_selected_axis_from_program_graph() {
        let plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_omit_dimension(1, true)
                .unwrap(),
        )
        .unwrap();
        let ir = crate::NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::nd_fft(&ir).unwrap();
        assert_eq!(
            ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(program.input_resource().unwrap().elements, 24);
        assert_eq!(program.output_resource().unwrap().elements, 24);
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name.contains("nd_pack_axis_0"))
        );
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name.contains("nd_scatter_axis_0"))
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("axis_1"))
        );

        let dd_plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_precision(Precision::DoubleDouble)
                .with_omit_dimension(1, true)
                .unwrap(),
        )
        .unwrap();
        let dd_ir = crate::DoubleDoubleNdFftIr::build(&dd_plan, Direction::Forward).unwrap();
        let dd_program = ProgramIr::double_double_nd(&dd_ir).unwrap();
        assert_eq!(
            dd_ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(dd_program.input_resource().unwrap().elements, 24);
        assert_eq!(dd_program.output_resource().unwrap().elements, 24);
        assert!(
            !dd_program
                .passes
                .iter()
                .any(|pass| pass.name.contains("axis_1"))
        );
    }

    #[test]
    fn formatted_batch_stride_nd_expands_only_external_allocations() {
        let plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_input_buffer_batch_stride(17)
                .with_output_buffer_batch_stride(19),
        )
        .unwrap();
        let ir = crate::NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::nd_fft(&ir).unwrap();
        let input = program.input_resource().unwrap();
        let output = program.output_resource().unwrap();
        assert_eq!(input.elements, 34);
        assert_eq!(output.elements, 38);
        assert_eq!(input.external_layout.unwrap().logical_len, 12);
        assert_eq!(input.external_layout.unwrap().physical_stride, 17);
        assert_eq!(output.external_layout.unwrap().logical_len, 12);
        assert_eq!(output.external_layout.unwrap().physical_stride, 19);
        assert!(
            program
                .resources
                .iter()
                .filter(|resource| resource.kind == ProgramResourceKind::Scratch)
                .all(|resource| resource.elements == 24
                    || resource.kind == ProgramResourceKind::LookupTable)
        );
        program.validate().unwrap();
    }

    #[test]
    fn formatted_tensor_stride_nd_exposes_physical_external_batches_and_dense_scratch() {
        let plan = FftPlan::build(
            FftConfig::new(vec![2, 3, 4])
                .with_batch_count(2)
                .with_input_buffer_axis_stride(1, 6)
                .unwrap()
                .with_input_buffer_axis_stride(0, 20)
                .unwrap()
                .with_output_buffer_axis_stride(1, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 24)
                .unwrap(),
        )
        .unwrap();
        let ir = crate::NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::nd_fft(&ir).unwrap();
        let input = program.input_resource().unwrap();
        let output = program.output_resource().unwrap();
        assert_eq!(input.elements, 80);
        assert_eq!(output.elements, 96);
        assert_eq!(input.external_layout.unwrap().logical_len, 40);
        assert_eq!(input.external_layout.unwrap().physical_stride, 40);
        assert_eq!(output.external_layout.unwrap().logical_len, 48);
        assert_eq!(output.external_layout.unwrap().physical_stride, 48);
        assert!(
            program
                .resources
                .iter()
                .filter(|resource| resource.kind == ProgramResourceKind::Scratch)
                .all(|resource| resource.elements == 48)
        );
        program.validate().unwrap();
    }

    #[test]
    fn formatted_zero_padding_nd_program_keeps_padding_on_dense_compute_boundaries() {
        let mut device = device();
        device.supports_f64 = true;
        for (domain, input_boundary) in [
            (ZeroPaddingDomain::Spatial, true),
            (ZeroPaddingDomain::Frequency, false),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![3, 4])
                    .with_batch_count(2)
                    .with_precision(Precision::F64ComputeF32Storage)
                    .with_input_buffer_axis_stride(0, 7)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, 9)
                    .unwrap()
                    .with_zero_padding(1, 1, 3)
                    .unwrap()
                    .with_zero_padding_domain(domain),
            )
            .unwrap();
            let ir = crate::NdFftIr::build(&plan, Direction::Forward, device).unwrap();
            let program = ProgramIr::nd_fft(&ir).unwrap();
            assert_eq!(program.input_resource().unwrap().scalar, ScalarType::F32);
            assert_eq!(program.output_resource().unwrap().scalar, ScalarType::F32);
            assert_eq!(program.input_resource().unwrap().elements, 42);
            assert_eq!(program.output_resource().unwrap().elements, 54);
            let names = program
                .passes
                .iter()
                .map(|pass| pass.name.as_str())
                .collect::<Vec<_>>();
            if input_boundary {
                assert_eq!(names[0], "vkfft_nd_gather_formatted_input_before_zero_pad");
                assert!(names[1].contains("nd_zero_pad_forward_input"));
                assert!(ir.input_formatted_copy.is_some());
                assert!(ir.output_formatted_copy.is_none());
            } else {
                assert!(names[names.len() - 2].contains("nd_zero_pad_forward_output"));
                assert_eq!(
                    names[names.len() - 1],
                    "vkfft_nd_scatter_formatted_output_after_zero_pad"
                );
                assert!(ir.input_formatted_copy.is_none());
                assert!(ir.output_formatted_copy.is_some());
            }
            let zero_pad = ir.zero_pad_pass.as_ref().unwrap();
            assert_eq!(zero_pad.input_storage_scalar, ScalarType::F64);
            assert_eq!(zero_pad.output_storage_scalar, ScalarType::F64);
            assert!(program.resources.iter().any(|resource| {
                resource.kind == ProgramResourceKind::Scratch
                    && resource.scalar == ScalarType::F64
                    && resource.elements == 24
                    && resource.name.contains("formatted")
            }));
            program.validate().unwrap();
        }
    }

    #[test]
    fn formatted_real_tensor_strides_insert_physical_gather_and_scatter_around_dense_program() {
        for (transform, input_stride, output_stride, input_elements, output_elements) in [
            (
                TransformKind::RealToComplex,
                11usize,
                7usize,
                66usize,
                42usize,
            ),
            (
                TransformKind::ComplexToReal,
                7usize,
                11usize,
                42usize,
                66usize,
            ),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![3usize, 8])
                    .with_batch_count(2)
                    .with_transform(transform)
                    .with_inverse_normalization(transform == TransformKind::ComplexToReal)
                    .with_input_buffer_axis_stride(0, input_stride)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, output_stride)
                    .unwrap(),
            )
            .unwrap();
            let ir = crate::NdRealFftIr::build(&plan, device()).unwrap();
            let program = ProgramIr::nd_real_fft(&ir).unwrap();
            let input = program.input_resource().unwrap();
            let output = program.output_resource().unwrap();
            assert_eq!(input.elements, input_elements);
            assert_eq!(output.elements, output_elements);
            assert_eq!(
                input.external_layout.unwrap().logical_len,
                ir.input_external_layout.batch_stride
            );
            assert_eq!(
                output.external_layout.unwrap().logical_len,
                ir.output_external_layout.batch_stride
            );
            assert!(
                program
                    .passes
                    .first()
                    .unwrap()
                    .name
                    .contains("nd_real_gather_formatted_input")
            );
            assert!(
                program
                    .passes
                    .last()
                    .unwrap()
                    .name
                    .contains("nd_real_scatter_formatted_output")
            );
            assert!(program.resources.iter().any(|resource| {
                resource.name.contains("nd_real_formatted_input")
                    && resource.elements == ir.input_tensor_len() * ir.batch_count
            }));
            assert!(program.resources.iter().any(|resource| {
                resource.name.contains("nd_real_formatted_output")
                    && resource.elements == ir.output_tensor_len() * ir.batch_count
            }));
            program.validate().unwrap();
        }
    }

    #[test]
    fn formatted_real_zero_padding_program_keeps_pad_between_dense_copies() {
        let mut device = device();
        device.supports_f64 = true;
        for (transform, input_elements, output_elements, pad_at_start) in [
            (TransformKind::RealToComplex, 66usize, 42usize, true),
            (TransformKind::ComplexToReal, 42usize, 66usize, false),
        ] {
            let (input_stride, output_stride) = match transform {
                TransformKind::RealToComplex => (11usize, 7usize),
                TransformKind::ComplexToReal => (7usize, 11usize),
                _ => unreachable!(),
            };
            let plan = FftPlan::build(
                FftConfig::new(vec![3usize, 8])
                    .with_batch_count(2)
                    .with_transform(transform)
                    .with_precision(Precision::F64ComputeF32Storage)
                    .with_inverse_normalization(transform == TransformKind::ComplexToReal)
                    .with_input_buffer_axis_stride(0, input_stride)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, output_stride)
                    .unwrap()
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_zero_padding(1, 2, 4)
                    .unwrap(),
            )
            .unwrap();
            let ir = crate::NdRealFftIr::build(&plan, device).unwrap();
            let program = ProgramIr::nd_real_fft(&ir).unwrap();
            assert_eq!(program.input_resource().unwrap().elements, input_elements);
            assert_eq!(program.output_resource().unwrap().elements, output_elements);
            assert_eq!(program.input_resource().unwrap().scalar, ScalarType::F32);
            assert_eq!(program.output_resource().unwrap().scalar, ScalarType::F32);
            let zero_pad = ir.zero_pad_pass.as_ref().unwrap();
            assert_eq!(zero_pad.input_storage_scalar, ScalarType::F64);
            assert_eq!(zero_pad.output_storage_scalar, ScalarType::F64);
            let names = program
                .passes
                .iter()
                .map(|pass| pass.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                names.first().copied(),
                Some("vkfft_nd_real_gather_formatted_input")
            );
            assert_eq!(
                names.last().copied(),
                Some("vkfft_nd_real_scatter_formatted_output")
            );
            if pad_at_start {
                assert!(names[1].contains("nd_zero_pad_forward_input"));
            } else {
                assert!(names[names.len() - 2].contains("nd_zero_pad_inverse_output"));
            }
            assert!(program.resources.iter().any(|resource| {
                resource.kind == ProgramResourceKind::Scratch
                    && resource.scalar == ScalarType::F64
                    && resource.name.contains("formatted")
            }));
            program.validate().unwrap();
        }
    }

    #[test]
    fn formatted_nd_r2r_program_wraps_dense_pipeline_with_physical_copies() {
        let plan = FftPlan::build(
            FftConfig::new(vec![2usize, 3, 4])
                .with_batch_count(2)
                .with_transform(TransformKind::Dct(crate::DctType::II))
                .with_input_buffer_axis_stride(1, 6)
                .unwrap()
                .with_input_buffer_axis_stride(0, 20)
                .unwrap()
                .with_output_buffer_axis_stride(1, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 24)
                .unwrap(),
        )
        .unwrap();
        let ir = crate::NdR2rIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::nd_r2r(&ir).unwrap();
        let input = program.input_resource().unwrap();
        let output = program.output_resource().unwrap();
        assert_eq!(input.elements, 80);
        assert_eq!(output.elements, 96);
        assert_eq!(input.external_layout.unwrap().physical_stride, 40);
        assert_eq!(output.external_layout.unwrap().physical_stride, 48);
        assert_eq!(
            program.passes.first().unwrap().name,
            "vkfft_nd_r2r_gather_formatted_input"
        );
        assert_eq!(
            program.passes.last().unwrap().name,
            "vkfft_nd_r2r_scatter_formatted_output"
        );
        let formatted_input = program
            .resources
            .iter()
            .find(|resource| resource.name.starts_with("nd_r2r_formatted_input"))
            .unwrap();
        let formatted_output = program
            .resources
            .iter()
            .find(|resource| resource.name.starts_with("nd_r2r_formatted_output"))
            .unwrap();
        assert_eq!(formatted_input.kind, ProgramResourceKind::Scratch);
        assert_eq!(formatted_output.kind, ProgramResourceKind::Scratch);
        assert_eq!(formatted_input.elements, 48);
        assert_eq!(formatted_output.elements, 48);
        assert!(formatted_input.external_layout.is_none());
        assert!(formatted_output.external_layout.is_none());
        program.validate().unwrap();
    }

    #[test]
    fn formatted_double_double_nd_c2c_program_keeps_dense_dd_scratch() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![2usize, 3, 4])
                    .with_batch_count(2)
                    .with_precision(precision)
                    .with_input_buffer_axis_stride(1, 6)
                    .unwrap()
                    .with_input_buffer_axis_stride(0, 20)
                    .unwrap()
                    .with_output_buffer_axis_stride(1, 7)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, 24)
                    .unwrap(),
            )
            .unwrap();
            let ir = crate::DoubleDoubleNdFftIr::build(&plan, Direction::Forward).unwrap();
            let program = ProgramIr::double_double_nd(&ir).unwrap();
            let input = program.input_resource().unwrap();
            let output = program.output_resource().unwrap();
            assert_eq!(input.scalar, external_scalar);
            assert_eq!(output.scalar, external_scalar);
            assert_eq!(input.elements, 80);
            assert_eq!(output.elements, 96);
            assert_eq!(input.external_layout.unwrap().physical_stride, 40);
            assert_eq!(output.external_layout.unwrap().physical_stride, 48);
            assert_eq!(
                program.passes.first().unwrap().name,
                "vkfft_dd_nd_gather_formatted_input"
            );
            assert_eq!(
                program.passes.last().unwrap().name,
                "vkfft_dd_nd_scatter_formatted_output"
            );
            for name in ["double_double_nd_tensor_a", "double_double_nd_tensor_b"] {
                let scratch = program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap();
                assert_eq!(scratch.kind, ProgramResourceKind::Scratch);
                assert_eq!(scratch.scalar, ScalarType::DoubleDouble);
                assert_eq!(scratch.elements, 48);
                assert!(scratch.external_layout.is_none());
            }
            program.validate().unwrap();
        }
    }

    #[test]
    fn formatted_double_double_real_and_r2r_programs_keep_dense_dd_scratch() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let real_plan = FftPlan::build(
                FftConfig::new(vec![2usize, 3, 8])
                    .with_batch_count(2)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(precision)
                    .with_input_buffer_axis_stride(1, 11)
                    .unwrap()
                    .with_input_buffer_axis_stride(0, 40)
                    .unwrap()
                    .with_output_buffer_axis_stride(1, 7)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, 24)
                    .unwrap(),
            )
            .unwrap();
            let real_ir = crate::DoubleDoubleNdRealFftIr::build(&real_plan).unwrap();
            let real_program = ProgramIr::double_double_nd_real(&real_ir).unwrap();
            let real_input = real_program.input_resource().unwrap();
            let real_output = real_program.output_resource().unwrap();
            assert_eq!(real_input.scalar, external_scalar);
            assert_eq!(real_output.scalar, external_scalar);
            assert_eq!(real_input.elements, 160);
            assert_eq!(real_output.elements, 96);
            assert_eq!(real_input.external_layout.unwrap().physical_stride, 80);
            assert_eq!(real_output.external_layout.unwrap().physical_stride, 48);
            assert_eq!(
                real_program.passes.first().unwrap().name,
                "vkfft_dd_nd_real_gather_formatted_input"
            );
            assert_eq!(
                real_program.passes.last().unwrap().name,
                "vkfft_dd_nd_real_scatter_formatted_output"
            );
            let full_scratch = real_program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_nd_real_full_scalar")
                .unwrap();
            assert_eq!(full_scratch.scalar, ScalarType::DoubleDouble);
            assert_eq!(full_scratch.elements, 96);
            let compact_scratch = real_program
                .resources
                .iter()
                .find(|resource| resource.name == "double_double_nd_real_compact_a")
                .unwrap();
            assert_eq!(compact_scratch.scalar, ScalarType::DoubleDouble);
            assert_eq!(compact_scratch.elements, 60);
            real_program.validate().unwrap();

            let r2r_plan = FftPlan::build(
                FftConfig::new(vec![2usize, 3, 4])
                    .with_batch_count(2)
                    .with_transform(TransformKind::Dct(crate::DctType::II))
                    .with_precision(precision)
                    .with_input_buffer_axis_stride(1, 6)
                    .unwrap()
                    .with_input_buffer_axis_stride(0, 20)
                    .unwrap()
                    .with_output_buffer_axis_stride(1, 7)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, 24)
                    .unwrap(),
            )
            .unwrap();
            let r2r_ir = crate::DoubleDoubleNdR2rIr::build(&r2r_plan, Direction::Forward).unwrap();
            let r2r_program = ProgramIr::double_double_nd_r2r(&r2r_ir).unwrap();
            let r2r_input = r2r_program.input_resource().unwrap();
            let r2r_output = r2r_program.output_resource().unwrap();
            assert_eq!(r2r_input.scalar, external_scalar);
            assert_eq!(r2r_output.scalar, external_scalar);
            assert_eq!(r2r_input.elements, 80);
            assert_eq!(r2r_output.elements, 96);
            assert_eq!(r2r_input.external_layout.unwrap().physical_stride, 40);
            assert_eq!(r2r_output.external_layout.unwrap().physical_stride, 48);
            assert_eq!(
                r2r_program.passes.first().unwrap().name,
                "vkfft_dd_nd_r2r_gather_formatted_input"
            );
            assert_eq!(
                r2r_program.passes.last().unwrap().name,
                "vkfft_dd_nd_r2r_scatter_formatted_output"
            );
            for name in [
                "double_double_nd_r2r_tensor_a",
                "double_double_nd_r2r_tensor_b",
            ] {
                let scratch = r2r_program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap();
                assert_eq!(scratch.scalar, ScalarType::DoubleDouble);
                assert_eq!(scratch.elements, 48);
            }
            r2r_program.validate().unwrap();
        }
    }

    #[test]
    fn double_double_omit_dimension_programs_drop_omitted_axis_passes() {
        let real_plan = FftPlan::build(
            FftConfig::new(vec![3usize, 8])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble)
                .with_omit_dimension(0, true)
                .unwrap(),
        )
        .unwrap();
        let real_ir = crate::DoubleDoubleNdRealFftIr::build(&real_plan).unwrap();
        assert!(real_ir.complex_axes.is_empty());
        let real_program = ProgramIr::double_double_nd_real(&real_ir).unwrap();
        assert!(
            !real_program
                .passes
                .iter()
                .any(|pass| pass.name.contains("nd_real_axis_0"))
        );
        assert!(
            real_program
                .passes
                .iter()
                .any(|pass| pass.name.contains("nd_real_last_axis"))
        );
        real_program.validate().unwrap();

        let r2r_plan = FftPlan::build(
            FftConfig::new(vec![3usize, 4])
                .with_transform(TransformKind::Dct(crate::DctType::II))
                .with_precision(Precision::DoubleDouble)
                .with_omit_dimension(1, true)
                .unwrap(),
        )
        .unwrap();
        let r2r_ir = crate::DoubleDoubleNdR2rIr::build(&r2r_plan, Direction::Forward).unwrap();
        assert_eq!(
            r2r_ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![0]
        );
        let r2r_program = ProgramIr::double_double_nd_r2r(&r2r_ir).unwrap();
        assert!(
            !r2r_program
                .passes
                .iter()
                .any(|pass| pass.name.contains("nd_r2r_axis_1"))
        );
        assert!(
            r2r_program
                .passes
                .iter()
                .any(|pass| pass.name.contains("nd_r2r_axis_0"))
        );
        r2r_program.validate().unwrap();
    }

    #[test]
    fn f64_four_step_uses_one_lazy_full_period_twiddle_lut() {
        let length = 1_048_576usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 48 * 1024;
        profile.max_threads_per_block = 1024;
        profile.supports_f64 = true;
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::F64)).unwrap();
        let ir = crate::RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_some());
        let kernels = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert!(
            kernels
                .iter()
                .all(|kernel| kernel.twiddle_lut_len() == Some(length))
        );

        let program = ProgramIr::recursive_fft(&ir).unwrap();
        let root_luts = program
            .resources
            .iter()
            .filter(|resource| {
                matches!(
                    resource.initialization,
                    ProgramResourceInitialization::StockhamUnitRoots { len } if len == length
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(root_luts.len(), 1);
        assert_eq!(root_luts[0].elements, length);
        assert!(program.passes.iter().all(|pass| {
            pass.bindings
                .iter()
                .any(|binding| binding.role == BufferRole::TwiddleLookupTable)
        }));
        assert!(!program.resources.iter().any(|resource| {
            matches!(
                &resource.initialization,
                ProgramResourceInitialization::Complex64(values) if values.len() == length
            )
        }));
    }

    #[test]
    fn two_upload_four_step_program_collapses_root_to_two_fft_passes() {
        let length = 1_048_576usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 48 * 1024;
        profile.max_threads_per_block = 1024;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = crate::RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_some());

        let program = ProgramIr::recursive_fft(&ir).unwrap();
        assert_eq!(program.passes.len(), 2);
        assert_eq!(program.resources.len(), 3);
        assert_eq!(
            program
                .resources
                .iter()
                .filter(|resource| resource.kind == ProgramResourceKind::Scratch)
                .count(),
            1
        );
        assert!(program.passes[0].name.ends_with("four_step_upload_1"));
        assert!(program.passes[1].name.ends_with("four_step_upload_0"));
        assert_eq!(program.passes[0].dispatch.x, 256);
        assert_eq!(program.passes[1].dispatch.x, 256);
        program.validate().unwrap();
    }

    #[test]
    fn three_upload_four_step_program_collapses_root_to_three_fft_passes() {
        let length = 8_388_608usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 48 * 1024;
        profile.max_threads_per_block = 1024;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = crate::RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_some());

        let program = ProgramIr::recursive_fft(&ir).unwrap();
        assert_eq!(program.passes.len(), 3);
        assert_eq!(program.resources.len(), 3);
        let scratch = program
            .resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Scratch)
            .unwrap()
            .id;
        assert_eq!(scratch, ProgramResourceId(2));
        assert!(program.passes[0].name.ends_with("four_step_upload_2"));
        assert!(program.passes[1].name.ends_with("four_step_upload_1"));
        assert!(program.passes[2].name.ends_with("four_step_upload_0"));
        assert_eq!(
            program
                .passes
                .iter()
                .map(|pass| pass.dispatch.x)
                .collect::<Vec<_>>(),
            vec![2048, 4096, 2048]
        );
        assert_eq!(program.passes[0].bindings[0].resource, ProgramResourceId(0));
        assert_eq!(program.passes[0].bindings[1].resource, ProgramResourceId(1));
        assert_eq!(program.passes[1].bindings[0].resource, ProgramResourceId(1));
        assert_eq!(program.passes[1].bindings[1].resource, scratch);
        assert_eq!(program.passes[2].bindings[0].resource, scratch);
        assert_eq!(program.passes[2].bindings[1].resource, ProgramResourceId(1));
        program.validate().unwrap();
    }

    #[test]
    fn rader_three_upload_reuses_output_only_for_same_scalar_storage() {
        let length = 1_922usize;
        let base_profile = DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            max_threads_per_block: 32,
            max_workgroup_size: [32, 32, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };

        for (precision, expected_exchange_scratch, expected_output_scalar) in [
            (Precision::F32, 1usize, ScalarType::F32),
            (Precision::F16StorageF32Compute, 2usize, ScalarType::F16),
        ] {
            let mut profile = base_profile;
            if precision == Precision::F16StorageF32Compute {
                profile.shared_memory_bytes = 2 * 1024;
                profile.shared_memory_pow2_bytes = 2 * 1024;
            }
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            let ir = crate::RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            assert_eq!(ir.scalar, ScalarType::F32);
            assert_eq!(ir.external_scalar, expected_output_scalar);
            assert_eq!(
                ir.rader_forced_upload_schedule
                    .as_ref()
                    .expect("N1922 must retain the forced-Rader schedule")
                    .axis_split,
                vec![2, 31, 31]
            );
            assert_eq!(
                ir.four_step_rader_upload_nodes()
                    .unwrap()
                    .expect("N1922 must materialize three Rader Four-step uploads")
                    .len(),
                3
            );

            let program = ProgramIr::recursive_fft(&ir).unwrap();
            let output = program.output_resource().unwrap();
            assert_eq!(output.scalar, expected_output_scalar);
            let exchange = program
                .resources
                .iter()
                .filter(|resource| {
                    resource.kind == ProgramResourceKind::Scratch
                        && resource.name.starts_with("rader_four_step_exchange")
                })
                .collect::<Vec<_>>();
            assert_eq!(exchange.len(), expected_exchange_scratch, "{precision:?}");
            if precision == Precision::F32 {
                assert!(program.passes.iter().any(|pass| {
                    pass.bindings
                        .iter()
                        .any(|binding| binding.resource == output.id)
                }));
            } else {
                assert_ne!(exchange[0].id, exchange[1].id);
                assert!(
                    exchange
                        .iter()
                        .all(|resource| resource.scalar == ScalarType::F32)
                );
            }
            program.validate().unwrap();
        }
    }

    #[test]
    fn double_double_three_upload_reuses_output_only_for_dd_storage() {
        let stockham_device = DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let forced_rader_device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };

        for (precision, expected_scratch, expected_output_scalar) in [
            (Precision::DoubleDouble, 1usize, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, 2usize, ScalarType::F64),
        ] {
            let stockham_plan = FftPlan::build_for_device(
                FftConfig::new(vec![8_192]).with_precision(precision),
                stockham_device,
            )
            .unwrap();
            let stockham_ir = crate::DoubleDoubleRecursiveFftIr::build_for_device(
                &stockham_plan,
                Direction::Forward,
                stockham_device,
            )
            .unwrap();
            assert_eq!(
                stockham_ir
                    .three_upload_four_step_plan
                    .expect("constrained DD N8192 must keep three-upload Four-step metadata")
                    .axis_split,
                [32, 16, 16]
            );
            let stockham_program = ProgramIr::double_double_recursive(&stockham_ir).unwrap();
            assert_eq!(
                stockham_program.output_resource().unwrap().scalar,
                expected_output_scalar
            );
            let stockham_exchange = stockham_program
                .resources
                .iter()
                .filter(|resource| {
                    resource.kind == ProgramResourceKind::Scratch
                        && resource
                            .name
                            .starts_with("double_double_three_upload_four_step_scratch")
                })
                .collect::<Vec<_>>();
            assert_eq!(stockham_exchange.len(), expected_scratch, "{precision:?}");
            assert!(
                stockham_exchange
                    .iter()
                    .all(|resource| resource.scalar == ScalarType::DoubleDouble)
            );
            stockham_program.validate().unwrap();

            let forced_length = 17usize * 47 * 128;
            let forced_plan =
                FftPlan::build(FftConfig::new(vec![forced_length]).with_precision(precision))
                    .unwrap();
            let forced_ir = crate::DoubleDoubleRecursiveFftIr::build_for_device(
                &forced_plan,
                Direction::Forward,
                forced_rader_device,
            )
            .unwrap();
            let forced_schedule = forced_ir
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD p47 composite must retain forced-Rader upload metadata");
            assert_eq!(forced_schedule.axis_split, vec![64, 47, 34]);
            assert_eq!(forced_schedule.upload_count, 3);
            let forced_program = ProgramIr::double_double_recursive(&forced_ir).unwrap();
            assert_eq!(
                forced_program.output_resource().unwrap().scalar,
                expected_output_scalar
            );
            let forced_exchange = forced_program
                .resources
                .iter()
                .filter(|resource| {
                    resource.kind == ProgramResourceKind::Scratch
                        && resource
                            .name
                            .starts_with("double_double_forced_rader_three_upload_scratch")
                })
                .collect::<Vec<_>>();
            assert_eq!(forced_exchange.len(), expected_scratch, "{precision:?}");
            assert!(
                forced_exchange
                    .iter()
                    .all(|resource| resource.scalar == ScalarType::DoubleDouble)
            );
            forced_program.validate().unwrap();
        }
    }

    #[test]
    fn memory_plan_reuses_dead_scratch_and_preserves_external_buffers() {
        let plan = FftPlan::build(FftConfig::new(vec![578])).unwrap();
        let ir = crate::RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let program = ProgramIr::recursive_fft(&ir).unwrap();
        let memory = program.memory_plan().unwrap();
        let input_allocation = memory
            .allocation_for(program.input_resource().unwrap().id)
            .unwrap();
        let output_allocation = memory
            .allocation_for(program.output_resource().unwrap().id)
            .unwrap();
        assert_ne!(input_allocation, output_allocation);
        assert_eq!(
            memory
                .allocations
                .iter()
                .filter(|allocation| allocation.kind == ProgramAllocationKind::LookupTable)
                .count(),
            1
        );
        assert!(memory.allocations.len() < program.resources.len());
        let logical_elements = program
            .resources
            .iter()
            .map(|resource| resource.elements)
            .sum::<usize>();
        assert!(memory.allocated_elements().unwrap() < logical_elements);
        assert!(memory.allocations.iter().any(|allocation| allocation.kind
            == ProgramAllocationKind::Scratch
            && allocation.resources.len() > 1));
    }

    #[test]
    fn double_double_r2r_program_uses_scalar_boundaries_and_dd_internal_resources() {
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let external_scalar = if precision == Precision::DoubleDouble {
                ScalarType::DoubleDouble
            } else {
                ScalarType::F64
            };
            for transform in [
                TransformKind::Dct(crate::DctType::I),
                TransformKind::Dst(crate::DstType::I),
                TransformKind::Dst(crate::DstType::II),
                TransformKind::Dst(crate::DstType::III),
                TransformKind::Dct(crate::DctType::II),
                TransformKind::Dct(crate::DctType::III),
                TransformKind::Dct(crate::DctType::IV),
                TransformKind::Dst(crate::DstType::IV),
            ] {
                let plan = FftPlan::build(
                    FftConfig::new(vec![9])
                        .with_batch_count(7)
                        .with_precision(precision)
                        .with_transform(transform)
                        .with_grouped_batch(0, 3)
                        .unwrap(),
                )
                .unwrap();
                let ir = crate::DoubleDoubleR2rIr::build(&plan, Direction::Forward).unwrap();
                let crate::DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } =
                    &ir.algorithm
                else {
                    panic!("all DD DCT/DST I-IV families must use FFT reduction");
                };
                assert_eq!(
                    *fft_len,
                    match transform {
                        TransformKind::Dct(crate::DctType::I) => 16,
                        TransformKind::Dst(crate::DstType::I) => 20,
                        TransformKind::Dct(crate::DctType::II | crate::DctType::III)
                        | TransformKind::Dst(crate::DstType::II | crate::DstType::III) => 9,
                        TransformKind::Dct(crate::DctType::IV)
                        | TransformKind::Dst(crate::DstType::IV) => 18,
                        _ => unreachable!(),
                    }
                );
                assert_eq!(ir.grouped_batch, 3);
                assert_eq!(fft.grouped_batch(), 3);
                assert_eq!(ir.batch_group_count(), 3);
                let child = ProgramIr::double_double_one_dim(fft).unwrap();
                let program = ProgramIr::double_double_r2r(&ir).unwrap();
                assert_eq!(program.passes.len(), child.passes.len() + 2);
                assert!(program.passes.iter().all(|pass| pass.dispatch.x == 3));
                for resource in [
                    program.input_resource().unwrap(),
                    program.output_resource().unwrap(),
                ] {
                    assert_eq!(resource.scalar, external_scalar);
                    assert_eq!(
                        resource.external_layout.unwrap().element_shape,
                        ProgramElementShape::Scalar
                    );
                }
                let phases = program
                    .resources
                    .iter()
                    .find(|resource| resource.name == "double_double_r2r_phases");
                if matches!(
                    transform,
                    TransformKind::Dct(
                        crate::DctType::II | crate::DctType::III | crate::DctType::IV
                    ) | TransformKind::Dst(
                        crate::DstType::II | crate::DstType::III | crate::DstType::IV
                    )
                ) {
                    let phases = phases.unwrap();
                    assert_eq!(phases.scalar, ScalarType::DoubleDouble);
                    assert_eq!(phases.elements, 9);
                    assert!(matches!(
                        phases.initialization,
                        ProgramResourceInitialization::ComplexDoubleDouble(_)
                    ));
                } else {
                    assert!(phases.is_none());
                }
                assert!(
                    program
                        .resources
                        .iter()
                        .skip(2)
                        .all(|resource| { resource.scalar == ScalarType::DoubleDouble })
                );
            }
        }
    }

    #[test]
    fn double_double_nd_r2r_program_preserves_scalar_tensor_and_complex_child_shapes() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![3, 4])
                    .with_batch_count(7)
                    .with_precision(precision)
                    .with_transform(TransformKind::Dct(crate::DctType::II))
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
            )
            .unwrap();
            let ir = crate::double_double_ir::DoubleDoubleNdR2rIr::build(&plan, Direction::Forward)
                .unwrap();
            let program = ProgramIr::double_double_nd_r2r(&ir).unwrap();
            assert!(ir.axes.iter().all(|axis| axis.grouped_batch == 3));
            assert_eq!(ir.axes[0].transform.grouped_batch, 3);
            assert_eq!(ir.axes[1].transform.grouped_batch, 3);
            assert!(program.passes.iter().any(|pass| pass.dispatch.x == 3));
            assert!(program.passes.iter().any(|pass| pass.dispatch.x == 7));
            assert!(program.passes.iter().any(|pass| pass.dispatch.x == 10));
            assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
            assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
            assert_eq!(
                program.input_resource().unwrap().element_shape(),
                ProgramElementShape::Scalar
            );
            assert_eq!(
                program.output_resource().unwrap().element_shape(),
                ProgramElementShape::Scalar
            );
            for resource_id in 2..=5 {
                let resource = &program.resources[resource_id];
                assert_eq!(resource.scalar, ScalarType::DoubleDouble);
                assert_eq!(resource.element_shape(), ProgramElementShape::Scalar);
                assert_eq!(resource.element_bytes(), 16);
            }
            assert!(program.resources.iter().any(
                |resource| resource.name.contains("r2r_fft_input")
                    && resource.element_shape() == ProgramElementShape::Complex
                    && resource.element_bytes() == 32
            ));
            assert!(program.passes.first().unwrap().name.contains("axis_1_pack"));
            assert!(
                program
                    .passes
                    .last()
                    .unwrap()
                    .name
                    .contains("axis_0_scatter")
            );
            assert!(
                program
                    .passes
                    .iter()
                    .any(|pass| pass.name.contains("axis_1_vkfft_dd_r2r_fft_dct2_pre"))
            );
            let memory = program.memory_plan().unwrap();
            for resource_id in 2..=5 {
                let allocation_id = memory
                    .allocation_for(ProgramResourceId(resource_id))
                    .unwrap();
                let allocation = &memory.allocations[allocation_id.0];
                let first = allocation.resources.first().copied().unwrap();
                assert_eq!(
                    program.resources[first.0].element_shape(),
                    ProgramElementShape::Scalar
                );
            }
            for allocation in &memory.allocations {
                let first_shape = allocation
                    .resources
                    .first()
                    .map(|resource| program.resources[resource.0].element_shape())
                    .unwrap();
                assert!(
                    allocation
                        .resources
                        .iter()
                        .all(
                            |resource| program.resources[resource.0].element_shape() == first_shape
                        ),
                    "scratch alias mixed scalar and complex shapes"
                );
            }
        }
    }

    #[test]
    fn double_double_nd_r2r_padding_stays_fused_in_scalar_tensor_boundaries() {
        for (precision, external_scalar) in [
            (Precision::DoubleDouble, ScalarType::DoubleDouble),
            (Precision::DoubleDoubleF64Storage, ScalarType::F64),
        ] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let config = FftConfig::new(vec![3, 4])
                    .with_batch_count(7)
                    .with_precision(precision)
                    .with_transform(TransformKind::Dct(crate::DctType::II))
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap()
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_zero_padding(1, 1, 3)
                    .unwrap();
                let ir =
                    crate::DoubleDoubleNdR2rIr::build(&FftPlan::build(config).unwrap(), direction)
                        .unwrap();
                assert!(ir.has_spatial_zero_padding());
                let program = ProgramIr::double_double_nd_r2r(&ir).unwrap();
                assert!(ir.axes.iter().all(|axis| axis.grouped_batch == 3));
                assert!(ir.axes.iter().all(|axis| axis.transform.grouped_batch == 3));
                assert!(program.passes.iter().any(|pass| pass.dispatch.x == 3));
                assert!(program.passes.iter().any(|pass| pass.dispatch.x == 7));
                assert!(program.passes.iter().any(|pass| pass.dispatch.x == 10));
                assert_eq!(program.input_resource().unwrap().scalar, external_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, external_scalar);
                assert!(
                    !program
                        .passes
                        .iter()
                        .any(|pass| pass.name.contains("zero_pad"))
                );
                for resource_id in 2..=5 {
                    let resource = &program.resources[resource_id];
                    assert_eq!(resource.scalar, ScalarType::DoubleDouble);
                    assert_eq!(resource.element_shape(), ProgramElementShape::Scalar);
                    assert_eq!(resource.element_bytes(), 16);
                }
                assert!(program.resources.iter().any(|resource| {
                    resource.name.contains("r2r_fft_input")
                        && resource.scalar == ScalarType::DoubleDouble
                        && resource.element_shape() == ProgramElementShape::Complex
                        && resource.element_bytes() == 32
                }));
                assert_eq!(
                    program.passes.first().unwrap().bindings[0].resource,
                    ProgramResourceId(0)
                );
                assert_eq!(
                    program.passes.last().unwrap().bindings[1].resource,
                    ProgramResourceId(1)
                );
            }
        }
    }
}
