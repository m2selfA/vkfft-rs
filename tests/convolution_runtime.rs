#![cfg_attr(
    not(any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )),
    allow(dead_code, unused_imports)
)]

use vkfft_rs::{
    Binary16, Complex32, Complex64, ConvolutionConjugation, ConvolutionIr, DeviceProfile,
    Direction, FftConfig, NdFormattedCopyOperation, OneDimFftIr, PlannerTuning, Precision,
    ProgramIr, ProgramResourceInitialization, ProgramResourceKind, ScalarType,
    execute_convolution_ir,
};

const N: usize = 16;
const BATCH: usize = 2;

fn kernel() -> Vec<Complex64> {
    (0..N)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.19 * x).cos() + 0.01 * x, (0.13 * x).sin() - 0.008 * x)
        })
        .collect()
}

fn kernel_spectrum() -> Vec<Complex64> {
    vkfft_rs::reference::dft(&kernel(), Direction::Forward, false)
}

fn input64() -> Vec<Complex64> {
    (0..N * BATCH)
        .map(|index| {
            let batch = index / N;
            let local = (index % N) as f64;
            Complex64::new(
                batch as f64 * 0.7 + (0.11 * local).sin() + 0.003 * local,
                -(batch as f64) * 0.4 + (0.07 * local).cos() - 0.005 * local,
            )
        })
        .collect()
}

fn circular_convolution(input: &[Complex64], kernel: &[Complex64]) -> Vec<Complex64> {
    (0..N)
        .map(|out| {
            (0..N).fold(Complex64::default(), |sum, index| {
                sum + input[index] * kernel[(out + N - index) % N]
            })
        })
        .collect()
}

fn expected64() -> Vec<Complex64> {
    let input = input64();
    let kernel = kernel();
    let mut output = Vec::with_capacity(N * BATCH);
    for batch in 0..BATCH {
        let base = batch * N;
        output.extend(circular_convolution(&input[base..base + N], &kernel));
    }
    output
}

fn config(precision: Precision) -> FftConfig {
    FftConfig::new(vec![N])
        .with_batch_count(BATCH)
        .with_precision(precision)
        .with_convolution(true)
}

fn assert_close64(actual: &[Complex64], expected: &[Complex64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
        .fold(0.0, f64::max);
    assert!(error <= tolerance, "{label} convolution error {error:e}");
}

fn assert_close32(actual: &[Complex32], expected: &[Complex64], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let dr = actual.re - expected.re as f32;
            let di = actual.im - expected.im as f32;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(error <= tolerance, "{label} convolution error {error:e}");
}

fn build(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    ConvolutionIr::build_from_spectrum(config(precision), kernel_spectrum(), profile).unwrap()
}

fn mixed_input64() -> Vec<Complex64> {
    input64()[..N].to_vec()
}

fn build_mixed(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![N])
            .with_precision(precision)
            .with_convolution(true),
        kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn quantize_mixed_caller(values: &[Complex64], precision: Precision) -> Vec<Complex64> {
    values
        .iter()
        .map(|value| match precision {
            Precision::F16StorageF32Compute => Complex64::new(
                Binary16::from_f32(value.re as f32).to_f32() as f64,
                Binary16::from_f32(value.im as f32).to_f32() as f64,
            ),
            Precision::F64ComputeF32Storage => {
                Complex64::new(value.re as f32 as f64, value.im as f32 as f64)
            }
            _ => *value,
        })
        .collect()
}

fn mixed_expected(ir: &ConvolutionIr, precision: Precision) -> Vec<Complex64> {
    let caller_input = quantize_mixed_caller(&mixed_input64(), precision);
    let compute_output = execute_convolution_ir(ir, &caller_input).unwrap();
    quantize_mixed_caller(&compute_output, precision)
}

const LARGE_N: usize = 8_192;

fn large_input64() -> Vec<Complex64> {
    (0..LARGE_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                (0.013 * x).sin() + 1.0e-5 * x,
                (0.019 * x).cos() - 8.0e-6 * x,
            )
        })
        .collect()
}

fn large_kernel_spectrum() -> Vec<Complex64> {
    (0..LARGE_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(0.9 + 0.05 * (0.003 * x).cos(), 0.04 * (0.007 * x).sin())
        })
        .collect()
}

fn build_large_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    // Freeze the same 32 KiB scheduler surface used by the pinned-upstream N8192
    // counterexample while retaining the runtime's real backend/vendor capabilities.
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![LARGE_N])
            .with_precision(Precision::F32)
            .with_convolution(true),
        large_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_stockham());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);
    ir
}

fn large_multi_kernel_spectrum() -> Vec<Complex64> {
    (0..MULTI_KERNEL_COUNT * LARGE_N)
        .map(|index| {
            let kernel_id = index / LARGE_N;
            let x = (index % LARGE_N) as f64;
            Complex64::new(
                0.73 + 0.08 * kernel_id as f64 + 0.03 * (0.004 * x).cos(),
                0.11 + 0.02 * kernel_id as f64 - 0.025 * (0.006 * x).sin(),
            )
        })
        .collect()
}

fn build_large_multi_kernel_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![LARGE_N])
            .with_precision(Precision::F32)
            .with_convolution(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        large_multi_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_multi_kernel_stockham());
    assert_eq!(ir.forward_fft.batch_count(), 1);
    assert_eq!(ir.inverse_fft.batch_count(), MULTI_KERNEL_COUNT);
    let step = ir.fused_two_upload_multi_kernel_stockham.as_ref().unwrap();
    assert_eq!(step.required_shared_memory_bytes, 32 * 1024);
    let program = ProgramIr::convolution(&ir).unwrap();
    assert_eq!(program.passes.len(), 3);
    assert_eq!(program.resources[0].elements, LARGE_N);
    assert_eq!(program.resources[1].elements, MULTI_KERNEL_COUNT * LARGE_N);
    ir
}

fn assert_large_two_upload_result(
    actual: &[Complex32],
    ir: &ConvolutionIr,
    input64: &[Complex64],
    label: &str,
) {
    let expected = execute_convolution_ir(ir, input64).unwrap();
    assert_close32(actual, &expected, 4.0e-3, label);
}

const SMOOTH_TWO_UPLOAD_N: usize = 5_760;

fn smooth_two_upload_input64() -> Vec<Complex64> {
    (0..SMOOTH_TWO_UPLOAD_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                (0.014 * x).sin() + 1.2e-5 * x,
                (0.021 * x).cos() - 9.0e-6 * x,
            )
        })
        .collect()
}

fn smooth_two_upload_kernel_spectrum() -> Vec<Complex64> {
    (0..SMOOTH_TWO_UPLOAD_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(0.78 + 0.06 * (0.007 * x).cos(), 0.05 * (0.011 * x).sin())
        })
        .collect()
}

fn build_smooth_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![SMOOTH_TWO_UPLOAD_N])
            .with_precision(Precision::F32)
            .with_convolution(true),
        smooth_two_upload_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_stockham());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);
    ir
}

fn assert_smooth_two_upload_result(
    actual: &[Complex32],
    ir: &ConvolutionIr,
    input64: &[Complex64],
    label: &str,
) {
    let expected = execute_convolution_ir(ir, input64).unwrap();
    assert_close32(actual, &expected, 4.0e-3, label);
}

const F64_TWO_UPLOAD_N: usize = 4_096;

fn f64_two_upload_input64() -> Vec<Complex64> {
    (0..F64_TWO_UPLOAD_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                (0.017 * x).sin() + 1.0e-5 * x,
                (0.023 * x).cos() - 7.0e-6 * x,
            )
        })
        .collect()
}

fn f64_two_upload_kernel_spectrum() -> Vec<Complex64> {
    (0..F64_TWO_UPLOAD_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(0.85 + 0.04 * (0.005 * x).cos(), 0.03 * (0.009 * x).sin())
        })
        .collect()
}

fn build_f64_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![F64_TWO_UPLOAD_N])
            .with_precision(Precision::F64)
            .with_convolution(true),
        f64_two_upload_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_stockham());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);
    ir
}

fn f64_multi_kernel_two_upload_spectrum() -> Vec<Complex64> {
    (0..MULTI_KERNEL_COUNT * F64_TWO_UPLOAD_N)
        .map(|index| {
            let kernel_id = index / F64_TWO_UPLOAD_N;
            let x = (index % F64_TWO_UPLOAD_N) as f64;
            Complex64::new(
                0.81 + 0.06 * kernel_id as f64 + 0.025 * (0.006 * x).cos(),
                0.09 + 0.018 * kernel_id as f64 - 0.02 * (0.01 * x).sin(),
            )
        })
        .collect()
}

fn build_f64_multi_kernel_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![F64_TWO_UPLOAD_N])
            .with_precision(Precision::F64)
            .with_convolution(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        f64_multi_kernel_two_upload_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_multi_kernel_stockham());
    let step = ir.fused_two_upload_multi_kernel_stockham.as_ref().unwrap();
    assert_eq!(step.required_shared_memory_bytes, 16 * 1024);
    assert!(step.special_twiddle_lut_len().unwrap().is_some());
    let program = ProgramIr::convolution(&ir).unwrap();
    assert_eq!(program.passes.len(), 3);
    assert_eq!(program.resources[0].elements, F64_TWO_UPLOAD_N);
    assert_eq!(
        program.resources[1].elements,
        MULTI_KERNEL_COUNT * F64_TWO_UPLOAD_N
    );
    ir
}

fn assert_f64_two_upload_result(
    actual: &[Complex64],
    ir: &ConvolutionIr,
    input: &[Complex64],
    label: &str,
) {
    let expected = execute_convolution_ir(ir, input).unwrap();
    assert_close64(actual, &expected, 3.0e-8, label);
}

fn build_conjugated_cross_power_small(profile: DeviceProfile) -> ConvolutionIr {
    ConvolutionIr::build_from_spectrum(
        config(Precision::F32)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn build_conjugated_cross_power_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![LARGE_N])
            .with_precision(Precision::F32)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        large_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_stockham());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);
    ir
}

fn build_f64_conjugated_cross_power_small(profile: DeviceProfile) -> ConvolutionIr {
    ConvolutionIr::build_from_spectrum(
        config(Precision::F64)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn build_f64_conjugated_cross_power_two_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 32 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![F64_TWO_UPLOAD_N])
            .with_precision(Precision::F64)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        f64_two_upload_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_two_upload_stockham());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);
    ir
}

fn matrix_input64(matrix_size: usize) -> Vec<Complex64> {
    (0..BATCH * matrix_size * N)
        .map(|index| {
            let local = (index % N) as f64;
            let coordinate = (index / N) % matrix_size;
            let batch = index / (matrix_size * N);
            Complex64::new(
                0.35 * batch as f64 + 0.12 * coordinate as f64 + (0.1 * local).sin(),
                -0.18 * batch as f64 + 0.08 * coordinate as f64 + (0.16 * local).cos(),
            )
        })
        .collect()
}

fn matrix_kernel_spectrum(matrix_size: usize, symmetric: bool) -> Vec<Complex64> {
    let planes = if symmetric {
        matrix_size * (matrix_size + 1) / 2
    } else {
        matrix_size * matrix_size
    };
    (0..planes * N)
        .map(|index| {
            let plane = index / N;
            let local = (index % N) as f64;
            Complex64::new(
                0.7 + 0.06 * plane as f64 + 0.004 * local,
                0.17 + 0.015 * plane as f64 - 0.002 * local,
            )
        })
        .collect()
}

fn build_matrix(
    profile: DeviceProfile,
    precision: Precision,
    matrix_size: usize,
    symmetric: bool,
    semantic_policy: bool,
) -> ConvolutionIr {
    let mut config = FftConfig::new(vec![N])
        .with_batch_count(BATCH)
        .with_precision(precision)
        .with_convolution(true)
        .with_matrix_convolution(matrix_size)
        .with_symmetric_convolution_kernel(symmetric);
    if semantic_policy {
        config = config
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true);
    }
    let ir = ConvolutionIr::build_from_spectrum(
        config,
        matrix_kernel_spectrum(matrix_size, symmetric),
        profile,
    )
    .unwrap();
    assert_eq!(ir.coordinate_count, matrix_size);
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);
    ir
}

const MULTI_KERNEL_COUNT: usize = 3;

fn multi_kernel_input64() -> Vec<Complex64> {
    input64()[..N].to_vec()
}

fn multi_kernel_spectrum() -> Vec<Complex64> {
    (0..MULTI_KERNEL_COUNT * N)
        .map(|index| {
            let kernel_id = index / N;
            let local = (index % N) as f64;
            Complex64::new(
                0.72 + 0.09 * kernel_id as f64 + 0.004 * local,
                0.16 + 0.025 * kernel_id as f64 - 0.003 * local,
            )
        })
        .collect()
}

fn build_multi_kernel(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![N])
            .with_precision(precision)
            .with_convolution(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        multi_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert_eq!(ir.forward_fft.batch_count(), 1);
    assert_eq!(ir.inverse_fft.batch_count(), MULTI_KERNEL_COUNT);
    assert!(ir.has_fused_multi_kernel_stockham_step());
    let step = ir.fused_multi_kernel_stockham_step.as_ref().unwrap();
    let program = ProgramIr::convolution(&ir).unwrap();
    assert_eq!(program.passes.len(), 1);
    assert_eq!(program.passes[0].name, step.name);
    assert_eq!(program.resources[0].elements, N);
    assert_eq!(program.resources[1].elements, MULTI_KERNEL_COUNT * N);
    ir
}

fn matrix_multi_kernel_input64() -> Vec<Complex64> {
    matrix_input64(2)[..2 * N].to_vec()
}

fn matrix_multi_kernel_spectrum() -> Vec<Complex64> {
    let kernel_planes = 3usize;
    (0..MULTI_KERNEL_COUNT * kernel_planes * N)
        .map(|index| {
            let local = (index % N) as f64;
            let plane = (index / N) % kernel_planes;
            let kernel_id = index / (kernel_planes * N);
            Complex64::new(
                0.68 + 0.08 * kernel_id as f64 + 0.04 * plane as f64 + 0.003 * local,
                0.14 + 0.02 * kernel_id as f64 + 0.01 * plane as f64 - 0.002 * local,
            )
        })
        .collect()
}

fn build_matrix_multi_kernel(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![N])
            .with_precision(precision)
            .with_convolution(true)
            .with_matrix_convolution(2)
            .with_symmetric_convolution_kernel(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        matrix_multi_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert_eq!(ir.batch_count, 1);
    assert_eq!(ir.coordinate_count, 2);
    assert_eq!(ir.forward_fft.batch_count(), 2);
    assert_eq!(ir.inverse_fft.batch_count(), MULTI_KERNEL_COUNT * 2);
    assert!(ir.has_fused_matrix_stockham_step());
    let step = ir.fused_matrix_stockham_step.as_ref().unwrap();
    let program = ProgramIr::convolution(&ir).unwrap();
    assert_eq!(program.passes.len(), 1);
    assert_eq!(program.passes[0].name, step.name);
    assert_eq!(program.resources[0].elements, 2 * N);
    assert_eq!(program.resources[1].elements, MULTI_KERNEL_COUNT * 2 * N);
    ir
}

const RADER_N: usize = 47;

fn direct_rader_input64() -> Vec<Complex64> {
    (0..RADER_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.45 + (0.13 * x).sin() + 0.003 * x,
                -0.22 + (0.17 * x).cos() - 0.002 * x,
            )
        })
        .collect()
}

fn direct_rader_kernel_spectrum() -> Vec<Complex64> {
    (0..RADER_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.82 + 0.05 * (0.07 * x).cos(),
                0.19 + 0.03 * (0.11 * x).sin(),
            )
        })
        .collect()
}

fn build_direct_rader(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![RADER_N])
            .with_precision(precision)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        direct_rader_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_direct_rader_step());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 1);
    ir
}

fn direct_rader_multi_kernel_spectrum() -> Vec<Complex64> {
    let mut spectrum = Vec::with_capacity(MULTI_KERNEL_COUNT * RADER_N);
    for kernel_id in 0..MULTI_KERNEL_COUNT {
        spectrum.extend((0..RADER_N).map(|index| {
            let x = index as f64;
            Complex64::new(
                0.76 + 0.06 * kernel_id as f64 + 0.025 * (0.07 * x).cos(),
                0.14 + 0.017 * kernel_id as f64 + 0.012 * (0.11 * x).sin(),
            )
        }));
    }
    spectrum
}

fn build_direct_rader_multi_kernel(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![RADER_N])
            .with_precision(precision)
            .with_convolution(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        direct_rader_multi_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_direct_rader_multi_kernel_step());
    assert_eq!(ir.forward_fft.batch_count(), 1);
    assert_eq!(ir.inverse_fft.batch_count(), MULTI_KERNEL_COUNT);
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 1);
    ir
}

const FFT_RADER_N: usize = 257;

fn fft_rader_input64() -> Vec<Complex64> {
    (0..FFT_RADER_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.31 + (0.037 * x).sin() + 0.0009 * x,
                -0.18 + (0.053 * x).cos() - 0.0007 * x,
            )
        })
        .collect()
}

fn fft_rader_kernel_spectrum() -> Vec<Complex64> {
    (0..FFT_RADER_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.91 + 0.035 * (0.029 * x).cos(),
                0.16 + 0.021 * (0.047 * x).sin(),
            )
        })
        .collect()
}

fn build_fft_rader(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![FFT_RADER_N])
            .with_precision(precision)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        fft_rader_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_fft_rader_step());
    assert!(!ir.has_fused_direct_rader_step());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 1);
    ir
}

const BLUESTEIN_N: usize = 103;

fn whole_axis_bluestein_input64() -> Vec<Complex64> {
    (0..BLUESTEIN_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.38 + (0.071 * x).sin() + 0.0017 * x,
                -0.27 + (0.113 * x).cos() - 0.0011 * x,
            )
        })
        .collect()
}

fn whole_axis_bluestein_kernel_spectrum() -> Vec<Complex64> {
    (0..BLUESTEIN_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.86 + 0.04 * (0.061 * x).cos(),
                0.21 + 0.025 * (0.097 * x).sin(),
            )
        })
        .collect()
}

fn build_whole_axis_bluestein(profile: DeviceProfile, precision: Precision) -> ConvolutionIr {
    let mut tuning = PlannerTuning::portable();
    tuning.max_rader_fft_prime = 100;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![BLUESTEIN_N])
            .with_precision(precision)
            .with_tuning(tuning)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        whole_axis_bluestein_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(matches!(ir.forward_fft, OneDimFftIr::Bluestein(_)));
    assert!(matches!(ir.inverse_fft, OneDimFftIr::Bluestein(_)));
    assert!(!ir.has_fused_stockham_step());
    assert!(!ir.has_fused_direct_rader_step());
    assert!(!ir.has_fused_two_upload_stockham());
    assert!(!ir.has_fused_three_upload_stockham());
    assert!(!ir.has_fused_inverse_stockham());
    let program = ProgramIr::convolution(&ir).unwrap();
    assert!(
        program
            .passes
            .iter()
            .any(|pass| pass.name == ir.multiply.name)
    );
    ir
}

const THREE_UPLOAD_N: usize = 8_388_608;
const THREE_UPLOAD_SHIFT: usize = 37;
const THREE_UPLOAD_MULTI_SHIFTS: [usize; MULTI_KERNEL_COUNT] = [0, 37, 101];

fn append_three_upload_phase_ramp(spectrum: &mut Vec<Complex64>, shift: usize) {
    let angle = -std::f64::consts::TAU * shift as f64 / THREE_UPLOAD_N as f64;
    let step = Complex64::new(angle.cos(), angle.sin());
    let mut value = Complex64::new(1.0, 0.0);
    for _ in 0..THREE_UPLOAD_N {
        spectrum.push(value);
        value *= step;
    }
}

fn three_upload_kernel_spectrum() -> Vec<Complex64> {
    let mut spectrum = Vec::with_capacity(THREE_UPLOAD_N);
    append_three_upload_phase_ramp(&mut spectrum, THREE_UPLOAD_SHIFT);
    spectrum
}

fn three_upload_multi_kernel_spectrum() -> Vec<Complex64> {
    let mut spectrum = Vec::with_capacity(MULTI_KERNEL_COUNT * THREE_UPLOAD_N);
    for shift in THREE_UPLOAD_MULTI_SHIFTS {
        append_three_upload_phase_ramp(&mut spectrum, shift);
    }
    spectrum
}

fn three_upload_input32() -> Vec<Complex32> {
    (0..THREE_UPLOAD_N)
        .map(|index| {
            let re = ((index % 251) as f32 - 125.0) / 127.0;
            let im = (((index * 17 + 11) % 257) as f32 - 128.0) / 129.0;
            Complex32::new(re, im)
        })
        .collect()
}

fn build_three_upload(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 48 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![THREE_UPLOAD_N])
            .with_precision(Precision::F32)
            .with_convolution(true),
        three_upload_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_three_upload_stockham());
    ir
}

fn build_three_upload_sequence_conjugated(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 48 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![THREE_UPLOAD_N])
            .with_precision(Precision::F32)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence),
        three_upload_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_three_upload_stockham());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 5);
    ir
}

fn build_three_upload_multi_kernel(mut profile: DeviceProfile) -> ConvolutionIr {
    profile.shared_memory_bytes = 48 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![THREE_UPLOAD_N])
            .with_precision(Precision::F32)
            .with_convolution(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT),
        three_upload_multi_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_three_upload_multi_kernel_stockham());
    let step = ir
        .fused_three_upload_multi_kernel_stockham
        .as_ref()
        .unwrap();
    assert_eq!(step.required_shared_memory_bytes, 34 * 1024);
    assert_eq!(step.forward_mapping.outer_batch_count, 1);
    assert_eq!(step.inverse_mapping.outer_batch_count, MULTI_KERNEL_COUNT);
    let program = ProgramIr::convolution(&ir).unwrap();
    assert_eq!(program.passes.len(), 5);
    assert_eq!(program.resources[0].elements, THREE_UPLOAD_N);
    assert_eq!(
        program.resources[1].elements,
        MULTI_KERNEL_COUNT * THREE_UPLOAD_N
    );
    ir
}

fn assert_three_upload_shift(actual: &[Complex32], input: &[Complex32], label: &str) {
    assert_eq!(actual.len(), THREE_UPLOAD_N);
    assert_eq!(input.len(), THREE_UPLOAD_N);
    let error = actual
        .iter()
        .enumerate()
        .map(|(index, actual)| {
            let expected = input[(index + THREE_UPLOAD_N - THREE_UPLOAD_SHIFT) % THREE_UPLOAD_N];
            let dr = actual.re - expected.re;
            let di = actual.im - expected.im;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        error <= 8.0e-3,
        "{label} N8388608 shift convolution error {error:e}"
    );
}

fn assert_three_upload_conjugated_shift(actual: &[Complex32], input: &[Complex32], label: &str) {
    assert_eq!(actual.len(), THREE_UPLOAD_N);
    assert_eq!(input.len(), THREE_UPLOAD_N);
    let error = actual
        .iter()
        .enumerate()
        .map(|(index, actual)| {
            // conj(DFT(x))[k] is the DFT of conj(x[-n]); the phase-ramp kernel then
            // applies the same +THREE_UPLOAD_SHIFT output displacement as the base gate.
            let source = (THREE_UPLOAD_SHIFT + THREE_UPLOAD_N - index) % THREE_UPLOAD_N;
            let expected = Complex32::new(input[source].re, -input[source].im);
            let dr = actual.re - expected.re;
            let di = actual.im - expected.im;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        error <= 8.0e-3,
        "{label} N8388608 conjugated shift error {error:e}"
    );
}

fn assert_three_upload_multi_kernel_shifts(actual: &[Complex32], input: &[Complex32], label: &str) {
    assert_eq!(actual.len(), MULTI_KERNEL_COUNT * THREE_UPLOAD_N);
    assert_eq!(input.len(), THREE_UPLOAD_N);
    for (kernel_id, shift) in THREE_UPLOAD_MULTI_SHIFTS.into_iter().enumerate() {
        let output = &actual[kernel_id * THREE_UPLOAD_N..(kernel_id + 1) * THREE_UPLOAD_N];
        let error = output
            .iter()
            .enumerate()
            .map(|(index, actual)| {
                let expected = input[(index + THREE_UPLOAD_N - shift) % THREE_UPLOAD_N];
                let dr = actual.re - expected.re;
                let di = actual.im - expected.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0f32, f32::max);
        assert!(
            error <= 1.0e-2,
            "{label} N8388608 K3 kernel {kernel_id} shift {shift} error {error:e}"
        );
    }
}

#[test]
fn convolution_mixed_storage_owns_typed_caller_copies_and_compute_midpoint() {
    use vkfft_rs::backend::vulkan::VulkanGlslBackend;

    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    for (precision, compute_scalar, external_scalar, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            2.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            5.0e-6,
        ),
    ] {
        let ir = build_mixed(profile, precision);
        assert_eq!(ir.scalar, compute_scalar);
        assert_eq!(ir.external_scalar, external_scalar);
        assert_eq!(ir.forward_fft.external_storage_scalar(), compute_scalar);
        assert_eq!(ir.inverse_fft.external_storage_scalar(), compute_scalar);
        let gather = ir.input_storage_copy.as_ref().unwrap();
        let scatter = ir.output_storage_copy.as_ref().unwrap();
        assert_eq!(
            gather.operation,
            NdFormattedCopyOperation::GatherExternalToDense
        );
        assert_eq!(
            scatter.operation,
            NdFormattedCopyOperation::ScatterDenseToExternal
        );
        assert_eq!(gather.input_storage_scalar, external_scalar);
        assert_eq!(gather.output_storage_scalar, compute_scalar);
        assert_eq!(scatter.input_storage_scalar, compute_scalar);
        assert_eq!(scatter.output_storage_scalar, external_scalar);
        assert!(ir.fused_stockham_step.is_none());
        assert!(ir.fused_direct_rader_step.is_none());
        assert!(ir.fused_fft_rader_step.is_none());
        assert!(ir.fused_two_upload_stockham.is_none());
        assert!(ir.fused_three_upload_stockham.is_none());
        assert!(ir.fused_inverse_stockham.is_none());

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.scalar, compute_scalar);
        assert_eq!(program.resources[0].scalar, external_scalar);
        assert_eq!(program.resources[1].scalar, external_scalar);
        assert!(
            program
                .resources
                .iter()
                .skip(2)
                .all(|resource| resource.scalar == compute_scalar)
        );
        assert_eq!(program.passes.len(), 5);
        assert_eq!(program.passes[0].name, gather.name);
        assert_eq!(program.passes[2].name, ir.multiply.name);
        assert_eq!(program.passes[4].name, scatter.name);

        let shaders = VulkanGlslBackend.lower_convolution(&ir).unwrap();
        assert_eq!(shaders.len(), 5);
        assert!(shaders[0].glsl.contains("typed NdFormattedCopyPassIr"));
        assert!(shaders[2].glsl.contains("typed ConvolutionMultiplyIr"));
        assert!(shaders[4].glsl.contains("typed NdFormattedCopyPassIr"));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let expected = mixed_expected(&ir, precision);
        let caller_input = quantize_mixed_caller(&mixed_input64(), precision);
        let direct =
            quantize_mixed_caller(&circular_convolution(&caller_input, &kernel()), precision);
        assert_close64(
            &expected,
            &direct,
            tolerance,
            &format!("{precision:?} mixed C2C CPU oracle"),
        );
    }
}

#[test]
fn convolution_program_owns_one_kernel_spectrum_and_spirv_multiply() {
    use vkfft_rs::backend::vulkan::VulkanGlslBackend;

    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    for precision in [Precision::F32, Precision::F64] {
        let ir = build(profile, precision);
        let program = ProgramIr::convolution(&ir).unwrap();
        let kernels = program
            .resources
            .iter()
            .filter(|resource| {
                resource.kind == ProgramResourceKind::LookupTable
                    && resource.name.starts_with("convolution_kernel_spectrum_")
            })
            .collect::<Vec<_>>();
        assert_eq!(kernels.len(), 1);
        assert!(matches!(
            &kernels[0].initialization,
            ProgramResourceInitialization::Complex64(values) if values.len() == N
        ));
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == ir.multiply.name)
        );
        let shaders = VulkanGlslBackend.lower_convolution(&ir).unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        match precision {
            Precision::F32 => {
                assert!(ir.has_fused_stockham_step());
                assert!(!ir.has_fused_inverse_stockham());
                assert_eq!(program.passes.len(), 1);
                assert_eq!(
                    program.passes[0].name,
                    ir.fused_stockham_step.as_ref().unwrap().name
                );
                assert!(
                    shaders[0]
                        .glsl
                        .contains("performConvolution Stockham convolutionStep IR")
                );
                assert!(shaders[0].glsl.contains("convolution_forward"));
                assert!(shaders[0].glsl.contains("convolution_inverse"));
                assert!(shaders[0].glsl.contains("vkfft_lut.data"));
            }
            Precision::F64 => {
                assert!(!ir.has_fused_stockham_step());
                assert!(ir.has_fused_inverse_stockham());
                assert_eq!(program.passes.len(), 2);
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("_input_mul_lut")
                );
                let fused_inverse = shaders
                    .last()
                    .expect("missing fused inverse Stockham shader");
                assert!(fused_inverse.glsl.contains("vkfft_lut.data"));
                assert!(!fused_inverse.glsl.contains("typed ConvolutionMultiplyIr"));
            }
            _ => unreachable!(),
        }
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn convolution_direct_rader_fuses_pinned_application_step() {
    use vkfft_rs::backend::vulkan::VulkanGlslBackend;

    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    let n = 47usize;
    let spectrum = (0..n)
        .map(|index| {
            let x = index as f64;
            Complex64::new(0.8 + 0.03 * (0.17 * x).cos(), 0.2 + 0.02 * (0.09 * x).sin())
        })
        .collect::<Vec<_>>();
    for precision in [Precision::F32, Precision::F64] {
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_precision(precision)
                .with_convolution(true)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
            spectrum.clone(),
            profile,
        )
        .unwrap();
        assert!(ir.has_fused_direct_rader_step());
        assert!(!ir.has_fused_stockham_step());
        assert!(!ir.has_fused_inverse_stockham());
        let step = ir.fused_direct_rader_step.as_ref().unwrap();
        assert_eq!(step.prime, n);
        assert_eq!(step.batch_count, 1);
        assert_eq!(
            step.required_shared_memory_bytes,
            n * step.scalar.complex_bytes()
        );

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, step.name);
        let luts = program
            .resources
            .iter()
            .filter(|resource| resource.kind == ProgramResourceKind::LookupTable)
            .collect::<Vec<_>>();
        assert_eq!(luts.len(), 2);
        assert_eq!(luts[0].elements, n);
        assert_eq!(luts[1].elements, n - 1);

        let shaders = VulkanGlslBackend.lower_convolution(&ir).unwrap();
        assert_eq!(shaders.len(), 1);
        assert!(
            shaders[0]
                .glsl
                .contains("Direct-Rader performConvolution convolutionStep")
        );
        assert!(shaders[0].glsl.contains("vkfft_rader_spectrum[47]"));
        assert!(
            shaders[0]
                .glsl
                .contains("vkfft_twiddle_lut.data[twiddle_index]")
        );
        assert!(
            shaders[0]
                .glsl
                .contains("inversesqrt(vkfft_convolution_norm)")
        );
        assert_eq!(shaders[0].compile_spirv().unwrap().words[0], 0x0723_0203);
    }
}

#[test]
fn convolution_fft_rader_fuses_pinned_application_step() {
    use vkfft_rs::backend::vulkan::VulkanGlslBackend;

    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_fft_rader(profile, precision);
        let step = ir.fused_fft_rader_step.as_ref().unwrap();
        assert_eq!(step.prime, FFT_RADER_N);
        assert_eq!(step.convolution_len, FFT_RADER_N - 1);
        assert_eq!(step.batch_count, 1);
        assert_eq!(step.workgroup_size.x, 17);
        assert_eq!(
            step.required_shared_memory_bytes,
            (2 * (FFT_RADER_N - 1) + FFT_RADER_N) * step.scalar.complex_bytes()
        );
        assert_eq!(
            step.requires_twiddle_lut().unwrap(),
            precision == Precision::F64
        );

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, step.name);
        assert_eq!(program.resources[0].elements, FFT_RADER_N);
        assert_eq!(program.resources[1].elements, FFT_RADER_N);
        let luts = program
            .resources
            .iter()
            .filter(|resource| resource.kind == ProgramResourceKind::LookupTable)
            .collect::<Vec<_>>();
        assert_eq!(luts.len(), if precision == Precision::F64 { 4 } else { 3 });
        assert_eq!(luts[0].elements, FFT_RADER_N);
        assert_eq!(luts[1].elements, FFT_RADER_N - 1);
        assert_eq!(luts[2].elements, FFT_RADER_N - 1);

        let shaders = VulkanGlslBackend.lower_convolution(&ir).unwrap();
        assert_eq!(shaders.len(), 1);
        let shader = &shaders[0];
        assert!(
            shader
                .glsl
                .contains("single-container FFT-Rader performConvolution convolutionStep")
        );
        assert!(shader.glsl.contains("vkfft_rader_natural[257]"));
        assert!(shader.glsl.contains("vkfft_rader_forward_lut.data[i]"));
        assert!(shader.glsl.contains("vkfft_rader_inverse_lut.data[i]"));
        assert!(shader.glsl.contains("inversesqrt(vkfft_convolution_norm)"));
        if precision == Precision::F64 {
            assert!(shader.glsl.contains("vkfft_twiddle_lut.data"));
        }
        assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
    }
}

#[test]
fn convolution_fft_rader_batch_two_remains_explicit() {
    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![FFT_RADER_N])
            .with_batch_count(2)
            .with_precision(Precision::F32)
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        fft_rader_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(!ir.has_fused_fft_rader_step());
    let program = ProgramIr::convolution(&ir).unwrap();
    assert!(program.passes.len() > 1);
    assert!(
        program
            .passes
            .iter()
            .any(|pass| pass.name == ir.multiply.name)
    );
}

#[test]
fn convolution_whole_axis_bluestein_keeps_explicit_application_multiply() {
    use vkfft_rs::backend::vulkan::VulkanGlslBackend;

    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_whole_axis_bluestein(profile, precision);
        let program = ProgramIr::convolution(&ir).unwrap();
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name == ir.multiply.name)
        );
        let shaders = VulkanGlslBackend.lower_convolution(&ir).unwrap();
        let explicit_multiply_count = shaders
            .iter()
            .filter(|shader| shader.glsl.contains("typed ConvolutionMultiplyIr"))
            .count();
        assert_eq!(explicit_multiply_count, 1);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn convolution_boundary_contracts_keep_multidimensional_surface_narrow() {
    use vkfft_rs::{TransformKind, VkFftError};

    let nd = FftConfig::new(vec![4, 4])
        .with_batch_count(2)
        .with_convolution(true);
    assert!(nd.validate().is_ok());

    let unsupported = [
        FftConfig::new(vec![N])
            .with_transform(TransformKind::RealToComplex)
            .with_convolution(true),
        FftConfig::new(vec![N])
            .with_input_buffer_batch_stride(N + 3)
            .with_convolution(true),
        FftConfig::new(vec![N])
            .with_zero_padding(0, 2, 5)
            .unwrap()
            .with_convolution(true),
    ];
    for config in unsupported {
        assert!(matches!(
            config.validate().unwrap_err(),
            VkFftError::UnsupportedKernelPath(_)
        ));
    }
    for precision in [
        Precision::F16StorageF32Compute,
        Precision::F64ComputeF32Storage,
    ] {
        assert!(
            FftConfig::new(vec![N])
                .with_precision(precision)
                .with_convolution(true)
                .validate()
                .is_ok(),
            "{precision:?} mixed C2C K1 should be admitted"
        );
    }

    // Kernel spectrum is an immutable frequency-space contract in P0, not an
    // arbitrary-length auxiliary buffer.
    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::Vulkan, vkfft_rs::GpuVendor::Nvidia);
    profile.supports_f64 = true;
    let error = ConvolutionIr::build_from_spectrum(
        config(Precision::F32),
        vec![Complex64::default(); N - 1],
        profile,
    )
    .unwrap_err();
    assert_eq!(
        error,
        VkFftError::InvalidKernelIr(
            "convolution kernel spectrum length must match its scalar/independent/matrix plane layout"
        )
    );
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "level-zero-runtime")]
fn build_portable_scalar_length(
    profile: DeviceProfile,
    length: usize,
    precision: Precision,
) -> ConvolutionIr {
    let kernel = (0..length)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.071 * x).cos() + 0.002 * x, (0.053 * x).sin() - 0.001 * x)
        })
        .collect::<Vec<_>>();
    let kernel_spectrum = vkfft_rs::reference::dft(&kernel, Direction::Forward, false);
    let mut config = FftConfig::new(vec![length])
        .with_convolution(true)
        .with_batch_count(1);
    config.precision = precision;
    ConvolutionIr::build_from_spectrum(config, kernel_spectrum, profile).unwrap()
}

#[cfg(feature = "level-zero-runtime")]
fn level_zero_portable_direct_rader_tuning() -> PlannerTuning {
    // Intel's device-default scheduler deliberately disables Direct-Rader above p17.
    // Keep that production policy intact, but make this execution gate explicitly opt into
    // portable tuning and exclude p47 from the FFT-Rader scan so the Direct-Rader kernel path
    // itself remains covered on Level Zero hardware.
    let mut tuning = PlannerTuning::portable();
    tuning.max_rader_fft_prime = RADER_N;
    tuning
}

#[cfg(feature = "level-zero-runtime")]
fn build_level_zero_portable_direct_rader(
    profile: DeviceProfile,
    precision: Precision,
) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![RADER_N])
            .with_precision(precision)
            .with_tuning(level_zero_portable_direct_rader_tuning())
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        direct_rader_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_direct_rader_step());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 1);
    ir
}

#[cfg(feature = "level-zero-runtime")]
fn build_level_zero_portable_direct_rader_multi_kernel(
    profile: DeviceProfile,
    precision: Precision,
) -> ConvolutionIr {
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![RADER_N])
            .with_precision(precision)
            .with_tuning(level_zero_portable_direct_rader_tuning())
            .with_convolution(true)
            .with_convolution_kernel_count(MULTI_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true),
        direct_rader_multi_kernel_spectrum(),
        profile,
    )
    .unwrap();
    assert!(ir.has_fused_direct_rader_multi_kernel_step());
    assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 1);
    ir
}

#[cfg(feature = "level-zero-runtime")]
const LEVEL_ZERO_THREE_UPLOAD_N: usize = 524_288;

#[cfg(feature = "level-zero-runtime")]
fn level_zero_three_upload_input64() -> Vec<Complex64> {
    (0..LEVEL_ZERO_THREE_UPLOAD_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                (0.023 * x).sin() + 0.0002 * x,
                (0.031 * x).cos() - 0.0001 * x,
            )
        })
        .collect()
}

#[cfg(feature = "level-zero-runtime")]
fn level_zero_three_upload_kernel_spectrum() -> Vec<Complex64> {
    (0..LEVEL_ZERO_THREE_UPLOAD_N)
        .map(|index| {
            let x = index as f64;
            Complex64::new(0.81 + 0.04 * (0.017 * x).cos(), 0.06 * (0.029 * x).sin())
        })
        .collect()
}

#[cfg(feature = "level-zero-runtime")]
fn build_level_zero_portable_three_upload(profile: DeviceProfile) -> ConvolutionIr {
    // Level Zero F32 uses the fixed-upstream 524288 three-stage Four-step threshold.
    // At this boundary performConvolution forces registerBoost=1 and therefore provides
    // a genuine device-policy three-upload witness without synthetic shared-memory limits.
    let ir = ConvolutionIr::build_from_spectrum(
        FftConfig::new(vec![LEVEL_ZERO_THREE_UPLOAD_N])
            .with_precision(Precision::F32)
            .with_convolution(true),
        level_zero_three_upload_kernel_spectrum(),
        profile,
    )
    .unwrap();
    let OneDimFftIr::Recursive(forward) = &ir.forward_fft else {
        panic!("Level Zero N524288 three-upload witness must retain recursive FFT");
    };
    let schedule = forward
        .stockham_upload_schedule
        .as_ref()
        .expect("Level Zero N524288 three-upload witness lost upload metadata");
    assert_eq!(schedule.upload_count, 3);
    assert_eq!(schedule.axis_split, vec![4096, 16, 8]);
    assert!(ir.has_fused_three_upload_stockham());
    let OneDimFftIr::Recursive(inverse) = &ir.inverse_fft else {
        panic!("Level Zero N524288 inverse witness must retain recursive FFT");
    };
    let forward_uploads = forward
        .four_step_stockham_upload_kernels()
        .unwrap()
        .expect("Level Zero N524288 forward three-upload kernels");
    let inverse_uploads = inverse
        .four_step_stockham_upload_kernels()
        .unwrap()
        .expect("Level Zero N524288 inverse three-upload kernels");
    let [_, _, forward_low] = forward_uploads.as_slice() else {
        panic!("Level Zero N524288 forward witness must have exactly three uploads");
    };
    let [_, _, inverse_low] = inverse_uploads.as_slice() else {
        panic!("Level Zero N524288 inverse witness must have exactly three uploads");
    };
    for low in [forward_low, inverse_low] {
        assert_eq!(
            low.execution_layout,
            vkfft_rs::StockhamExecutionLayout::SharedPingPong
        );
        assert_eq!(low.shared_memory.buffers, 2);
        assert_eq!(low.required_shared_memory_bytes().unwrap(), 64 * 1024);
        assert!(low.register_stockham_stages().unwrap().is_none());
        assert_eq!(low.twiddle_lut_len(), Some(LEVEL_ZERO_THREE_UPLOAD_N));
    }
    let fused = ir
        .fused_three_upload_stockham
        .as_ref()
        .expect("Level Zero N524288 must fuse the ping-pong three-upload application step");
    assert_eq!(
        fused.special_twiddle_lut_len().unwrap(),
        Some(LEVEL_ZERO_THREE_UPLOAD_N)
    );
    let program = ProgramIr::convolution(&ir).unwrap();
    assert_eq!(program.passes.len(), 5);
    program.validate().unwrap();
    let root_luts = program
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.initialization,
                ProgramResourceInitialization::StockhamUnitRoots { len }
                    if len == LEVEL_ZERO_THREE_UPLOAD_N
            )
        })
        .count();
    assert_eq!(root_luts, 1);
    let special = program
        .passes
        .iter()
        .find(|pass| pass.name == fused.name)
        .expect("Level Zero N524288 program lost fused three-upload pass");
    assert!(
        special.bindings.iter().any(|binding| binding.binding == 3
            && binding.role == vkfft_rs::BufferRole::TwiddleLookupTable)
    );
    let shader = vkfft_rs::backend::vulkan::VulkanGlslBackend
        .lower_convolution_three_upload_stockham_step(fused)
        .unwrap();
    assert!(
        shader
            .glsl
            .contains("three-upload ping-pong performConvolution")
    );
    assert!(shader.glsl.contains("VKFFT_TWIDDLE_LUT_N = 524288u"));
    assert_eq!(shader.required_shared_memory_bytes, 64 * 1024);
    assert!(
        shader
            .glsl
            .contains("uint vkfft_k2 = vkfft_batch % VKFFT_B;")
    );
    assert!(
        shader
            .glsl
            .contains("uint vkfft_four_step_group = vkfft_batch / VKFFT_B;")
    );
    assert!(
        shader
            .glsl
            .contains("uint vkfft_k3 = vkfft_four_step_group % VKFFT_C;")
    );
    shader.compile_spirv().unwrap();
    let native = vkfft_rs::backend::NativeSourceBackend::new(vkfft_rs::Backend::LevelZero)
        .lower_convolution(&ir)
        .unwrap();
    native.validate().unwrap();
    assert_eq!(native.shaders.len(), 5);
    let special_native = native
        .program
        .passes
        .iter()
        .zip(&native.shaders)
        .find(|(pass, _)| pass.name == fused.name)
        .map(|(_, shader)| shader)
        .expect("Level Zero native source lost fused three-upload pass");
    assert!(special_native.source.contains("VKFFT_TWIDDLE_LUT_N"));
    assert_eq!(
        special_native
            .source
            .matches("VKFFT_TWIDDLE_LUT_N =")
            .count(),
        1,
        "Level Zero native special pass must define its twiddle-LUT period exactly once"
    );
    assert!(special_native.bindings.iter().any(|binding| {
        binding.binding == 3 && binding.role == vkfft_rs::BufferRole::TwiddleLookupTable
    }));
    ir
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_portable_algorithm_witnesses_are_structurally_stable() {
    let mut profile =
        DeviceProfile::generic(vkfft_rs::Backend::LevelZero, vkfft_rs::GpuVendor::Intel);
    profile.shared_memory_bytes = 64 * 1024;
    profile.shared_memory_pow2_bytes = 64 * 1024;
    profile.max_threads_per_block = 256;
    profile.max_workgroup_size = [256, 256, 64];
    profile.coalesced_memory_bytes = 64;
    assert!(
        build_level_zero_portable_direct_rader(profile, Precision::F32)
            .has_fused_direct_rader_step()
    );
    assert!(
        build_level_zero_portable_direct_rader_multi_kernel(profile, Precision::F32)
            .has_fused_direct_rader_multi_kernel_step()
    );
    assert!(build_level_zero_portable_three_upload(profile).has_fused_three_upload_stockham());
}

#[cfg(feature = "level-zero-runtime")]
fn run_level_zero_portable_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    let profile = runtime.device_profile();

    let scalar_input64 = input64();
    let scalar_input32 = scalar_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let scalar = build(profile, Precision::F32);
    let expected = execute_convolution_ir(&scalar, &scalar_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&scalar, &scalar_input32)
        .unwrap();
    assert_close32(&actual, &expected, 3.0e-4, runtime.device_name());

    let cross_power = build_conjugated_cross_power_small(profile);
    let expected = execute_convolution_ir(&cross_power, &scalar_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&cross_power, &scalar_input32)
        .unwrap();
    assert_close32(&actual, &expected, 2.0e-3, runtime.device_name());

    for (matrix_count, symmetric, semantic_policy) in [(2usize, true, true), (3usize, false, false)]
    {
        let matrix_input64 = matrix_input64(matrix_count);
        let matrix_input32 = matrix_input64
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect::<Vec<_>>();
        let matrix = build_matrix(
            profile,
            Precision::F32,
            matrix_count,
            symmetric,
            semantic_policy,
        );
        let expected = execute_convolution_ir(&matrix, &matrix_input64).unwrap();
        let actual = runtime
            .execute_convolution_f32(&matrix, &matrix_input32)
            .unwrap();
        assert_close32(&actual, &expected, 3.0e-3, runtime.device_name());
    }

    let multi_input64 = multi_kernel_input64();
    let multi_input32 = multi_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let multi_kernel = build_multi_kernel(profile, Precision::F32);
    let expected = execute_convolution_ir(&multi_kernel, &multi_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&multi_kernel, &multi_input32)
        .unwrap();
    assert_close32(&actual, &expected, 3.0e-3, runtime.device_name());

    let matrix_multi_input64 = matrix_multi_kernel_input64();
    let matrix_multi_input32 = matrix_multi_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix_multi_kernel = build_matrix_multi_kernel(profile, Precision::F32);
    let expected = execute_convolution_ir(&matrix_multi_kernel, &matrix_multi_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&matrix_multi_kernel, &matrix_multi_input32)
        .unwrap();
    assert_close32(&actual, &expected, 3.0e-3, runtime.device_name());

    let direct_input64 = direct_rader_input64();
    let direct_input32 = direct_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let direct = build_level_zero_portable_direct_rader(profile, Precision::F32);
    let expected = execute_convolution_ir(&direct, &direct_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&direct, &direct_input32)
        .unwrap();
    assert_close32(&actual, &expected, 4.0e-3, runtime.device_name());

    let direct_multi = build_level_zero_portable_direct_rader_multi_kernel(profile, Precision::F32);
    let expected = execute_convolution_ir(&direct_multi, &direct_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&direct_multi, &direct_input32)
        .unwrap();
    assert_close32(&actual, &expected, 5.0e-3, runtime.device_name());

    for length in [103usize, 257usize] {
        let ir = build_portable_scalar_length(profile, length, Precision::F32);
        let input = (0..length)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (0.037 * x).sin() + 0.0005 * x,
                    (0.029 * x).cos() - 0.0003 * x,
                )
            })
            .collect::<Vec<_>>();
        let input64 = input
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect::<Vec<_>>();
        let expected = execute_convolution_ir(&ir, &input64).unwrap();
        let actual = runtime.execute_convolution_f32(&ir, &input).unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
            })
            .fold(0.0f64, f64::max);
        let reference_scale = expected
            .iter()
            .map(|value| value.norm_sqr().sqrt())
            .fold(1.0f64, f64::max);
        assert!(
            max_error < 2.0e-3 * reference_scale,
            "portable Level Zero convolution N{length} error {max_error:e}, reference scale {reference_scale:e}"
        );
    }

    let large_ir = build_large_two_upload(profile);
    let large_input64 = large_input64();
    let large_input32 = large_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let large_actual = runtime
        .execute_convolution_f32(&large_ir, &large_input32)
        .unwrap();
    assert_large_two_upload_result(
        &large_actual,
        &large_ir,
        &large_input64,
        runtime.device_name(),
    );

    let smooth_ir = build_smooth_two_upload(profile);
    let smooth_input64 = smooth_two_upload_input64();
    let smooth_input32 = smooth_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let smooth_actual = runtime
        .execute_convolution_f32(&smooth_ir, &smooth_input32)
        .unwrap();
    assert_smooth_two_upload_result(
        &smooth_actual,
        &smooth_ir,
        &smooth_input64,
        runtime.device_name(),
    );

    let three_ir = build_level_zero_portable_three_upload(profile);
    let three_input64 = level_zero_three_upload_input64();
    let three_input32 = three_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let expected = execute_convolution_ir(&three_ir, &three_input64).unwrap();
    let actual = runtime
        .execute_convolution_f32(&three_ir, &three_input32)
        .unwrap();
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| {
            (f64::from(actual.re) - expected.re).hypot(f64::from(actual.im) - expected.im)
        })
        .fold(0.0f64, f64::max);
    let reference_scale = expected
        .iter()
        .map(|value| value.norm_sqr().sqrt())
        .fold(1.0f64, f64::max);
    assert!(
        max_error < 3.0e-3 * reference_scale,
        "Level Zero N524288 three-upload fallback error {max_error:e}, reference scale {reference_scale:e} on {}",
        runtime.device_name()
    );
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    let expected = expected64();
    let input64 = input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let f32_ir = build(runtime.device_profile(), Precision::F32);
    let actual32 = runtime.execute_convolution_f32(&f32_ir, &input32).unwrap();
    assert_close32(&actual32, &expected, 3.0e-4, runtime.device_name());

    let semantic_ir = build_conjugated_cross_power_small(runtime.device_profile());
    let semantic_expected = execute_convolution_ir(&semantic_ir, &input64).unwrap();
    let semantic_actual = runtime
        .execute_convolution_f32(&semantic_ir, &input32)
        .unwrap();
    assert_close32(
        &semantic_actual,
        &semantic_expected,
        2.0e-3,
        runtime.device_name(),
    );

    let matrix2_input64 = matrix_input64(2);
    let matrix2_input32 = matrix2_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix2_ir = build_matrix(runtime.device_profile(), Precision::F32, 2, true, true);
    let matrix2_expected = execute_convolution_ir(&matrix2_ir, &matrix2_input64).unwrap();
    let matrix2_actual = runtime
        .execute_convolution_f32(&matrix2_ir, &matrix2_input32)
        .unwrap();
    assert_close32(
        &matrix2_actual,
        &matrix2_expected,
        3.0e-3,
        runtime.device_name(),
    );

    let matrix3_input64 = matrix_input64(3);
    let matrix3_input32 = matrix3_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix3_ir = build_matrix(runtime.device_profile(), Precision::F32, 3, false, false);
    let matrix3_expected = execute_convolution_ir(&matrix3_ir, &matrix3_input64).unwrap();
    let matrix3_actual = runtime
        .execute_convolution_f32(&matrix3_ir, &matrix3_input32)
        .unwrap();
    assert_close32(
        &matrix3_actual,
        &matrix3_expected,
        3.0e-3,
        runtime.device_name(),
    );

    let multi_input64 = multi_kernel_input64();
    let multi_input32 = multi_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let multi_ir = build_multi_kernel(runtime.device_profile(), Precision::F32);
    let multi_expected = execute_convolution_ir(&multi_ir, &multi_input64).unwrap();
    let multi_actual = runtime
        .execute_convolution_f32(&multi_ir, &multi_input32)
        .unwrap();
    assert_close32(
        &multi_actual,
        &multi_expected,
        3.0e-3,
        runtime.device_name(),
    );

    let matrix_multi_input64 = matrix_multi_kernel_input64();
    let matrix_multi_input32 = matrix_multi_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix_multi_ir = build_matrix_multi_kernel(runtime.device_profile(), Precision::F32);
    let matrix_multi_expected =
        execute_convolution_ir(&matrix_multi_ir, &matrix_multi_input64).unwrap();
    let matrix_multi_actual = runtime
        .execute_convolution_f32(&matrix_multi_ir, &matrix_multi_input32)
        .unwrap();
    assert_close32(
        &matrix_multi_actual,
        &matrix_multi_expected,
        3.0e-3,
        runtime.device_name(),
    );

    let rader_input64 = direct_rader_input64();
    let rader_input32 = rader_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let rader_ir = build_direct_rader(runtime.device_profile(), Precision::F32);
    let rader_expected = execute_convolution_ir(&rader_ir, &rader_input64).unwrap();
    let rader_actual = runtime
        .execute_convolution_f32(&rader_ir, &rader_input32)
        .unwrap();
    assert_close32(
        &rader_actual,
        &rader_expected,
        4.0e-3,
        runtime.device_name(),
    );

    let rader_multi_ir = build_direct_rader_multi_kernel(runtime.device_profile(), Precision::F32);
    let rader_multi_expected = execute_convolution_ir(&rader_multi_ir, &rader_input64).unwrap();
    let rader_multi_actual = runtime
        .execute_convolution_f32(&rader_multi_ir, &rader_input32)
        .unwrap();
    assert_close32(
        &rader_multi_actual,
        &rader_multi_expected,
        5.0e-3,
        runtime.device_name(),
    );

    let fft_rader_input64 = fft_rader_input64();
    let fft_rader_input32 = fft_rader_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let fft_rader_ir = build_fft_rader(runtime.device_profile(), Precision::F32);
    let fft_rader_expected = execute_convolution_ir(&fft_rader_ir, &fft_rader_input64).unwrap();
    let fft_rader_actual = runtime
        .execute_convolution_f32(&fft_rader_ir, &fft_rader_input32)
        .unwrap();
    assert_close32(
        &fft_rader_actual,
        &fft_rader_expected,
        8.0e-3,
        runtime.device_name(),
    );

    let bluestein_input64 = whole_axis_bluestein_input64();
    let bluestein_input32 = bluestein_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let bluestein_ir = build_whole_axis_bluestein(runtime.device_profile(), Precision::F32);
    let bluestein_expected = execute_convolution_ir(&bluestein_ir, &bluestein_input64).unwrap();
    let bluestein_actual = runtime
        .execute_convolution_f32(&bluestein_ir, &bluestein_input32)
        .unwrap();
    assert_close32(
        &bluestein_actual,
        &bluestein_expected,
        5.0e-3,
        runtime.device_name(),
    );

    let large_ir = build_large_two_upload(runtime.device_profile());
    let large_input64 = large_input64();
    let large_input32 = large_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let large_actual = runtime
        .execute_convolution_f32(&large_ir, &large_input32)
        .unwrap();
    assert_large_two_upload_result(
        &large_actual,
        &large_ir,
        &large_input64,
        runtime.device_name(),
    );

    let semantic_two_ir = build_conjugated_cross_power_two_upload(runtime.device_profile());
    let semantic_two_expected = execute_convolution_ir(&semantic_two_ir, &large_input64).unwrap();
    let semantic_two_actual = runtime
        .execute_convolution_f32(&semantic_two_ir, &large_input32)
        .unwrap();
    assert_close32(
        &semantic_two_actual,
        &semantic_two_expected,
        8.0e-3,
        runtime.device_name(),
    );

    let large_multi_ir = build_large_multi_kernel_two_upload(runtime.device_profile());
    let large_multi_expected = execute_convolution_ir(&large_multi_ir, &large_input64).unwrap();
    let large_multi_actual = runtime
        .execute_convolution_f32(&large_multi_ir, &large_input32)
        .unwrap();
    assert_close32(
        &large_multi_actual,
        &large_multi_expected,
        1.0e-2,
        runtime.device_name(),
    );

    let smooth_ir = build_smooth_two_upload(runtime.device_profile());
    let smooth_input64 = smooth_two_upload_input64();
    let smooth_input32 = smooth_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let smooth_actual = runtime
        .execute_convolution_f32(&smooth_ir, &smooth_input32)
        .unwrap();
    assert_smooth_two_upload_result(
        &smooth_actual,
        &smooth_ir,
        &smooth_input64,
        runtime.device_name(),
    );

    let three_upload_ir = build_three_upload(runtime.device_profile());
    let three_upload_input = three_upload_input32();
    let three_upload_actual = runtime
        .execute_convolution_f32(&three_upload_ir, &three_upload_input)
        .unwrap();
    assert_three_upload_shift(
        &three_upload_actual,
        &three_upload_input,
        runtime.device_name(),
    );

    let conjugated_three_ir = build_three_upload_sequence_conjugated(runtime.device_profile());
    let conjugated_three_actual = runtime
        .execute_convolution_f32(&conjugated_three_ir, &three_upload_input)
        .unwrap();
    assert_three_upload_conjugated_shift(
        &conjugated_three_actual,
        &three_upload_input,
        runtime.device_name(),
    );
    drop(three_upload_actual);
    drop(three_upload_ir);
    drop(conjugated_three_actual);
    drop(conjugated_three_ir);

    let three_upload_multi_ir = build_three_upload_multi_kernel(runtime.device_profile());
    let three_upload_multi_actual = runtime
        .execute_convolution_f32(&three_upload_multi_ir, &three_upload_input)
        .unwrap();
    assert_three_upload_multi_kernel_shifts(
        &three_upload_multi_actual,
        &three_upload_input,
        runtime.device_name(),
    );
    drop(three_upload_multi_actual);
    drop(three_upload_multi_ir);

    if runtime.device_profile().supports_f64 {
        let f64_ir = build(runtime.device_profile(), Precision::F64);
        let actual64 = runtime.execute_convolution_f64(&f64_ir, &input64).unwrap();
        assert_close64(&actual64, &expected, 3.0e-10, runtime.device_name());

        let f64_two_upload_ir = build_f64_two_upload(runtime.device_profile());
        let f64_two_upload_input = f64_two_upload_input64();
        let f64_two_upload_actual = runtime
            .execute_convolution_f64(&f64_two_upload_ir, &f64_two_upload_input)
            .unwrap();
        assert_f64_two_upload_result(
            &f64_two_upload_actual,
            &f64_two_upload_ir,
            &f64_two_upload_input,
            runtime.device_name(),
        );

        let f64_multi_ir = build_f64_multi_kernel_two_upload(runtime.device_profile());
        let f64_multi_expected =
            execute_convolution_ir(&f64_multi_ir, &f64_two_upload_input).unwrap();
        let f64_multi_actual = runtime
            .execute_convolution_f64(&f64_multi_ir, &f64_two_upload_input)
            .unwrap();
        assert_close64(
            &f64_multi_actual,
            &f64_multi_expected,
            4.0e-8,
            runtime.device_name(),
        );

        let semantic_f64_ir = build_f64_conjugated_cross_power_small(runtime.device_profile());
        let semantic_f64_expected = execute_convolution_ir(&semantic_f64_ir, &input64).unwrap();
        let semantic_f64_actual = runtime
            .execute_convolution_f64(&semantic_f64_ir, &input64)
            .unwrap();
        assert_close64(
            &semantic_f64_actual,
            &semantic_f64_expected,
            3.0e-9,
            runtime.device_name(),
        );

        let matrix_f64_ir = build_matrix(runtime.device_profile(), Precision::F64, 2, true, true);
        let matrix_f64_expected = execute_convolution_ir(&matrix_f64_ir, &matrix2_input64).unwrap();
        let matrix_f64_actual = runtime
            .execute_convolution_f64(&matrix_f64_ir, &matrix2_input64)
            .unwrap();
        assert_close64(
            &matrix_f64_actual,
            &matrix_f64_expected,
            4.0e-9,
            runtime.device_name(),
        );

        let multi_f64_ir = build_multi_kernel(runtime.device_profile(), Precision::F64);
        let multi_f64_expected = execute_convolution_ir(&multi_f64_ir, &multi_input64).unwrap();
        let multi_f64_actual = runtime
            .execute_convolution_f64(&multi_f64_ir, &multi_input64)
            .unwrap();
        assert_close64(
            &multi_f64_actual,
            &multi_f64_expected,
            4.0e-9,
            runtime.device_name(),
        );

        let matrix_multi_f64_ir =
            build_matrix_multi_kernel(runtime.device_profile(), Precision::F64);
        let matrix_multi_f64_expected =
            execute_convolution_ir(&matrix_multi_f64_ir, &matrix_multi_input64).unwrap();
        let matrix_multi_f64_actual = runtime
            .execute_convolution_f64(&matrix_multi_f64_ir, &matrix_multi_input64)
            .unwrap();
        assert_close64(
            &matrix_multi_f64_actual,
            &matrix_multi_f64_expected,
            4.0e-9,
            runtime.device_name(),
        );

        let rader_f64_ir = build_direct_rader(runtime.device_profile(), Precision::F64);
        let rader_f64_expected = execute_convolution_ir(&rader_f64_ir, &rader_input64).unwrap();
        let rader_f64_actual = runtime
            .execute_convolution_f64(&rader_f64_ir, &rader_input64)
            .unwrap();
        assert_close64(
            &rader_f64_actual,
            &rader_f64_expected,
            5.0e-9,
            runtime.device_name(),
        );

        let rader_multi_f64_ir =
            build_direct_rader_multi_kernel(runtime.device_profile(), Precision::F64);
        let rader_multi_f64_expected =
            execute_convolution_ir(&rader_multi_f64_ir, &rader_input64).unwrap();
        let rader_multi_f64_actual = runtime
            .execute_convolution_f64(&rader_multi_f64_ir, &rader_input64)
            .unwrap();
        assert_close64(
            &rader_multi_f64_actual,
            &rader_multi_f64_expected,
            8.0e-9,
            runtime.device_name(),
        );

        let fft_rader_f64_ir = build_fft_rader(runtime.device_profile(), Precision::F64);
        let fft_rader_f64_expected =
            execute_convolution_ir(&fft_rader_f64_ir, &fft_rader_input64).unwrap();
        let fft_rader_f64_actual = runtime
            .execute_convolution_f64(&fft_rader_f64_ir, &fft_rader_input64)
            .unwrap();
        assert_close64(
            &fft_rader_f64_actual,
            &fft_rader_f64_expected,
            2.0e-8,
            runtime.device_name(),
        );

        let bluestein_f64_ir = build_whole_axis_bluestein(runtime.device_profile(), Precision::F64);
        let bluestein_f64_expected =
            execute_convolution_ir(&bluestein_f64_ir, &bluestein_input64).unwrap();
        let bluestein_f64_actual = runtime
            .execute_convolution_f64(&bluestein_f64_ir, &bluestein_input64)
            .unwrap();
        assert_close64(
            &bluestein_f64_actual,
            &bluestein_f64_expected,
            1.0e-8,
            runtime.device_name(),
        );

        let semantic_f64_two_ir =
            build_f64_conjugated_cross_power_two_upload(runtime.device_profile());
        let semantic_f64_two_expected =
            execute_convolution_ir(&semantic_f64_two_ir, &f64_two_upload_input).unwrap();
        let semantic_f64_two_actual = runtime
            .execute_convolution_f64(&semantic_f64_two_ir, &f64_two_upload_input)
            .unwrap();
        assert_close64(
            &semantic_f64_two_actual,
            &semantic_f64_two_expected,
            5.0e-8,
            runtime.device_name(),
        );
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn run_mixed_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    let input64 = mixed_input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let f16_ir = build_mixed(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_expected = mixed_expected(&f16_ir, Precision::F16StorageF32Compute);
    let f16_actual = runtime.execute_convolution_f32(&f16_ir, &input32).unwrap();
    assert_close32(
        &f16_actual,
        &f16_expected,
        2.5e-2,
        &format!("{} F16-storage/F32-compute", runtime.device_name()),
    );

    if runtime.device_profile().supports_f64 {
        let f64_ir = build_mixed(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f64_expected = mixed_expected(&f64_ir, Precision::F64ComputeF32Storage);
        let f64_actual = runtime.execute_convolution_f64(&f64_ir, &input64).unwrap();
        assert_close64(
            &f64_actual,
            &f64_expected,
            5.0e-6,
            &format!("{} F64-compute/F32-storage", runtime.device_name()),
        );
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_perform_convolution_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_mixed_storage_c2c_convolution_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_mixed_native(&runtime);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_perform_convolution_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_mixed_storage_c2c_convolution_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_mixed_native(&runtime);
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_perform_convolution_or_skips() {
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero convolution gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let runtime = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    run_level_zero_portable_native(&runtime);
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_mixed_storage_c2c_convolution_or_skips() {
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero mixed-storage convolution gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let runtime = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    run_mixed_native(&runtime);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_mixed_storage_c2c_convolution_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let input64 = mixed_input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let f16_ir = build_mixed(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_expected = mixed_expected(&f16_ir, Precision::F16StorageF32Compute);
    let f16_actual = runtime.execute_convolution_f32(&f16_ir, &input32).unwrap();
    assert_close32(
        &f16_actual,
        &f16_expected,
        2.5e-2,
        "Vulkan F16-storage/F32-compute",
    );

    if runtime.device_profile().supports_f64 {
        let f64_ir = build_mixed(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f64_expected = mixed_expected(&f64_ir, Precision::F64ComputeF32Storage);
        let f64_actual = runtime.execute_convolution_f64(&f64_ir, &input64).unwrap();
        assert_close64(
            &f64_actual,
            &f64_expected,
            5.0e-6,
            "Vulkan F64-compute/F32-storage",
        );
    }
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_multi_kernel_direct_rader_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let input64 = direct_rader_input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let ir = build_direct_rader_multi_kernel(runtime.device_profile(), Precision::F32);
    let expected = execute_convolution_ir(&ir, &input64).unwrap();
    let actual = runtime.execute_convolution_f32(&ir, &input32).unwrap();
    assert_close32(
        &actual,
        &expected,
        5.0e-3,
        "Vulkan F32 Direct-Rader K3 conjugated cross-power",
    );

    if runtime.device_profile().supports_f64 {
        let ir = build_direct_rader_multi_kernel(runtime.device_profile(), Precision::F64);
        let expected = execute_convolution_ir(&ir, &input64).unwrap();
        let actual = runtime.execute_convolution_f64(&ir, &input64).unwrap();
        assert_close64(
            &actual,
            &expected,
            8.0e-9,
            "Vulkan F64 Direct-Rader K3 conjugated cross-power",
        );
    }
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_perform_convolution_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let expected = expected64();
    let input64 = input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let f32_ir = build(runtime.device_profile(), Precision::F32);
    let actual32 = runtime.execute_convolution_f32(&f32_ir, &input32).unwrap();
    assert_close32(&actual32, &expected, 3.0e-4, "Vulkan F32");
    let semantic_ir = build_conjugated_cross_power_small(runtime.device_profile());
    let semantic_expected = execute_convolution_ir(&semantic_ir, &input64).unwrap();
    let semantic_actual = runtime
        .execute_convolution_f32(&semantic_ir, &input32)
        .unwrap();
    assert_close32(
        &semantic_actual,
        &semantic_expected,
        2.0e-3,
        "Vulkan F32 conjugated cross-power",
    );
    let matrix2_input64 = matrix_input64(2);
    let matrix2_input32 = matrix2_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix2_ir = build_matrix(runtime.device_profile(), Precision::F32, 2, true, true);
    let matrix2_expected = execute_convolution_ir(&matrix2_ir, &matrix2_input64).unwrap();
    let matrix2_actual = runtime
        .execute_convolution_f32(&matrix2_ir, &matrix2_input32)
        .unwrap();
    assert_close32(
        &matrix2_actual,
        &matrix2_expected,
        3.0e-3,
        "Vulkan 2x2 symmetric matrix conjugated cross-power",
    );

    let matrix3_input64 = matrix_input64(3);
    let matrix3_input32 = matrix3_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix3_ir = build_matrix(runtime.device_profile(), Precision::F32, 3, false, false);
    let matrix3_expected = execute_convolution_ir(&matrix3_ir, &matrix3_input64).unwrap();
    let matrix3_actual = runtime
        .execute_convolution_f32(&matrix3_ir, &matrix3_input32)
        .unwrap();
    assert_close32(
        &matrix3_actual,
        &matrix3_expected,
        3.0e-3,
        "Vulkan 3x3 matrix convolution",
    );
    let multi_input64 = multi_kernel_input64();
    let multi_input32 = multi_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let multi_ir = build_multi_kernel(runtime.device_profile(), Precision::F32);
    let multi_expected = execute_convolution_ir(&multi_ir, &multi_input64).unwrap();
    let multi_actual = runtime
        .execute_convolution_f32(&multi_ir, &multi_input32)
        .unwrap();
    assert_close32(
        &multi_actual,
        &multi_expected,
        3.0e-3,
        "Vulkan F32 multi-kernel conjugated cross-power",
    );

    let matrix_multi_input64 = matrix_multi_kernel_input64();
    let matrix_multi_input32 = matrix_multi_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix_multi_ir = build_matrix_multi_kernel(runtime.device_profile(), Precision::F32);
    let matrix_multi_expected =
        execute_convolution_ir(&matrix_multi_ir, &matrix_multi_input64).unwrap();
    let matrix_multi_actual = runtime
        .execute_convolution_f32(&matrix_multi_ir, &matrix_multi_input32)
        .unwrap();
    assert_close32(
        &matrix_multi_actual,
        &matrix_multi_expected,
        3.0e-3,
        "Vulkan F32 2x2 symmetric matrix multi-kernel conjugated cross-power",
    );

    let rader_input64 = direct_rader_input64();
    let rader_input32 = rader_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let rader_ir = build_direct_rader(runtime.device_profile(), Precision::F32);
    let rader_expected = execute_convolution_ir(&rader_ir, &rader_input64).unwrap();
    let rader_actual = runtime
        .execute_convolution_f32(&rader_ir, &rader_input32)
        .unwrap();
    assert_close32(
        &rader_actual,
        &rader_expected,
        4.0e-3,
        "Vulkan F32 Direct-Rader conjugated cross-power",
    );

    let fft_rader_input64 = fft_rader_input64();
    let fft_rader_input32 = fft_rader_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let fft_rader_ir = build_fft_rader(runtime.device_profile(), Precision::F32);
    let fft_rader_expected = execute_convolution_ir(&fft_rader_ir, &fft_rader_input64).unwrap();
    let fft_rader_actual = runtime
        .execute_convolution_f32(&fft_rader_ir, &fft_rader_input32)
        .unwrap();
    assert_close32(
        &fft_rader_actual,
        &fft_rader_expected,
        8.0e-3,
        "Vulkan F32 FFT-Rader conjugated cross-power",
    );

    let bluestein_input64 = whole_axis_bluestein_input64();
    let bluestein_input32 = bluestein_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let bluestein_ir = build_whole_axis_bluestein(runtime.device_profile(), Precision::F32);
    let bluestein_expected = execute_convolution_ir(&bluestein_ir, &bluestein_input64).unwrap();
    let bluestein_actual = runtime
        .execute_convolution_f32(&bluestein_ir, &bluestein_input32)
        .unwrap();
    assert_close32(
        &bluestein_actual,
        &bluestein_expected,
        5.0e-3,
        "Vulkan F32 whole-axis Bluestein application multiply",
    );

    let large_ir = build_large_two_upload(runtime.device_profile());
    let large_input64 = large_input64();
    let large_input32 = large_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let large_actual = runtime
        .execute_convolution_f32(&large_ir, &large_input32)
        .unwrap();
    assert_large_two_upload_result(&large_actual, &large_ir, &large_input64, "Vulkan N8192 F32");
    let semantic_two_ir = build_conjugated_cross_power_two_upload(runtime.device_profile());
    let semantic_two_expected = execute_convolution_ir(&semantic_two_ir, &large_input64).unwrap();
    let semantic_two_actual = runtime
        .execute_convolution_f32(&semantic_two_ir, &large_input32)
        .unwrap();
    assert_close32(
        &semantic_two_actual,
        &semantic_two_expected,
        8.0e-3,
        "Vulkan N8192 conjugated cross-power",
    );
    let large_multi_ir = build_large_multi_kernel_two_upload(runtime.device_profile());
    let large_multi_expected = execute_convolution_ir(&large_multi_ir, &large_input64).unwrap();
    let large_multi_actual = runtime
        .execute_convolution_f32(&large_multi_ir, &large_input32)
        .unwrap();
    assert_close32(
        &large_multi_actual,
        &large_multi_expected,
        1.0e-2,
        "Vulkan N8192 K3 two-upload conjugated cross-power",
    );
    let smooth_ir = build_smooth_two_upload(runtime.device_profile());
    let smooth_input64 = smooth_two_upload_input64();
    let smooth_input32 = smooth_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let smooth_actual = runtime
        .execute_convolution_f32(&smooth_ir, &smooth_input32)
        .unwrap();
    assert_smooth_two_upload_result(
        &smooth_actual,
        &smooth_ir,
        &smooth_input64,
        "Vulkan N5760 F32",
    );
    let three_upload_ir = build_three_upload(runtime.device_profile());
    let three_upload_input = three_upload_input32();
    let three_upload_actual = runtime
        .execute_convolution_f32(&three_upload_ir, &three_upload_input)
        .unwrap();
    assert_three_upload_shift(
        &three_upload_actual,
        &three_upload_input,
        "Vulkan N8388608 F32",
    );
    let conjugated_three_ir = build_three_upload_sequence_conjugated(runtime.device_profile());
    let conjugated_three_actual = runtime
        .execute_convolution_f32(&conjugated_three_ir, &three_upload_input)
        .unwrap();
    assert_three_upload_conjugated_shift(
        &conjugated_three_actual,
        &three_upload_input,
        "Vulkan N8388608 sequence-conjugated F32",
    );
    drop(three_upload_actual);
    drop(three_upload_ir);
    drop(conjugated_three_actual);
    drop(conjugated_three_ir);

    let three_upload_multi_ir = build_three_upload_multi_kernel(runtime.device_profile());
    let three_upload_multi_actual = runtime
        .execute_convolution_f32(&three_upload_multi_ir, &three_upload_input)
        .unwrap();
    assert_three_upload_multi_kernel_shifts(
        &three_upload_multi_actual,
        &three_upload_input,
        "Vulkan N8388608 K3 three-upload F32",
    );
    drop(three_upload_multi_actual);
    drop(three_upload_multi_ir);
    if runtime.device_profile().supports_f64 {
        let f64_ir = build(runtime.device_profile(), Precision::F64);
        let actual64 = runtime.execute_convolution_f64(&f64_ir, &input64).unwrap();
        assert_close64(&actual64, &expected, 3.0e-10, "Vulkan F64");

        let f64_two_upload_ir = build_f64_two_upload(runtime.device_profile());
        let f64_two_upload_input = f64_two_upload_input64();
        let f64_two_upload_actual = runtime
            .execute_convolution_f64(&f64_two_upload_ir, &f64_two_upload_input)
            .unwrap();
        assert_f64_two_upload_result(
            &f64_two_upload_actual,
            &f64_two_upload_ir,
            &f64_two_upload_input,
            "Vulkan N4096 F64",
        );

        let f64_multi_ir = build_f64_multi_kernel_two_upload(runtime.device_profile());
        let f64_multi_expected =
            execute_convolution_ir(&f64_multi_ir, &f64_two_upload_input).unwrap();
        let f64_multi_actual = runtime
            .execute_convolution_f64(&f64_multi_ir, &f64_two_upload_input)
            .unwrap();
        assert_close64(
            &f64_multi_actual,
            &f64_multi_expected,
            4.0e-8,
            "Vulkan N4096 K3 two-upload conjugated cross-power",
        );

        let semantic_f64_ir = build_f64_conjugated_cross_power_small(runtime.device_profile());
        let semantic_f64_expected = execute_convolution_ir(&semantic_f64_ir, &input64).unwrap();
        let semantic_f64_actual = runtime
            .execute_convolution_f64(&semantic_f64_ir, &input64)
            .unwrap();
        assert_close64(
            &semantic_f64_actual,
            &semantic_f64_expected,
            3.0e-9,
            "Vulkan F64 conjugated cross-power",
        );

        let matrix_f64_ir = build_matrix(runtime.device_profile(), Precision::F64, 2, true, true);
        let matrix_f64_expected = execute_convolution_ir(&matrix_f64_ir, &matrix2_input64).unwrap();
        let matrix_f64_actual = runtime
            .execute_convolution_f64(&matrix_f64_ir, &matrix2_input64)
            .unwrap();
        assert_close64(
            &matrix_f64_actual,
            &matrix_f64_expected,
            4.0e-9,
            "Vulkan F64 2x2 symmetric matrix conjugated cross-power",
        );

        let multi_f64_ir = build_multi_kernel(runtime.device_profile(), Precision::F64);
        let multi_f64_expected = execute_convolution_ir(&multi_f64_ir, &multi_input64).unwrap();
        let multi_f64_actual = runtime
            .execute_convolution_f64(&multi_f64_ir, &multi_input64)
            .unwrap();
        assert_close64(
            &multi_f64_actual,
            &multi_f64_expected,
            4.0e-9,
            "Vulkan F64 multi-kernel conjugated cross-power",
        );

        let matrix_multi_f64_ir =
            build_matrix_multi_kernel(runtime.device_profile(), Precision::F64);
        let matrix_multi_f64_expected =
            execute_convolution_ir(&matrix_multi_f64_ir, &matrix_multi_input64).unwrap();
        let matrix_multi_f64_actual = runtime
            .execute_convolution_f64(&matrix_multi_f64_ir, &matrix_multi_input64)
            .unwrap();
        assert_close64(
            &matrix_multi_f64_actual,
            &matrix_multi_f64_expected,
            4.0e-9,
            "Vulkan F64 2x2 symmetric matrix multi-kernel conjugated cross-power",
        );

        let rader_f64_ir = build_direct_rader(runtime.device_profile(), Precision::F64);
        let rader_f64_expected = execute_convolution_ir(&rader_f64_ir, &rader_input64).unwrap();
        let rader_f64_actual = runtime
            .execute_convolution_f64(&rader_f64_ir, &rader_input64)
            .unwrap();
        assert_close64(
            &rader_f64_actual,
            &rader_f64_expected,
            5.0e-9,
            "Vulkan F64 Direct-Rader conjugated cross-power",
        );

        let fft_rader_f64_ir = build_fft_rader(runtime.device_profile(), Precision::F64);
        let fft_rader_f64_expected =
            execute_convolution_ir(&fft_rader_f64_ir, &fft_rader_input64).unwrap();
        let fft_rader_f64_actual = runtime
            .execute_convolution_f64(&fft_rader_f64_ir, &fft_rader_input64)
            .unwrap();
        assert_close64(
            &fft_rader_f64_actual,
            &fft_rader_f64_expected,
            2.0e-8,
            "Vulkan F64 FFT-Rader conjugated cross-power",
        );

        let bluestein_f64_ir = build_whole_axis_bluestein(runtime.device_profile(), Precision::F64);
        let bluestein_f64_expected =
            execute_convolution_ir(&bluestein_f64_ir, &bluestein_input64).unwrap();
        let bluestein_f64_actual = runtime
            .execute_convolution_f64(&bluestein_f64_ir, &bluestein_input64)
            .unwrap();
        assert_close64(
            &bluestein_f64_actual,
            &bluestein_f64_expected,
            1.0e-8,
            "Vulkan F64 whole-axis Bluestein application multiply",
        );

        let semantic_f64_two_ir =
            build_f64_conjugated_cross_power_two_upload(runtime.device_profile());
        let semantic_f64_two_expected =
            execute_convolution_ir(&semantic_f64_two_ir, &f64_two_upload_input).unwrap();
        let semantic_f64_two_actual = runtime
            .execute_convolution_f64(&semantic_f64_two_ir, &f64_two_upload_input)
            .unwrap();
        assert_close64(
            &semantic_f64_two_actual,
            &semantic_f64_two_expected,
            5.0e-8,
            "Vulkan N4096 F64 conjugated cross-power",
        );
    }
}
