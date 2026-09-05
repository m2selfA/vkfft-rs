//! Error-free-transform based double-double arithmetic.
//!
//! VkFFT exposes double-double precision modes whose compute value is represented by
//! an unevaluated sum of two IEEE-754 doubles.  GPU lowering is intentionally kept
//! separate from this module; these primitives establish the real arithmetic semantics
//! that future IR/backend work must preserve rather than aliasing the mode to plain F64.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::complex::{Complex, Complex64};
use crate::config::Direction;
use crate::error::{Result, VkFftError};

/// One approximately 106-bit-significand value represented as `hi + lo`.
///
/// Finite values are kept normalized: `hi` carries the rounded leading component and
/// `lo` carries the residual that was lost from the leading F64 operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

pub type ComplexDoubleDouble = Complex<DoubleDouble>;

impl DoubleDouble {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    /// Double-double constants split into the nearest F64 leading word and its residual.
    pub const PI: Self = Self {
        hi: core::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };
    pub const TAU: Self = Self {
        hi: core::f64::consts::TAU,
        lo: 2.449_293_598_294_706_4e-16,
    };
    pub const FRAC_PI_2: Self = Self {
        hi: core::f64::consts::FRAC_PI_2,
        lo: 6.123_233_995_736_766e-17,
    };

    pub const fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Construct and normalize a two-component value.
    pub fn from_parts(hi: f64, lo: f64) -> Self {
        if !hi.is_finite() || !lo.is_finite() {
            return Self {
                hi: hi + lo,
                lo: 0.0,
            };
        }
        let (hi, lo) = two_sum(hi, lo);
        Self { hi, lo }
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    pub fn abs(self) -> Self {
        if self.hi.is_sign_negative() || (self.hi == 0.0 && self.lo.is_sign_negative()) {
            -self
        } else {
            self
        }
    }

    pub fn scale_f64(self, rhs: f64) -> Self {
        self * Self::from_f64(rhs)
    }

    pub fn recip(self) -> Self {
        Self::ONE / self
    }

    /// High-precision sine/cosine for the bounded angles used by FFT unit roots.
    /// The input is reduced to the nearest multiple of pi/2 in double-double
    /// arithmetic, then evaluated on [-pi/4, pi/4] by double-double Taylor terms.
    pub fn sin_cos(self) -> (Self, Self) {
        if !self.is_finite() {
            let value = self.to_f64();
            return (Self::from_f64(value.sin()), Self::from_f64(value.cos()));
        }
        let quadrant = (self.to_f64() / core::f64::consts::FRAC_PI_2).round() as i64;
        let reduced = self - Self::FRAC_PI_2 * Self::from_f64(quadrant as f64);
        let x2 = reduced * reduced;

        let mut sin = reduced;
        let mut sin_term = reduced;
        for n in 1..=18u32 {
            let denominator = ((2 * n) * (2 * n + 1)) as f64;
            sin_term = -(sin_term * x2) / Self::from_f64(denominator);
            sin += sin_term;
        }

        let mut cos = Self::ONE;
        let mut cos_term = Self::ONE;
        for n in 1..=18u32 {
            let denominator = ((2 * n - 1) * (2 * n)) as f64;
            cos_term = -(cos_term * x2) / Self::from_f64(denominator);
            cos += cos_term;
        }

        match quadrant.rem_euclid(4) {
            0 => (sin, cos),
            1 => (cos, -sin),
            2 => (-sin, -cos),
            3 => (-cos, sin),
            _ => unreachable!(),
        }
    }
}

impl From<f64> for DoubleDouble {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<DoubleDouble> for f64 {
    fn from(value: DoubleDouble) -> Self {
        value.to_f64()
    }
}

impl Add for DoubleDouble {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if !self.is_finite() || !rhs.is_finite() {
            return Self::from_f64(self.to_f64() + rhs.to_f64());
        }
        let (s1, mut s2) = two_sum(self.hi, rhs.hi);
        let (t1, t2) = two_sum(self.lo, rhs.lo);
        s2 += t1;
        let (s1, mut s2) = quick_two_sum(s1, s2);
        s2 += t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        Self { hi, lo }
    }
}

impl AddAssign for DoubleDouble {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for DoubleDouble {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for DoubleDouble {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for DoubleDouble {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if !self.is_finite() || !rhs.is_finite() {
            return Self::from_f64(self.to_f64() * rhs.to_f64());
        }
        let (p1, mut p2) = two_prod(self.hi, rhs.hi);
        p2 += self.hi * rhs.lo + self.lo * rhs.hi;
        let (p1, mut p2) = quick_two_sum(p1, p2);
        p2 += self.lo * rhs.lo;
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }
}

impl Div for DoubleDouble {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if !self.is_finite() || !rhs.is_finite() || (rhs.hi == 0.0 && rhs.lo == 0.0) {
            return Self::from_f64(self.to_f64() / rhs.to_f64());
        }

        // Three quotient refinements are enough to recover the double-double
        // residual after the leading F64 quotient.
        let q1 = self.hi / rhs.hi;
        let mut quotient = Self::from_f64(q1);
        let remainder = self - rhs * quotient;
        let q2 = remainder.hi / rhs.hi;
        quotient += Self::from_f64(q2);
        let remainder = self - rhs * quotient;
        let q3 = remainder.hi / rhs.hi;
        quotient + Self::from_f64(q3)
    }
}

impl DivAssign for DoubleDouble {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl MulAssign for DoubleDouble {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for DoubleDouble {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Complex<DoubleDouble> {
    pub fn from_complex64(value: Complex64) -> Self {
        Self::new(
            DoubleDouble::from_f64(value.re),
            DoubleDouble::from_f64(value.im),
        )
    }

    pub fn to_complex64(self) -> Complex64 {
        Complex64::new(self.re.to_f64(), self.im.to_f64())
    }

    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    pub fn scale_f64(self, scale: f64) -> Self {
        Self::new(self.re.scale_f64(scale), self.im.scale_f64(scale))
    }

    pub fn scale_dd(self, scale: DoubleDouble) -> Self {
        Self::new(self.re * scale, self.im * scale)
    }
}

/// Return the direction-specific `exp(+/- i*2*pi*k/N)` unit root in double-double
/// arithmetic. Forward transforms use the conventional negative exponent.
pub fn unit_root(index: usize, len: usize, direction: Direction) -> Result<ComplexDoubleDouble> {
    if len == 0 {
        return Err(VkFftError::ZeroLength { axis: 0 });
    }
    if len > (1usize << 53) || index > (1usize << 53) {
        return Err(VkFftError::ValueOutOfRange {
            field: "double-double unit-root index/length",
        });
    }
    let index = index % len;
    let angle = DoubleDouble::TAU * DoubleDouble::from_f64(index as f64)
        / DoubleDouble::from_f64(len as f64);
    let angle = match direction {
        Direction::Forward => -angle,
        Direction::Inverse => angle,
    };
    let (sin, cos) = angle.sin_cos();
    Ok(ComplexDoubleDouble::new(cos, sin))
}

/// Correctness-oriented double-double DFT oracle. This deliberately favors a direct
/// expression over FFT staging so later executable double-double kernels can be checked
/// against an independent high-precision path.
pub fn dft(
    input: &[ComplexDoubleDouble],
    direction: Direction,
    normalize_inverse: bool,
) -> Result<Vec<ComplexDoubleDouble>> {
    if input.is_empty() {
        return Err(VkFftError::ZeroLength { axis: 0 });
    }
    let len = input.len();
    if len > (1usize << 53) {
        return Err(VkFftError::ValueOutOfRange {
            field: "double-double DFT length",
        });
    }
    let mut output = vec![ComplexDoubleDouble::default(); len];
    for (k, slot) in output.iter_mut().enumerate() {
        let mut sum = ComplexDoubleDouble::default();
        for (n, value) in input.iter().copied().enumerate() {
            let root_index = n.checked_mul(k).ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double DFT root index",
            })? % len;
            sum += value * unit_root(root_index, len, direction)?;
        }
        if direction == Direction::Inverse && normalize_inverse {
            sum = sum.scale_dd(DoubleDouble::ONE / DoubleDouble::from_f64(len as f64));
        }
        *slot = sum;
    }
    Ok(output)
}

/// Knuth's error-free sum: `sum + error == a + b` for finite inputs.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    (sum, a_roundoff + b_roundoff)
}

/// Dekker's fast normalization when the leading magnitude dominates the residual.
#[inline]
pub fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    (sum, b - (sum - a))
}

/// Error-free product using the hardware/compiler FMA contract exposed by Rust.
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, a.mul_add(b, -product))
}

/// Encode the stable native GPU buffer ABI for one double-double complex value:
/// `re.hi, re.lo, im.hi, im.lo`, each in native-endian binary64 representation.
pub fn encode_complex_double_double(values: &[ComplexDoubleDouble]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 32);
    for value in values {
        bytes.extend_from_slice(&value.re.hi.to_ne_bytes());
        bytes.extend_from_slice(&value.re.lo.to_ne_bytes());
        bytes.extend_from_slice(&value.im.hi.to_ne_bytes());
        bytes.extend_from_slice(&value.im.lo.to_ne_bytes());
    }
    bytes
}

/// Decode the stable 32-byte double-double complex buffer ABI.
pub fn decode_complex_double_double(bytes: &[u8]) -> Result<Vec<ComplexDoubleDouble>> {
    if !bytes.len().is_multiple_of(32) {
        return Err(VkFftError::InvalidKernelIr(
            "double-double complex buffer byte length must be a multiple of 32",
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 32);
    for chunk in bytes.chunks_exact(32) {
        let word = |offset: usize| -> f64 {
            f64::from_ne_bytes(
                chunk[offset..offset + 8]
                    .try_into()
                    .expect("8-byte DD word"),
            )
        };
        values.push(ComplexDoubleDouble::new(
            DoubleDouble::from_parts(word(0), word(8)),
            DoubleDouble::from_parts(word(16), word(24)),
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_precision_contract() {
        assert_eq!(core::mem::size_of::<DoubleDouble>(), 16);
        assert_eq!(core::mem::size_of::<ComplexDoubleDouble>(), 32);
    }

    #[test]
    fn complex_buffer_codec_preserves_all_four_binary64_words() {
        let values = vec![
            ComplexDoubleDouble::new(
                DoubleDouble::from_parts(1.0e16, 1.0),
                DoubleDouble::from_parts(-3.0, 2.0e-30),
            ),
            ComplexDoubleDouble::new(
                DoubleDouble::from_parts(core::f64::consts::PI, DoubleDouble::PI.lo),
                DoubleDouble::from_parts(core::f64::consts::E, -7.0e-31),
            ),
        ];
        let bytes = encode_complex_double_double(&values);
        assert_eq!(bytes.len(), values.len() * 32);
        assert_eq!(decode_complex_double_double(&bytes).unwrap(), values);
        assert!(decode_complex_double_double(&bytes[..bytes.len() - 1]).is_err());
        assert_eq!(
            &bytes[0..8],
            &values[0].re.hi.to_ne_bytes(),
            "ABI word 0 must be re.hi"
        );
        assert_eq!(&bytes[8..16], &values[0].re.lo.to_ne_bytes());
        assert_eq!(&bytes[16..24], &values[0].im.hi.to_ne_bytes());
        assert_eq!(&bytes[24..32], &values[0].im.lo.to_ne_bytes());
    }

    #[test]
    fn two_sum_retains_unit_below_large_f64_ulp() {
        let large = DoubleDouble::from_f64(1.0e16);
        let accumulated = large + DoubleDouble::ONE;
        assert_eq!(accumulated.hi, 1.0e16);
        assert_eq!(accumulated.lo, 1.0);
        let recovered = accumulated - large;
        assert_eq!(recovered.hi, 1.0);
        assert_eq!(recovered.lo, 0.0);
    }

    #[test]
    fn two_prod_retains_exact_rounding_residual() {
        let value = 134_217_729.0_f64; // 2^27 + 1
        let product = DoubleDouble::from_f64(value) * DoubleDouble::from_f64(value);
        assert_eq!(product.hi, value * value);
        assert_eq!(product.lo, 1.0);
    }

    #[test]
    fn normalized_mul_preserves_small_residual_under_exact_integer_scale() {
        let residual = f64::EPSILON / 4.0;
        let value = DoubleDouble::from_parts(3.0, residual);
        let scaled = value * DoubleDouble::from_f64(7.0);
        let expected = DoubleDouble::from_parts(21.0, 7.0 * residual);
        let error = (scaled - expected).abs();
        assert_eq!(error, DoubleDouble::ZERO);
        assert!(scaled.lo.abs() <= f64::EPSILON * scaled.hi.abs());
    }

    #[test]
    fn division_refinement_recovers_double_double_dividend() {
        let value = DoubleDouble::from_parts(1.0e16, 1.0);
        let divisor = DoubleDouble::from_f64(3.0);
        let quotient = value / divisor;
        let recovered = quotient * divisor;
        let error = (recovered - value).abs();
        assert!(error.hi.abs() <= 1.0e-15, "{error:?}");
    }

    #[test]
    fn unit_root_quadrants_are_exact_and_conjugate() {
        let forward = unit_root(1, 4, Direction::Forward).unwrap();
        let inverse = unit_root(1, 4, Direction::Inverse).unwrap();
        assert_eq!(forward.re, DoubleDouble::ZERO);
        assert_eq!(forward.im, -DoubleDouble::ONE);
        assert_eq!(inverse, forward.conj());
    }

    #[test]
    fn sin_cos_preserves_unit_circle_beyond_f64_roundoff() {
        let angle = DoubleDouble::TAU / DoubleDouble::from_f64(7.0);
        let (sin, cos) = angle.sin_cos();
        let norm = sin * sin + cos * cos;
        let error = (norm - DoubleDouble::ONE).abs();
        assert!(error.hi.abs() < 1.0e-29, "{error:?}");
    }

    #[test]
    fn dft_retains_sub_f64_impulse_and_round_trips() {
        let base = DoubleDouble::from_f64(1.0e16);
        let mut input = vec![ComplexDoubleDouble::new(base, DoubleDouble::ZERO); 4];
        input[0].re += DoubleDouble::ONE;
        let spectrum = dft(&input, Direction::Forward, false).unwrap();
        let dc_offset = spectrum[0].re - DoubleDouble::from_f64(4.0e16);
        assert_eq!(dc_offset.to_f64(), 1.0);
        for bin in spectrum.iter().skip(1) {
            assert_eq!(bin.re.to_f64(), 1.0);
            assert_eq!(bin.im.to_f64(), 0.0);
        }
        let restored = dft(&spectrum, Direction::Inverse, true).unwrap();
        for (actual, expected) in restored.iter().zip(&input) {
            assert_eq!(*actual, *expected);
        }
    }

    #[test]
    fn complex_double_double_uses_double_double_channels() {
        let a = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.0e16, 1.0),
            DoubleDouble::from_f64(2.0),
        );
        let b = ComplexDoubleDouble::from_complex64(Complex64::new(1.0, -3.0));
        let product = a * b;

        // Real = (1e16 + 1) + 6; imaginary = -3*(1e16 + 1) + 2.
        let real_offset = product.re - DoubleDouble::from_f64(1.0e16);
        assert_eq!(real_offset.to_f64(), 7.0);
        let imag_offset = product.im - DoubleDouble::from_f64(-3.0e16);
        assert_eq!(imag_offset.to_f64(), -1.0);
        assert_eq!(a.conj().im.to_f64(), -2.0);
    }
}
