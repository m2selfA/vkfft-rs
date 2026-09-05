//! Pure-Rust IEEE-754 binary16 storage values.
//!
//! VkFFT's half-precision mode stores complex values as two binary16 lanes while
//! computing in F32.  This module owns the exact host-side bit contract so mixed-
//! storage runtimes never depend on platform C `half` layouts or lossy ad-hoc casts.

use crate::complex::{Complex, Complex32};
use crate::error::{Result, VkFftError};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Binary16(u16);

pub type Complex16 = Complex<Binary16>;

impl Binary16 {
    pub const ZERO: Self = Self(0x0000);
    pub const NEG_ZERO: Self = Self(0x8000);
    pub const INFINITY: Self = Self(0x7c00);
    pub const NEG_INFINITY: Self = Self(0xfc00);
    pub const NAN: Self = Self(0x7e00);
    pub const MAX: Self = Self(0x7bff);
    pub const MIN_POSITIVE_NORMAL: Self = Self(0x0400);
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self(0x0001);

    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7c00) == 0x7c00 && (self.0 & 0x03ff) != 0
    }

    pub const fn is_infinite(self) -> bool {
        (self.0 & 0x7fff) == 0x7c00
    }

    pub const fn is_finite(self) -> bool {
        (self.0 & 0x7c00) != 0x7c00
    }

    /// Convert F32 to binary16 using IEEE round-to-nearest, ties-to-even.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x007f_ffff;

        if exponent == 0xff {
            if mantissa == 0 {
                return Self(sign | 0x7c00);
            }
            let payload = ((mantissa >> 13) as u16).max(1);
            return Self(sign | 0x7c00 | payload);
        }

        let mut half_exponent = exponent - 127 + 15;
        if half_exponent >= 0x1f {
            return Self(sign | 0x7c00);
        }

        if half_exponent <= 0 {
            if half_exponent < -10 {
                return Self(sign);
            }
            let mantissa = mantissa | 0x0080_0000;
            let shift = (14 - half_exponent) as u32;
            let mut half_mantissa = mantissa >> shift;
            let remainder_mask = (1u32 << shift) - 1;
            let remainder = mantissa & remainder_mask;
            let halfway = 1u32 << (shift - 1);
            if remainder > halfway || (remainder == halfway && (half_mantissa & 1) != 0) {
                half_mantissa += 1;
            }
            // Rounding the largest subnormal upward naturally produces 0x0400.
            return Self(sign | half_mantissa as u16);
        }

        let mut half_mantissa = mantissa >> 13;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1) != 0) {
            half_mantissa += 1;
            if half_mantissa == 0x0400 {
                half_mantissa = 0;
                half_exponent += 1;
                if half_exponent >= 0x1f {
                    return Self(sign | 0x7c00);
                }
            }
        }
        Self(sign | ((half_exponent as u16) << 10) | half_mantissa as u16)
    }

    /// Every finite binary16 value is exactly representable as F32.
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = ((bits & 0x8000) as u32) << 16;
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let mantissa = (bits & 0x03ff) as u32;
        let bits32 = match exponent {
            0 if mantissa == 0 => sign,
            0 => {
                let mut normalized = mantissa;
                let mut unbiased = -14i32;
                while normalized & 0x0400 == 0 {
                    normalized <<= 1;
                    unbiased -= 1;
                }
                normalized &= 0x03ff;
                sign | (((unbiased + 127) as u32) << 23) | (normalized << 13)
            }
            0x1f => sign | 0x7f80_0000 | (mantissa << 13),
            _ => {
                let exponent32 = (exponent - 15 + 127) as u32;
                sign | (exponent32 << 23) | (mantissa << 13)
            }
        };
        f32::from_bits(bits32)
    }
}

impl From<f32> for Binary16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Binary16> for f32 {
    fn from(value: Binary16) -> Self {
        value.to_f32()
    }
}

impl Complex<Binary16> {
    pub fn from_complex32(value: Complex32) -> Self {
        Self::new(Binary16::from_f32(value.re), Binary16::from_f32(value.im))
    }

    pub fn to_complex32(self) -> Complex32 {
        Complex32::new(self.re.to_f32(), self.im.to_f32())
    }
}

/// Encode caller-visible F32 complex values into the native-endian four-byte
/// binary16 complex ABI consumed by GPU storage buffers.
pub fn encode_complex16_native(values: &[Complex32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&Binary16::from_f32(value.re).to_bits().to_ne_bytes());
        bytes.extend_from_slice(&Binary16::from_f32(value.im).to_bits().to_ne_bytes());
    }
    bytes
}

/// Decode the native-endian binary16 complex storage ABI and widen every lane to F32.
pub fn decode_complex16_native(bytes: &[u8]) -> Result<Vec<Complex32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VkFftError::InvalidKernelIr(
            "binary16 complex byte buffer must be a multiple of four",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            let re = Binary16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32();
            let im = Binary16::from_bits(u16::from_ne_bytes([chunk[2], chunk[3]])).to_f32();
            Complex32::new(re, im)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary16_and_complex_layout_match_storage_contract() {
        assert_eq!(core::mem::size_of::<Binary16>(), 2);
        assert_eq!(core::mem::align_of::<Binary16>(), 2);
        assert_eq!(core::mem::size_of::<Complex16>(), 4);
    }

    #[test]
    fn known_ieee_values_encode_exactly() {
        assert_eq!(Binary16::from_f32(0.0).to_bits(), 0x0000);
        assert_eq!(Binary16::from_f32(-0.0).to_bits(), 0x8000);
        assert_eq!(Binary16::from_f32(1.0).to_bits(), 0x3c00);
        assert_eq!(Binary16::from_f32(-2.0).to_bits(), 0xc000);
        assert_eq!(Binary16::from_f32(65504.0).to_bits(), 0x7bff);
        assert_eq!(Binary16::from_f32(2.0f32.powi(-14)).to_bits(), 0x0400);
        assert_eq!(Binary16::from_f32(2.0f32.powi(-24)).to_bits(), 0x0001);
        assert_eq!(Binary16::from_f32(f32::INFINITY), Binary16::INFINITY);
        assert_eq!(
            Binary16::from_f32(f32::NEG_INFINITY),
            Binary16::NEG_INFINITY
        );
        assert!(Binary16::from_f32(f32::NAN).is_nan());
    }

    #[test]
    fn conversion_rounds_halfway_to_even() {
        let half_ulp = 2.0f32.powi(-11);
        assert_eq!(Binary16::from_f32(1.0 + half_ulp).to_bits(), 0x3c00);
        assert_eq!(Binary16::from_f32(1.0 + 3.0 * half_ulp).to_bits(), 0x3c02);
    }

    #[test]
    fn every_binary16_bit_pattern_round_trips_through_f32() {
        for bits in 0u16..=u16::MAX {
            let value = Binary16::from_bits(bits);
            let round_trip = Binary16::from_f32(value.to_f32());
            assert_eq!(round_trip.to_bits(), bits, "binary16 pattern 0x{bits:04x}");
        }
    }

    #[test]
    fn complex_storage_codec_round_trips_quantized_values() {
        let input = [
            Complex32::new(1.0004, -0.3333),
            Complex32::new(2.0f32.powi(-24), 65504.0),
            Complex32::new(-0.0, f32::INFINITY),
        ];
        let bytes = encode_complex16_native(&input);
        assert_eq!(bytes.len(), input.len() * 4);
        let decoded = decode_complex16_native(&bytes).unwrap();
        for (actual, source) in decoded.iter().zip(&input) {
            assert_eq!(actual.re, Binary16::from_f32(source.re).to_f32());
            assert_eq!(actual.im, Binary16::from_f32(source.im).to_f32());
        }
        assert!(matches!(
            decode_complex16_native(&bytes[..bytes.len() - 1]),
            Err(VkFftError::InvalidKernelIr(_))
        ));
    }

    #[test]
    fn complex_conversion_quantizes_each_lane_independently() {
        let input = Complex32::new(1.0004, -0.3333);
        let stored = Complex16::from_complex32(input);
        assert_eq!(stored.re, Binary16::from_f32(input.re));
        assert_eq!(stored.im, Binary16::from_f32(input.im));
        let restored = stored.to_complex32();
        assert_eq!(restored.re, stored.re.to_f32());
        assert_eq!(restored.im, stored.im.to_f32());
    }
}
