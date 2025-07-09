use crate::ring::ScalarRing;
use icicle_core::bignum::BigNum;
use icicle_core::polynomial_ring::PolynomialRing;
use icicle_core::traits::GenerateRandom;
use icicle_runtime::errors::{eIcicleError, IcicleError};

// Define the Polynomial Ring Zq[X]/X^d+1
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct PolyRing {
    values: [ScalarRing; 64],
}

impl Default for PolyRing {
    fn default() -> Self {
        Self {
            values: [ScalarRing::default(); 64],
        }
    }
}

impl PolynomialRing for PolyRing {
    type Base = ScalarRing;

    const DEGREE: usize = 64;
    const MODULUS_COEFF: i32 = -1;

    fn values(&self) -> &[Self::Base] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [Self::Base] {
        &mut self.values
    }

    fn zero() -> Self {
        Self {
            values: [ScalarRing::zero(); 64],
        }
    }

    fn from_slice(input: &[Self::Base]) -> Result<Self, IcicleError> {
        if input.len() != Self::DEGREE {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "Input length does not match the degree of the polynomial ring",
            ));
        }
        let mut values = [ScalarRing::zero(); 64];
        values.copy_from_slice(input);
        Ok(Self { values })
    }
}

impl GenerateRandom for PolyRing {
    fn generate_random(size: usize) -> Vec<PolyRing> {
        use std::mem::forget;

        let flat_base_field_vec: Vec<ScalarRing> = ScalarRing::generate_random(size * Self::DEGREE);

        let ptr = flat_base_field_vec.as_ptr() as *mut PolyRing;
        let len = size;
        let cap = flat_base_field_vec.capacity() / Self::DEGREE;

        // Avoid double-drop
        forget(flat_base_field_vec);

        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }
}

#[cfg(test)]
mod tests {
    use crate::polynomial_ring::PolyRing;
    use icicle_core::test_polynomial_ring;

    test_polynomial_ring!(PolyRing);
}
