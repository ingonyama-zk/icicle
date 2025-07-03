use crate::field::PrimeField;
use icicle_runtime::memory::reinterpret::{reinterpret_slice, reinterpret_slice_mut};
use icicle_runtime::memory::HostOrDeviceSlice;

/// Trait representing a polynomial ring: R = Base[X] / (X^DEGREE - MODULUS_COEFF)
pub trait PolynomialRing: Sized + Clone + PartialEq + core::fmt::Debug {
    /// Base field type
    type Base: Copy;

    /// Number of terms in the polynomial (polynomials are degree < DEGREE)
    const DEGREE: usize;

    /// The constant term in the modulus polynomial: X^DEGREE - MODULUS_COEFF
    const MODULUS_COEFF: i32;

    /// Returns the internal values (coefficients or evaluations)
    fn values(&self) -> &[Self::Base];

    /// Mutable access to internal values
    fn values_mut(&mut self) -> &mut [Self::Base];

    /// Construct a zero polynomial (all values = 0)
    fn zero() -> Self;

    /// Construct from a slice (should panic or assert if length ≠ DEGREE)
    fn from_slice(values: &[Self::Base]) -> Self;
}

/// Reinterprets a slice of polynomials as a flat slice of their base field elements (read-only).
///
/// This is useful for passing polynomial vectors to scalar vectorized operations.
///
/// # Safety
/// - The layout of each `P` must match `[P::Base; DEGREE]` exactly (assumed via `#[repr(C)]`)
/// - The memory must be properly aligned and valid for reads
#[inline(always)]
pub fn flatten_polyring_slice<'a, P>(
    input: &'a (impl HostOrDeviceSlice<P> + ?Sized),
) -> impl HostOrDeviceSlice<P::Base> + 'a
where
    P: PolynomialRing,
    P::Base: PrimeField + 'a,
{
    // Note that this can never fail here for a valid P
    unsafe { reinterpret_slice::<P, P::Base>(input).expect("Internal error") }
}

/// Reinterprets a mutable slice of polynomials as a flat mutable slice of their base field elements.
///
/// # Safety
/// - The layout of each `P` must match `[P::Base; DEGREE]`
/// - Caller must ensure exclusive access and valid alignment for mutation
#[inline(always)]
pub fn flatten_polyring_slice_mut<'a, P>(
    input: &'a mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> impl HostOrDeviceSlice<P::Base> + 'a
where
    P: PolynomialRing,
    P::Base: PrimeField + 'a,
{
    // Note that this can never fail here for a valid P
    unsafe { reinterpret_slice_mut::<P, P::Base>(input).expect("Internal error") }
}

#[macro_export]
macro_rules! impl_polynomial_ring {
    ($polyring:ident, $base:ty, $degree:expr, $modulus_coeff:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct $polyring {
            values: [$base; $degree],
        }

        use icicle_core::polynomial_ring::PolynomialRing;
        use icicle_core::traits::GenerateRandom;

        impl PolynomialRing for $polyring {
            type Base = $base;

            const DEGREE: usize = $degree;
            const MODULUS_COEFF: i32 = $modulus_coeff;

            fn values(&self) -> &[Self::Base] {
                &self.values
            }

            fn values_mut(&mut self) -> &mut [Self::Base] {
                &mut self.values
            }

            fn zero() -> Self {
                Self {
                    values: [<$base>::zero(); $degree],
                }
            }

            fn from_slice(input: &[Self::Base]) -> Self {
                assert_eq!(input.len(), Self::DEGREE);
                let mut values = [<$base>::zero(); $degree];
                values.copy_from_slice(input);
                Self { values }
            }
        }

        impl GenerateRandom for $polyring {
            fn generate_random(size: usize) -> Vec<$polyring> {
                use std::mem::{forget, ManuallyDrop};
                use std::slice;

                let flat_base_field_vec: Vec<$base> = $base::generate_random(size * Self::DEGREE);

                let ptr = flat_base_field_vec.as_ptr() as *mut $polyring;
                let len = size;
                let cap = flat_base_field_vec.capacity() / Self::DEGREE;

                // Avoid double-drop
                forget(flat_base_field_vec);

                unsafe { Vec::from_raw_parts(ptr, len, cap) }
            }
        }
    };
}

#[macro_export]
macro_rules! test_polynomial_ring {
    ($type:ty) => {
        mod test_polynomial_ring {
            use super::*;
            use icicle_core::tests::*;

            #[test]
            fn test_zero_and_from_slice() {
                check_zero_and_from_slice::<$type>();
            }

            #[test]
            fn test_flatten_slices() {
                check_polyring_flatten_host_memory::<$type>();
                check_polyring_flatten_device_memory::<$type>();
            }
        }
    };
}
