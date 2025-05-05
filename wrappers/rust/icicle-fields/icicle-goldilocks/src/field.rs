use icicle_core::field::PrimeField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible};

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2; // Goldilocks uses 2 limbs for 64-bit field

impl_field!(ScalarField, "goldilocks", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarField, "goldilocks", goldilocks);
impl_montgomery_convertible!(ScalarField, goldilocks_scalar_convert_montgomery);
impl_generate_random!(ScalarField, goldilocks_generate_scalars);

#[cfg(test)]
mod tests {
    use super::ScalarField; // No extension field
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
}
