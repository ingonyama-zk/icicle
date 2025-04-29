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

impl_field!(GoldilocksField, "goldilocks", SCALAR_LIMBS, true);
impl_field_arithmetic!(GoldilocksField, "goldilocks", goldilocks);
impl_montgomery_convertible!(GoldilocksField, goldilocks_scalar_convert_montgomery);
impl_generate_random!(GoldilocksField, goldilocks_generate_scalars);

#[cfg(test)]
mod tests {
    use super::GoldilocksField; // No extension field
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(GoldilocksField);
}
