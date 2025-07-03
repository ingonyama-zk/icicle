use icicle_core::field::PrimeField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible};

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

use icicle_runtime::errors::{eIcicleError, IcicleError};
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2; // Goldilocks uses 2 limbs for 64-bit field
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_field!(ScalarField, "goldilocks", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarField, "goldilocks", goldilocks);
impl_montgomery_convertible!(ScalarField, goldilocks_scalar_convert_montgomery);
impl_generate_random!(ScalarField, goldilocks_generate_scalars);
impl_field!(ExtensionField, "goldilocks_extension", EXTENSION_LIMBS, true);
impl_field_arithmetic!(ExtensionField, "goldilocks_extension", goldilocks_extension);
impl_montgomery_convertible!(ExtensionField, goldilocks_extension_scalar_convert_montgomery);
impl_generate_random!(ExtensionField, goldilocks_extension_generate_scalars);

#[cfg(test)]
mod tests {
    use super::{ExtensionField, ScalarField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
    mod extension {
        use super::*;

        impl_field_tests!(ExtensionField);
    }
}
