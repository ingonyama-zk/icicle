use icicle_core::field::PrimeField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible};

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::{HostOrDeviceSlice, IntoIcicleSliceMut};
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_field!(ScalarField, "koalabear", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarField, "koalabear", koalabear);
impl_montgomery_convertible!(ScalarField, koalabear_scalar_convert_montgomery);
impl_generate_random!(ScalarField, koalabear_generate_scalars);

impl_field!(ExtensionField, "koalabear_extension", EXTENSION_LIMBS, true);
impl_field_arithmetic!(ExtensionField, "koalabear_extension", koalabear_extension);
impl_montgomery_convertible!(ExtensionField, koalabear_extension_scalar_convert_montgomery);
impl_generate_random!(ExtensionField, koalabear_extension_generate_scalars);

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
