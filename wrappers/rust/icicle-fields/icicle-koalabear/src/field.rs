use icicle_core::field::PrimeField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible};

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_field!(KoalabearField, "koalabear", SCALAR_LIMBS);
impl_field_arithmetic!(KoalabearField, "koalabear", koalabear);
impl_montgomery_convertible!(KoalabearField, koalabear_scalar_convert_montgomery);
impl_generate_random!(KoalabearField, koalabear_generate_scalars);

impl_field!(KoalabearExtensionField, "koalabear_extension", EXTENSION_LIMBS);
impl_field_arithmetic!(KoalabearExtensionField, "koalabear_extension", koalabear_extension);
impl_montgomery_convertible!(KoalabearExtensionField, koalabear_extension_scalar_convert_montgomery);
impl_generate_random!(KoalabearExtensionField, koalabear_extension_generate_scalars);

#[cfg(test)]
mod tests {
    use super::{KoalabearExtensionField, KoalabearField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(KoalabearField);
    mod extension {
        use super::*;

        impl_field_tests!(KoalabearExtensionField);
    }
}
