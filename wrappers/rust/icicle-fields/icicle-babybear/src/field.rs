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

impl_field!(BabybearField, "babybear", SCALAR_LIMBS, true);
impl_field_arithmetic!(BabybearField, "babybear", babybear);
impl_montgomery_convertible!(BabybearField, babybear_scalar_convert_montgomery);
impl_generate_random!(BabybearField, babybear_generate_scalars);

impl_field!(BabybearExtensionField, "babybear_extension", EXTENSION_LIMBS, true);
impl_field_arithmetic!(BabybearExtensionField, "babybear_extension", babybear_extension);
impl_montgomery_convertible!(BabybearExtensionField, babybear_extension_scalar_convert_montgomery);
impl_generate_random!(BabybearExtensionField, babybear_extension_generate_scalars);

#[cfg(test)]
mod tests {
    use super::{BabybearExtensionField, BabybearField};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(BabybearField);
    mod extension {
        use super::*;

        impl_field_tests!(BabybearExtensionField);
    }
}
