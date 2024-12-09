use icicle_core::traits::{FieldImpl, ScalarImpl, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::fmt::{Debug, Display};
use std::ops::{Mul, Add, Sub};
use hex::FromHex;

pub(crate) const SCALAR_LIMBS: usize = 1;
pub(crate) const EXTENSION_LIMBS: usize = 4;

impl_scalar_field!("babybear", babybear_f, babybear_sf, ScalarField, SCALAR_LIMBS);
impl_scalar_field!(
    "babybear_extension",
    babybear_extension_f,
    babybear_extension_sf,
    ExtensionField,
    EXTENSION_LIMBS
);

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
