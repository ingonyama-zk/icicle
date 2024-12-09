use icicle_core::traits::{FieldImpl, ScalarImpl, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_scalar_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::fmt::{Debug, Display};
use std::ops::{Mul, Add, Sub};
use hex::FromHex;

pub(crate) const SCALAR_LIMBS: usize = 8;

impl_scalar_field!("stark252", stark252_f, stark252_sf, ScalarField, SCALAR_LIMBS);

#[cfg(test)]
mod tests {
    use super::ScalarField;
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
}
