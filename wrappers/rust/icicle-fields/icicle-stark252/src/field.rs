use icicle_core::field::PrimeField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible};

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

use icicle_runtime::errors::{eIcicleError, IcicleError};
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 8;

impl_field!(ScalarField, "stark252", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarField, "stark252", stark252);
impl_montgomery_convertible!(ScalarField, stark252_scalar_convert_montgomery);
impl_generate_random!(ScalarField, stark252_generate_scalars);

#[cfg(test)]
mod tests {
    use super::ScalarField;
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarField);
}
