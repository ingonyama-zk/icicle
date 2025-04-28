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

impl_field!(M31Field, "m31", SCALAR_LIMBS, true);
impl_field_arithmetic!(M31Field, "m31", m31);
impl_montgomery_convertible!(M31Field, m31_scalar_convert_montgomery);
impl_generate_random!(M31Field, m31_generate_scalars);

impl_field!(M31ExtensionField, "m31_extension", EXTENSION_LIMBS, true);
impl_field_arithmetic!(M31ExtensionField, "m31_extension", m31_extension);
impl_montgomery_convertible!(M31ExtensionField, m31_extension_scalar_convert_montgomery);
impl_generate_random!(M31ExtensionField, m31_extension_generate_scalars);

#[cfg(test)]
mod tests {
    use super::{M31ExtensionField, M31Field};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(M31Field);
    mod extension {
        use super::*;

        impl_field_tests!(M31ExtensionField);
    }
}
