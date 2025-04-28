use icicle_core::field::PrimeField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible};

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;

pub(crate) const SCALAR_LIMBS: usize = 2;

impl_field!(LabradorScalarRing, "labrador", SCALAR_LIMBS);
impl_field_arithmetic!(LabradorScalarRing, "labrador", labrador);
impl_montgomery_convertible!(LabradorScalarRing, labrador_scalar_convert_montgomery);
impl_generate_random!(LabradorScalarRing, labrador_generate_scalars);

impl_field!(LabradorScalarRingRns, "labrador_rns", SCALAR_LIMBS);
impl_field_arithmetic!(LabradorScalarRingRns, "labrador_rns", labrador_rns);
impl_montgomery_convertible!(LabradorScalarRingRns, labrador_rns_scalar_convert_montgomery);
impl_generate_random!(LabradorScalarRingRns, labrador_rns_generate_scalars);

#[cfg(test)]
mod tests {
    use super::{LabradorScalarRing, LabradorScalarRingRns};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(LabradorScalarRing);
    mod rns {
        use super::*;
        impl_field_tests!(LabradorScalarRingRns);
    }
}
