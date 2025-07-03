use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::{
    field::PrimeField, impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

pub(crate) const SCALAR_LIMBS: usize = 2;

impl_field!(ScalarRing, "labrador", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarRing, "labrador", labrador);
impl_montgomery_convertible!(ScalarRing, labrador_scalar_convert_montgomery);
impl_generate_random!(ScalarRing, labrador_generate_scalars);

impl_field!(ScalarRingRns, "labrador_rns", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarRingRns, "labrador_rns", labrador_rns);
impl_montgomery_convertible!(ScalarRingRns, labrador_rns_scalar_convert_montgomery);
impl_generate_random!(ScalarRingRns, labrador_rns_generate_scalars);

#[cfg(test)]
mod tests {
    use super::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_field_tests;
    use icicle_core::tests::*;

    impl_field_tests!(ScalarRing);
    mod rns {
        use super::*;
        impl_field_tests!(ScalarRingRns);
    }
}
