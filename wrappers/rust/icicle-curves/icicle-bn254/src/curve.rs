use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    impl_curve, impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;
#[cfg(not(feature = "no_g2"))]
pub(crate) const G2_BASE_LIMBS: usize = 16;

impl_field!(Bn254ScalarField, "bn254", SCALAR_LIMBS, true);
impl_field_arithmetic!(Bn254ScalarField, "bn254", bn254_sf);
impl_montgomery_convertible!(Bn254ScalarField, bn254_scalar_convert_montgomery);
impl_generate_random!(Bn254ScalarField, bn254_generate_scalars);

impl_field!(Bn254BaseField, "bn254_base_field", BASE_LIMBS, false);
impl_curve!(
    "bn254",
    bn254,
    CurveCfg,
    Bn254ScalarField,
    Bn254BaseField,
    G1Affine,
    G1Projective
);

#[cfg(not(feature = "no_g2"))]
impl_field!(G2Bn254BaseField, "bn254_g2_base_field", G2_BASE_LIMBS, false);
#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bn254_g2",
    bn254_g2,
    G2CurveCfg,
    Bn254ScalarField,
    G2Bn254BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_g2"))]
    use super::G2CurveCfg;
    use super::{Bn254ScalarField, CurveCfg};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(Bn254ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, G2CurveCfg);
    }
}
