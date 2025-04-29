use icicle_core::traits::{Arithmetic, MontgomeryConvertible};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    impl_curve, impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible,
    traits::GenerateRandom,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 12;
#[cfg(not(feature = "no_g2"))]
pub(crate) const G2_BASE_LIMBS: usize = 24;

impl_field!(Bls12_381ScalarField, "bls12_381", SCALAR_LIMBS, true);
impl_field_arithmetic!(Bls12_381ScalarField, "bls12_381", bls12_381_sf);
impl_montgomery_convertible!(Bls12_381ScalarField, bls12_381_scalar_convert_montgomery);
impl_generate_random!(Bls12_381ScalarField, bls12_381_generate_scalars);

impl_field!(Bls12381BaseField, "bls12_381_base_field", BASE_LIMBS, false);

impl_curve!(
    "bls12_381",
    bls12_381,
    CurveCfg,
    Bls12_381ScalarField,
    Bls12381BaseField,
    G1Affine,
    G1Projective
);

#[cfg(not(feature = "no_g2"))]
impl_field!(Bls12381G2BaseField, "bls12_381_g2_base_field", G2_BASE_LIMBS, false);

#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bls12_381_g2",
    bls12_381_g2,
    G2CurveCfg,
    Bls12_381ScalarField,
    Bls12381G2BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_g2"))]
    use super::G2CurveCfg;
    use super::{Bls12_381ScalarField, CurveCfg};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(Bls12_381ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, G2CurveCfg);
    }
}
