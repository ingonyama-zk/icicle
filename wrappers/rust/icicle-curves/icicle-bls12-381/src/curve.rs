use icicle_core::traits::{Arithmetic, MontgomeryConvertible};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    impl_curve, impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible,
    traits::GenerateRandom,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, IcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 12;
#[cfg(feature = "g2")]
pub(crate) const G2_BASE_LIMBS: usize = 24;

impl_field!(ScalarField, "bls12_381", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarField, "bls12_381", bls12_381_sf);
impl_montgomery_convertible!(ScalarField, bls12_381_scalar_convert_montgomery);
impl_generate_random!(ScalarField, bls12_381_generate_scalars);

impl_field!(BaseField, "bls12_381_base_field", BASE_LIMBS, false);
impl_curve!(
    "bls12_381",
    bls12_381,
    CurveCfg,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);

#[cfg(feature = "g2")]
impl_field!(G2BaseField, "bls12_381_g2_base_field", G2_BASE_LIMBS, false);
#[cfg(feature = "g2")]
impl_curve!(
    "bls12_381_g2",
    bls12_381_g2,
    G2CurveCfg,
    ScalarField,
    G2BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    use super::{CurveCfg, ScalarField, BASE_LIMBS};
    #[cfg(feature = "g2")]
    use super::{G2CurveCfg, G2_BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, G2CurveCfg);
    }
}
