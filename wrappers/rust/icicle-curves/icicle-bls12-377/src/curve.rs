use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    impl_curve, impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::{HostOrDeviceSlice, IntoIcicleSliceMut}, stream::IcicleStream};
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 12;
#[cfg(not(feature = "no_g2"))]
pub(crate) const G2_BASE_LIMBS: usize = 24;

impl_field!(ScalarField, "bls12_377", SCALAR_LIMBS, true);
impl_field_arithmetic!(ScalarField, "bls12_377", bls12_377_sf);
impl_montgomery_convertible!(ScalarField, bls12_377_scalar_convert_montgomery);
impl_generate_random!(ScalarField, bls12_377_generate_scalars);

#[cfg(not(feature = "bw6-761"))]
impl_field!(BaseField, "bw6_761", BASE_LIMBS, false);
#[cfg(feature = "bw6-761")]
impl_field!(BaseField, "bw6_761", BASE_LIMBS, true);
#[cfg(feature = "bw6-761")]
impl_field_arithmetic!(BaseField, "bw6_761", bw6_761_sf);
#[cfg(feature = "bw6-761")]
impl_montgomery_convertible!(BaseField, bw6_761_scalar_convert_montgomery);
#[cfg(feature = "bw6-761")]
impl_generate_random!(BaseField, bw6_761_generate_scalars);

impl_curve!(
    "bls12_377",
    bls12_377,
    CurveCfg,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);

#[cfg(not(feature = "no_g2"))]
impl_field!(G2BaseField, "bls12_377_g2_base_field", G2_BASE_LIMBS, false);

#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bls12_377_g2",
    bls12_377_g2,
    G2CurveCfg,
    ScalarField,
    G2BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_g2"))]
    use super::G2CurveCfg;
    use super::{CurveCfg, ScalarField};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, G2CurveCfg);
    }
}
