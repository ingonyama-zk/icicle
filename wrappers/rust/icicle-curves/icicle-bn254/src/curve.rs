use hex::FromHex;
use std::fmt::{Debug, Display};
use std::ops::{Mul, Add, Sub};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    traits::{FieldImpl, ScalarImpl, MontgomeryConvertible, GenerateRandom},
    impl_curve, impl_field, impl_scalar_field,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{
    stream::{IcicleStream},
    memory::HostOrDeviceSlice,
    errors::eIcicleError,
};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;
#[cfg(not(feature = "no_g2"))]
pub(crate) const G2_BASE_LIMBS: usize = 16;

impl_scalar_field!("bn254", bn254_f, bn254_sf, ScalarField, SCALAR_LIMBS);
impl_field!("bn254_point_field", bn254_bf, BaseField, BASE_LIMBS);

impl_curve!("bn254", bn254, Bn254Curve, ScalarField, BaseField, G1Affine, G1Projective);

#[cfg(not(feature = "no_g2"))]
impl_field!("bn254_g2_point_field", bn254_g2_bf, G2BaseField, G2_BASE_LIMBS);
#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bn254_g2",
    bn254_g2,
    Bn254G2Curve,
    ScalarField,
    G2BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    use super::{ScalarField, BASE_LIMBS, Bn254Curve};
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;
    use icicle_core::curve::Curve;

    #[cfg(not(feature = "no_g2"))]
    use super::{Bn254G2Curve, G2_BASE_LIMBS};

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, Bn254Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, Bn254G2Curve);
    }
}
