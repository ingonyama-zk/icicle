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
pub(crate) const BASE_LIMBS: usize = 12;
#[cfg(not(feature = "no_g2"))]
pub(crate) const G2_BASE_LIMBS: usize = 24;

impl_scalar_field!("bls12_377", bls12_377_f, bls12_377_sf, ScalarField, SCALAR_LIMBS);

#[cfg(feature = "bw6-761")]
impl_scalar_field!("bw6_761", bw6_761_f, bw6_761_sf, BaseField, BASE_LIMBS);

#[cfg(not(feature = "bw6-761"))]
impl_field!("bls12_377_point_field", bls12_377_bf, BaseField, BASE_LIMBS);

impl_curve!(
    "bls12_377",
    bls12_377,
    Bls12377Curve,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);

#[cfg(not(feature = "no_g2"))]
impl_field!("bls12_377_g2_point_field", bls12_377_g2_bf, G2BaseField, G2_BASE_LIMBS);
#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bls12_377_g2",
    bls12_377_g2,
    Bls12377G2Curve,
    ScalarField,
    G2BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    use super::{ScalarField, BASE_LIMBS, Bls12377Curve};
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;
    use icicle_core::curve::Curve;

    #[cfg(not(feature = "no_g2"))]
    use super::{Bls12377G2Curve, G2_BASE_LIMBS};

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, Bls12377Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, Bls12377G2Curve);
    }
}
