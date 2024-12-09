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
use icicle_bls12_377::curve::BaseField as bls12_377BaseField;

pub(crate) const BASE_LIMBS: usize = 24;

impl_field!("bw6_761_point_field", bw6_761_bf, BaseField, BASE_LIMBS);
pub type ScalarField = bls12_377BaseField;
impl_curve!(
    "bw6_761",
    bw6_761,
    Bw6761Curve,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);
#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bw6_761_g2",
    bw6_761_g2,
    Bw6761G2Curve,
    ScalarField,
    BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    use super::{Bw6761Curve, ScalarField, BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::traits::FieldImpl;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;
    
    #[cfg(not(feature = "no_g2"))]
    use super::Bw6761G2Curve;
    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, Bw6761Curve);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(BASE_LIMBS, Bw6761G2Curve);
    }
}
