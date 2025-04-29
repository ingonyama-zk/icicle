use icicle_bls12_377::curve::Bls12_377BaseField;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    impl_curve, impl_field,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, stream::IcicleStream};
use std::fmt::{Debug, Display};

pub(crate) const BASE_LIMBS: usize = 24;

impl_field!(Bw6761BaseField, "bw6_761_g2_base_field", BASE_LIMBS, false);

pub type Bw6761ScalarField = Bls12_377BaseField;
impl_curve!(
    "bw6_761",
    bw6_761,
    CurveCfg,
    Bw6761ScalarField,
    Bw6761BaseField,
    G1Affine,
    G1Projective
);
#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bw6_761_g2",
    bw6_761_g2,
    G2CurveCfg,
    Bw6761ScalarField,
    Bw6761BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_g2"))]
    use super::G2CurveCfg;
    use super::{Bw6761ScalarField, CurveCfg};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(Bw6761ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(BASE_LIMBS, G2CurveCfg);
    }
}
