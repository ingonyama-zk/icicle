use icicle_bls12_377::curve::BaseField as bls12_377BaseField;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::Field,
    impl_curve, impl_field,
    traits::FieldConfig,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, stream::IcicleStream};

pub(crate) const BASE_LIMBS: usize = 24;

impl_field!("bw6_761_g2_base_field", BASE_LIMBS, BaseField, BaseCfg);
pub type ScalarField = bls12_377BaseField;
impl_curve!(
    "bw6_761",
    bw6_761,
    CurveCfg,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);
#[cfg(not(feature = "no_g2"))]
impl_curve!(
    "bw6_761_g2",
    bw6_761_g2,
    G2CurveCfg,
    ScalarField,
    BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_g2"))]
    use super::G2CurveCfg;
    use super::{CurveCfg, ScalarField, BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::traits::FieldImpl;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(not(feature = "no_g2"))]
    mod g2 {
        use super::*;
        impl_curve_tests!(BASE_LIMBS, G2CurveCfg);
    }
}
