use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::{Field, MontgomeryConvertibleField},
    impl_curve, impl_field, impl_scalar_field,
    traits::{FieldConfig, FieldImpl, GenerateRandom},
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;
#[cfg(feature = "g2")]
pub(crate) const G2_BASE_LIMBS: usize = 16;

impl_scalar_field!("bn254", bn254_sf, SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_field!("bn254_base_field", BASE_LIMBS, BaseField, BaseCfg);
impl_curve!("bn254", bn254, CurveCfg, ScalarField, BaseField, G1Affine, G1Projective);

#[cfg(feature = "g2")]
impl_field!("bn254_g2_base_field", G2_BASE_LIMBS, G2BaseField, G2BaseCfg);
#[cfg(feature = "g2")]
impl_curve!(
    "bn254_g2",
    bn254_g2,
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
    use icicle_core::traits::FieldImpl;
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
