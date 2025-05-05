use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::{Field, MontgomeryConvertibleField},
    impl_curve, impl_field, impl_scalar_field,
    traits::{FieldConfig, FieldImpl, GenerateRandom},
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 12;
pub(crate) const BASE_LIMBS: usize = 24;

impl_scalar_field!("bw6_761", bw6_761_sf, SCALAR_LIMBS, ScalarField, ScalarCfg);

// NOTE: Even though both G1 and G2 use the same base field, we define two different field types
//       to avoid using incorrect FFI functions.
impl_field!("bw6_761_base_field", BASE_LIMBS, BaseField, BaseCfg);
#[cfg(feature = "g2")]
impl_field!("bw6_761_g2_base_field", BASE_LIMBS, G2BaseField, G2BaseCfg);

impl_curve!(
    "bw6_761",
    bw6_761,
    CurveCfg,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);

#[cfg(feature = "g2")]
impl_curve!(
    "bw6_761_g2",
    bw6_761_g2,
    G2CurveCfg,
    ScalarField,
    G2BaseField,
    G2Affine,
    G2Projective
);

#[cfg(test)]
mod tests {
    #[cfg(feature = "g2")]
    use super::G2CurveCfg;
    use super::{CurveCfg, ScalarField, BASE_LIMBS};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_curve_tests!(BASE_LIMBS, G2CurveCfg);
    }
}