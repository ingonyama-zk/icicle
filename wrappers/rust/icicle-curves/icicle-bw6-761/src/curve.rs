use crate::icicle_core::traits::{FieldImpl, GenerateRandom};
#[cfg(feature = "arkworks")]
use ark_bw6_761::{g1::Config as ArkG1Config, Fq, Fr};
use icicle_core::curve::{Affine, CurveConfig, Projective};
use icicle_core::field::{Field, FieldConfig};
use icicle_core::{impl_base_field, impl_scalar_field};
use std::ffi::c_uint;

pub(crate) const SCALAR_LIMBS: usize = 12;
pub(crate) const BASE_LIMBS: usize = 24;

impl_scalar_field!(SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_base_field!(BASE_LIMBS, BaseField, BaseCfg);
impl_curve!(ScalarField, BaseField);

#[cfg(test)]
mod tests {
    use super::{CurveCfg, CurveConfig, ScalarCfg, BASE_LIMBS};
    use icicle_core::tests::{check_affine_projective_convert, check_point_equality, check_scalar_equality};
    use icicle_core::{
        impl_curve_ark_tests, impl_curve_tests,
        traits::{ArkConvertible, GenerateRandom},
    };

    use ark_bw6_761::G1Affine as ArkG1Affine;

    impl_curve_tests!(BASE_LIMBS, CurveCfg);

    impl_curve_ark_tests!(CurveCfg, ArkG1Affine, ScalarCfg);
}
