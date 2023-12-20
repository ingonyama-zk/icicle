use crate::icicle_core::traits::{FieldImpl, GenerateRandom};
#[cfg(feature = "arkworks")]
use ark_bls12_381::{g1::Config as ArkG1Config, Fq, Fr};
use icicle_core::curve::{Affine, CurveConfig, Projective};
use icicle_core::field::{Field, FieldConfig};
use icicle_core::{impl_base_field, impl_scalar_field};
use std::ffi::c_uint;

pub(crate) const SCALAR_LIMBS: usize = 4;
pub(crate) const BASE_LIMBS: usize = 6;

impl_scalar_field!(SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_base_field!(BASE_LIMBS, BaseField, BaseCfg);
impl_curve!(ScalarField, BaseField);

#[cfg(test)]
mod tests {
    use super::{CurveCfg, CurveConfig, ScalarCfg, BaseCfg, BASE_LIMBS};
    use icicle_core::tests::{check_affine_projective_convert, check_point_equality, check_scalar_equality};
    use icicle_core::{
        curve::{Affine, Projective},
        impl_curve_ark_tests, impl_curve_tests,
        traits::{ArkConvertible, GenerateRandom},
    };

    use ark_bls12_381::G1Affine as ArkG1Affine;

    impl_curve_tests!(BASE_LIMBS, BaseCfg, CurveCfg);

    impl_curve_ark_tests!(CurveCfg, ArkG1Affine, ScalarCfg);
}
