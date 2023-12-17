use crate::icicle_core::traits::FieldImpl;
#[cfg(feature = "arkworks")]
use ark_bn254::{g1::Config as ArkG1Config, Fq, Fr};
use icicle_core::curve::{Affine, CurveConfig, Projective};
use icicle_core::field::{Field, FieldConfig};
use icicle_core::{impl_base_field, impl_scalar_field};
use std::ffi::c_uint;

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;

impl_scalar_field!(SCALAR_LIMBS, ScalarField, ScalarCfg);
impl_base_field!(BASE_LIMBS, BaseField, BaseCfg);
impl_curve!(ScalarField, BaseField);

#[cfg(test)]
mod tests {
    use super::{
        generate_random_affine_points, generate_random_projective_points, generate_random_scalars, BaseField, G1Affine,
        G1Projective, ScalarField, BASE_LIMBS,
    };
    use crate::icicle_core::traits::{FieldImpl, GetLimbs};
    use icicle_core::{impl_curve_ark_tests, impl_curve_tests, traits::ArkConvertible};

    use ark_bn254::G1Affine as ArkG1Affine;

    impl_curve_tests!();

    impl_curve_ark_tests!();
}
