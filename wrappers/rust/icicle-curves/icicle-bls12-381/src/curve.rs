use crate::icicle_core::traits::FieldImpl;
#[cfg(feature = "arkworks")]
use ark_bls12_381::{g1::Config as ArkG1Config, Fq, Fr};
use icicle_core::curve::{Affine, CurveConfig, Projective};
use icicle_core::field::{Field, FieldConfig};
use std::ffi::c_uint;

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 12;

impl_curve!(SCALAR_LIMBS, BASE_LIMBS,);

#[cfg(test)]
mod tests {
    use super::{
        generate_random_affine_points, generate_random_projective_points, generate_random_scalars, BaseField, G1Affine,
        G1Projective, ScalarField, BASE_LIMBS,
    };
    use crate::icicle_core::traits::{FieldImpl, GetLimbs};
    use icicle_core::{impl_curve_ark_tests, impl_curve_tests, traits::ArkConvertible};

    use ark_bls12_381::G1Affine as ArkG1Affine;

    impl_curve_tests!();

    impl_curve_ark_tests!();
}
