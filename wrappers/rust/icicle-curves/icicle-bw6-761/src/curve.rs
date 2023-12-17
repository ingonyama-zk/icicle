#[cfg(feature = "arkworks")]
use ark_bw6_761::{g1::Config as ArkG1Config, Fq, Fr};
use icicle_core::curve::{Affine, CurveConfig, Projective};
use icicle_core::field::{Field, FieldConfig};
use std::ffi::{c_uint, c_void};

pub(crate) const SCALAR_LIMBS: usize = 12;
pub(crate) const BASE_LIMBS: usize = 24;

impl_curve!(SCALAR_LIMBS, BASE_LIMBS,);

#[cfg(test)]
mod tests {
    use super::{
        generate_random_affine_points, generate_random_projective_points, generate_random_scalars, BaseField, G1Affine,
        G1Projective, ScalarField, BASE_LIMBS,
    };
    use icicle_core::{impl_curve_ark_tests, impl_curve_tests, traits::ArkConvertible};

    use ark_bw6_761::G1Affine as ArkG1Affine;

    impl_curve_tests!();

    impl_curve_ark_tests!();
}
