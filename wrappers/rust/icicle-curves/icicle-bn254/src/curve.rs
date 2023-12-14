#[cfg(feature = "arkworks")]
use ark_bn254::{g1::Config as ArkG1Config, Fq, Fr};
use icicle_core::curve::{Affine, CurveConfig, Projective};
use icicle_core::field::{Field, FieldConfig};
use std::ffi::{c_uint, c_void};

impl_curve!(
    8,
    8,
);

#[cfg(test)]
mod tests {
    use super::{
        generate_random_affine_points, generate_random_projective_points, generate_random_scalars, BaseField, G1Affine,
        G1Projective, ScalarField, BASE_LIMBS,
    };
    use icicle_core::{traits::ArkConvertible, impl_curve_tests, impl_curve_ark_tests};

    use ark_bn254::G1Affine as ArkG1Affine;

    impl_curve_tests!();

    impl_curve_ark_tests!();
}
