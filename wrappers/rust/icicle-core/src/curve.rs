use crate::affine::Affine;
use crate::bignum::BigNum;
use crate::field::Field;
use crate::projective::Projective;
use crate::traits::{GenerateRandom, MontgomeryConvertible};

pub trait Curve {
    type BaseField: BigNum;
    type ScalarField: Field + MontgomeryConvertible + GenerateRandom;

    type Affine: Affine<BaseField = Self::BaseField>;
    type Projective: Projective<Affine = Self::Affine, ScalarField = Self::ScalarField, BaseField = Self::BaseField>;

    fn get_generator() -> Self::Projective;

    fn generate_random_affine_points(size: usize) -> Vec<Self::Affine> {
        Self::Affine::generate_random(size)
    }

    fn generate_random_projective_points(size: usize) -> Vec<Self::Projective> {
        Self::Projective::generate_random(size)
    }
}

#[macro_export]
macro_rules! impl_curve {
    (
        $curve_prefix:literal,
        $curve:ident,
        $scalar_field:ident,
        $base_field:ident,
        $affine_type:ident,
        $projective_type:ident
    ) => {
        icicle_core::impl_affine!($affine_type, $curve_prefix, $base_field);
        icicle_core::impl_projective!(
            $projective_type,
            $curve_prefix,
            $scalar_field,
            $base_field,
            $affine_type
        );

        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $curve {}

        impl icicle_core::curve::Curve for $curve {
            type BaseField = $base_field;
            type ScalarField = $scalar_field;

            type Affine = $affine_type;
            type Projective = $projective_type;

            fn get_generator() -> $projective_type {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_generator")]
                    pub(crate) fn generator(result: *mut $projective_type);
                }

                unsafe {
                    let mut result = $projective_type::zero();
                    generator(&mut result);
                    result
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_curve_tests {
    (
        $base_limbs:ident,
        $curve:ident
    ) => {
        pub mod test_curve {
            use super::*;
            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_affine_projective_convert() {
                initialize();
                check_affine_projective_convert::<$curve>()
            }

            #[test]
            fn test_point_equality() {
                initialize();
                check_point_equality::<$curve>()
            }

            #[test]
            fn test_points_convert_montgomery() {
                initialize();
                check_montgomery_convert_host::<<$curve as icicle_core::curve::Curve>::Projective>();
                check_montgomery_convert_host::<<$curve as icicle_core::curve::Curve>::Affine>();
                check_montgomery_convert_device::<<$curve as icicle_core::curve::Curve>::Projective>();
                check_montgomery_convert_device::<<$curve as icicle_core::curve::Curve>::Affine>();
            }

            #[test]
            fn test_point_arithmetic() {
                check_point_arithmetic::<$curve>();
            }

            #[test]
            fn test_generator() {
                check_generator::<$curve>();
            }
        }
    };
}
