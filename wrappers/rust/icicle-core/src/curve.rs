#[macro_export]
macro_rules! impl_curve {
    (
        $curve_prefix:literal,
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
    };
}

#[macro_export]
macro_rules! impl_curve_tests {
    (
        $base_limbs:ident,
        $projective_type:ident
    ) => {
        pub mod test_curve {
            use super::*;
            use icicle_core::projective::Projective;
            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_affine_projective_convert() {
                initialize();
                check_affine_projective_convert::<$projective_type>()
            }

            #[test]
            fn test_point_equality() {
                initialize();
                check_point_equality::<$projective_type>()
            }

            #[test]
            fn test_points_convert_montgomery() {
                initialize();
                check_montgomery_convert_host::<$projective_type>();
                check_montgomery_convert_host::<<$projective_type as Projective>::Affine>();
                check_montgomery_convert_device::<$projective_type>();
                check_montgomery_convert_device::<<$projective_type as Projective>::Affine>();
            }

            #[test]
            fn test_point_arithmetic() {
                check_point_arithmetic::<$projective_type>();
            }

            #[test]
            fn test_generator() {
                check_generator::<$projective_type>();
            }
        }
    };
}
