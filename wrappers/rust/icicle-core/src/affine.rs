use crate::{
    bignum::BigNum,
    traits::{GenerateRandom, MontgomeryConvertible, Zero},
};
use std::fmt::Debug;

/// An [affine](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) elliptic curve point.
pub trait Affine: Zero + Debug + PartialEq + Copy + Clone + MontgomeryConvertible + GenerateRandom {
    type BaseField: BigNum;

    fn x(&self) -> Self::BaseField;
    fn y(&self) -> Self::BaseField;
    fn from_xy(x: Self::BaseField, y: Self::BaseField) -> Self;

    fn from_limbs(x: <Self::BaseField as BigNum>::Limbs, y: <Self::BaseField as BigNum>::Limbs) -> Self {
        Self::from_xy(Self::BaseField::from(x), Self::BaseField::from(y))
    }

    fn zero() -> Self {
        Self::from_xy(Self::BaseField::zero(), Self::BaseField::zero())
    }
}

#[macro_export]
macro_rules! impl_affine {
    (
        $affine:ident,
        $curve_prefix:literal,
        $base_field:ident
    ) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        #[repr(C)]
        pub struct $affine {
            x: $base_field,
            y: $base_field,
        }

        impl $affine {
            pub fn convert_montgomery(
                input: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                is_into: bool,
                stream: &IcicleStream,
            ) -> eIcicleError {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_affine_convert_montgomery")]
                    pub(crate) fn convert_affine_montgomery(
                        input: *const $affine,
                        size: usize,
                        is_into: bool,
                        config: &icicle_core::vec_ops::VecOpsConfig,
                        output: *mut $affine,
                    ) -> eIcicleError;
                }

                if input.is_on_device() && !input.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = icicle_core::vec_ops::VecOpsConfig::default();
                config.is_a_on_device = input.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                unsafe {
                    convert_affine_montgomery(
                        input.as_mut_ptr(),
                        input.len(),
                        is_into,
                        &config,
                        input.as_mut_ptr(),
                    )
                }
            }
        }

        impl icicle_core::affine::Affine for $affine {
            type BaseField = $base_field;

            fn x(&self) -> Self::BaseField {
                self.x
            }

            fn y(&self) -> Self::BaseField {
                self.y
            }

            fn from_xy(x: Self::BaseField, y: Self::BaseField) -> Self {
                Self { x, y }
            }
        }

        impl icicle_core::traits::GenerateRandom for $affine {
            fn generate_random(size: usize) -> Vec<Self> {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_generate_affine_points")]
                    pub(crate) fn generate_affine_points(points: *mut $affine, size: usize);
                }

                let mut res = vec![<$affine as icicle_core::traits::Zero>::zero(); size];
                unsafe { generate_affine_points(&mut res as *mut _ as *mut $affine, size) };
                res
            }
        }

        impl icicle_core::traits::MontgomeryConvertible for $affine {
            fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
                $affine::convert_montgomery(values, true, stream)
            }

            fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
                $affine::convert_montgomery(values, false, stream)
            }
        }

        impl icicle_core::traits::Zero for $affine {
            // While this is not a true zero point and not even a valid point, it's still useful
            // both as a handy default as well as a representation of zero points in other codebases
            fn zero() -> Self {
                <Self as icicle_core::affine::Affine>::zero()
            }
        }
    };
}
