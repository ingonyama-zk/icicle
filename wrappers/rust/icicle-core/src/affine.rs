use crate::{
    bignum::BigNum,
    traits::{GenerateRandom, MontgomeryConvertible},
};
use std::fmt::Debug;

/// An [affine](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) elliptic curve point.
pub trait Affine: Debug + Default + PartialEq + Copy + Clone + MontgomeryConvertible + GenerateRandom {
    type BaseField: BigNum;

    fn x(&self) -> Self::BaseField;
    fn y(&self) -> Self::BaseField;
    fn from_xy(x: Self::BaseField, y: Self::BaseField) -> Self;

    fn from_limbs(x: <Self::BaseField as BigNum>::Limbs, y: <Self::BaseField as BigNum>::Limbs) -> Self {
        Self::from_xy(Self::BaseField::from(x), Self::BaseField::from(y))
    }

    // While this is not a true zero point and not even a valid point, it's still useful
    // both as a handy default as well as a representation of zero points in other codebases
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
        #[derive(Debug, Default, PartialEq, Copy, Clone)]
        #[repr(C)]
        pub struct $affine {
            x: $base_field,
            y: $base_field,
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

                let mut res = vec![$affine::zero(); size];
                unsafe { generate_affine_points(&mut res as *mut _ as *mut $affine, size) };
                res
            }
        }

        icicle_core::impl_montgomery_convertible_ffi!($affine, concat!($curve_prefix, "_affine_convert_montgomery"));
    };
}
