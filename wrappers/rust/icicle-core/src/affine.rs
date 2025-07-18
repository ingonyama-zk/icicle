use crate::{
    bignum::BigNum,
    field::Field,
    traits::{GenerateRandom, MontgomeryConvertible},
};
use std::fmt::Debug;

/// An [affine](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) elliptic curve point.
pub trait Affine: Debug + Default + PartialEq + Copy + Clone + MontgomeryConvertible + GenerateRandom {
    type BaseField: Field;

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
            pub x: $base_field,
            pub y: $base_field,
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

        icicle_core::impl_generate_random!($affine, concat!($curve_prefix, "_generate_affine_points"));
        icicle_core::impl_montgomery_convertible!($affine, concat!($curve_prefix, "_affine_convert_montgomery"));
    };
}
