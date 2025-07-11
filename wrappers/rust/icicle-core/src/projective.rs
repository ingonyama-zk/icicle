use crate::{
    affine::Affine,
    bignum::BigNum,
    field::Field,
    traits::{GenerateRandom, MontgomeryConvertible},
};
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

/// A [projective](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) elliptic curve point.
pub trait Projective:
    Debug
    + Default
    + PartialEq
    + Copy
    + Clone
    + MontgomeryConvertible
    + GenerateRandom
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::ScalarField, Output = Self>
    + From<Self::Affine>
    + Into<Self::Affine>
{
    type ScalarField: Field + MontgomeryConvertible;
    type BaseField: Field;
    type Affine: Affine;

    fn x(&self) -> Self::BaseField;
    fn y(&self) -> Self::BaseField;
    fn z(&self) -> Self::BaseField;

    fn from_xyz(x: Self::BaseField, y: Self::BaseField, z: Self::BaseField) -> Self;

    fn from_limbs(
        x: <Self::BaseField as BigNum>::Limbs,
        y: <Self::BaseField as BigNum>::Limbs,
        z: <Self::BaseField as BigNum>::Limbs,
    ) -> Self {
        Self::from_xyz(
            Self::BaseField::from(x),
            Self::BaseField::from(y),
            Self::BaseField::from(z),
        )
    }

    fn is_on_curve(self) -> bool;
    fn to_affine(self) -> Self::Affine {
        self.into()
    }
    fn from_affine(aff: Self::Affine) -> Self {
        Self::from(aff)
    }

    fn zero() -> Self {
        Self::from_xyz(Self::BaseField::zero(), Self::BaseField::one(), Self::BaseField::zero())
    }

    fn get_generator() -> Self;
}

#[macro_export]
macro_rules! impl_projective {
    (
        $projective:ident,
        $curve_prefix:literal,
        $scalar_field:ident,
        $base_field:ident,
        $affine:ident
    ) => {
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct $projective {
            pub x: $base_field,
            pub y: $base_field,
            pub z: $base_field,
        }

        impl icicle_core::projective::Projective for $projective {
            type ScalarField = $scalar_field;
            type BaseField = $base_field;
            type Affine = $affine;

            fn x(&self) -> Self::BaseField {
                self.x
            }
            fn y(&self) -> Self::BaseField {
                self.y
            }
            fn z(&self) -> Self::BaseField {
                self.z
            }
            fn from_xyz(x: Self::BaseField, y: Self::BaseField, z: Self::BaseField) -> Self {
                Self { x, y, z }
            }

            fn is_on_curve(self) -> bool {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_is_on_curve")]
                    pub(crate) fn is_on_curve(point: *const $projective) -> bool;
                }
                unsafe { is_on_curve(&self as *const Self) }
            }

            fn get_generator() -> Self {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_generator")]
                    pub(crate) fn generator(result: *mut $projective);
                }

                unsafe {
                    let mut result = Self::zero();
                    generator(&mut result);
                    result
                }
            }
        }

        impl PartialEq for $projective {
            fn eq(&self, other: &Self) -> bool {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_projective_eq")]
                    pub(crate) fn eq(point1: *const $projective, point2: *const $projective) -> bool;
                }
                unsafe { eq(self as *const Self, other as *const Self) }
            }
        }

        impl std::ops::Add for $projective {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_ecadd")]
                    pub(crate) fn add(point1: *const $projective, point2: *const $projective, result: *mut $projective);
                }
                let mut result = Self::zero();
                unsafe {
                    add(
                        &self as *const Self,
                        &other as *const Self,
                        &mut result as *mut _ as *mut $projective,
                    )
                };
                result
            }
        }

        impl std::ops::Sub for $projective {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_ecsub")]
                    pub(crate) fn sub(point1: *const $projective, point2: *const $projective, result: *mut $projective);
                }
                let mut result = Self::zero();
                unsafe {
                    sub(
                        &self as *const Self,
                        &other as *const Self,
                        &mut result as *mut _ as *mut $projective,
                    )
                };
                result
            }
        }

        impl std::ops::Mul<$scalar_field> for $projective {
            type Output = Self;

            fn mul(self, other: $scalar_field) -> Self {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_mul_scalar")]
                    pub(crate) fn mul_scalar(
                        point1: *const $projective,
                        point2: *const $scalar_field,
                        result: *mut $projective,
                    );
                }
                let mut result = Self::zero();
                unsafe {
                    mul_scalar(
                        &self as *const Self,
                        &other as *const $scalar_field,
                        &mut result as *mut _ as *mut Self,
                    )
                };
                result
            }
        }

        impl From<$affine> for $projective
        where
            $affine: icicle_core::affine::Affine,
        {
            fn from(aff: $affine) -> Self {
                if aff == <$affine as icicle_core::affine::Affine>::zero() {
                    return Self::zero();
                }
                Self {
                    x: aff.x,
                    y: aff.y,
                    z: $base_field::one(),
                }
            }
        }

        impl Into<$affine> for $projective {
            fn into(self) -> $affine {
                extern "C" {
                    #[link_name = concat!($curve_prefix, "_to_affine")]
                    pub(crate) fn proj_to_affine(point: *const $projective, point_out: *mut $affine);
                }
                let mut aff = $affine::zero();
                unsafe { proj_to_affine(&self as *const Self, &mut aff as *mut _ as *mut $affine) };
                aff
            }
        }

        icicle_core::impl_generate_random!($projective, concat!($curve_prefix, "_generate_projective_points"));

        icicle_core::impl_montgomery_convertible!(
            $projective,
            concat!($curve_prefix, "_projective_convert_montgomery")
        );
    };
}
