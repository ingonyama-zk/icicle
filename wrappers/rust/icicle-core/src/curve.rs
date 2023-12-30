#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::traits::FieldImpl;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective, SWCurveConfig};
use std::ffi::c_uint;
use std::fmt::Debug;

pub trait Curve: Debug + PartialEq + Copy + Clone {
    type BaseField: FieldImpl;
    type ScalarField: FieldImpl;

    #[doc(hidden)]
    fn eq_proj(point1: *const Projective<Self>, point2: *const Projective<Self>) -> c_uint;
    #[doc(hidden)]
    fn to_affine(point: *const Projective<Self>, point_aff: *mut Affine<Self>);
    #[doc(hidden)]
    fn generate_random_projective_points(size: usize) -> Vec<Projective<Self>>;
    #[doc(hidden)]
    fn generate_random_affine_points(size: usize) -> Vec<Affine<Self>>;

    #[cfg(feature = "arkworks")]
    type ArkSWConfig: SWCurveConfig;
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Projective<C: Curve> {
    pub x: C::BaseField,
    pub y: C::BaseField,
    pub z: C::BaseField,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Affine<C: Curve> {
    pub x: C::BaseField,
    pub y: C::BaseField,
}

impl<C: Curve> Affine<C> {
    // While this is not a true zero point and not even a valid point, it's still useful
    // both as a handy default as well as a representation of zero points in other codebases
    pub fn zero() -> Self {
        Affine {
            x: C::BaseField::zero(),
            y: C::BaseField::zero(),
        }
    }

    pub fn from_limbs(x: <C::BaseField as FieldImpl>::Repr, y: <C::BaseField as FieldImpl>::Repr) -> Self {
        Affine {
            x: C::BaseField::from(x),
            y: C::BaseField::from(y),
        }
    }

    pub fn to_projective(&self) -> Projective<C> {
        Projective {
            x: self.x,
            y: self.y,
            z: C::BaseField::one(),
        }
    }
}

impl<C: Curve> From<Affine<C>> for Projective<C> {
    fn from(item: Affine<C>) -> Self {
        Self {
            x: item.x,
            y: item.y,
            z: C::BaseField::one(),
        }
    }
}

impl<C: Curve> Projective<C> {
    pub fn zero() -> Self {
        Projective {
            x: C::BaseField::zero(),
            y: C::BaseField::one(),
            z: C::BaseField::zero(),
        }
    }

    pub fn from_limbs(
        x: <C::BaseField as FieldImpl>::Repr,
        y: <C::BaseField as FieldImpl>::Repr,
        z: <C::BaseField as FieldImpl>::Repr,
    ) -> Self {
        Projective {
            x: C::BaseField::from(x),
            y: C::BaseField::from(y),
            z: C::BaseField::from(z),
        }
    }
}

impl<C: Curve> PartialEq for Projective<C> {
    fn eq(&self, other: &Self) -> bool {
        C::eq_proj(self as *const _, other as *const _) != 0
    }
}

impl<C: Curve> From<Projective<C>> for Affine<C> {
    fn from(proj: Projective<C>) -> Self {
        let mut aff = Self::zero();
        C::to_affine(&proj as *const _, &mut aff as *mut _);
        aff
    }
}

#[cfg(feature = "arkworks")]
impl<C: Curve> ArkConvertible for Affine<C>
where
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    type ArkEquivalent = ArkAffine<C::ArkSWConfig>;

    fn to_ark(&self) -> Self::ArkEquivalent {
        let proj_x = self
            .x
            .to_ark();
        let proj_y = self
            .y
            .to_ark();
        Self::ArkEquivalent::new_unchecked(proj_x, proj_y)
    }

    fn from_ark(ark: Self::ArkEquivalent) -> Self {
        Self {
            x: C::BaseField::from_ark(ark.x),
            y: C::BaseField::from_ark(ark.y),
        }
    }
}

#[cfg(feature = "arkworks")]
impl<C: Curve> ArkConvertible for Projective<C>
where
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    type ArkEquivalent = ArkProjective<C::ArkSWConfig>;

    fn to_ark(&self) -> Self::ArkEquivalent {
        let proj_x = self
            .x
            .to_ark();
        let proj_y = self
            .y
            .to_ark();
        let proj_z = self
            .z
            .to_ark();

        // conversion between projective used in icicle and Jacobian used in arkworks
        let proj_x = proj_x * proj_z;
        let proj_y = proj_y * proj_z * proj_z;
        Self::ArkEquivalent::new_unchecked(proj_x, proj_y, proj_z)
    }

    fn from_ark(ark: Self::ArkEquivalent) -> Self {
        // conversion between Jacobian used in arkworks and projective used in icicle
        let proj_x = ark.x * ark.z;
        let proj_z = ark.z * ark.z * ark.z;
        Self {
            x: C::BaseField::from_ark(proj_x),
            y: C::BaseField::from_ark(ark.y),
            z: C::BaseField::from_ark(proj_z),
        }
    }
}

#[macro_export]
macro_rules! impl_curve {
    (
        $curve_prefix:literal,
        $curve:ident,
        $scalar_field:ident,
        $base_field:ident
    ) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $curve {}

        pub type G1Affine = Affine<$curve>;
        pub type G1Projective = Projective<$curve>;

        extern "C" {
            #[link_name = concat!($curve_prefix, "Eq")]
            fn eq(point1: *const G1Projective, point2: *const G1Projective) -> c_uint;
            #[link_name = concat!($curve_prefix, "ToAffine")]
            fn proj_to_affine(point: *const G1Projective, point_out: *mut G1Affine);
            #[link_name = concat!($curve_prefix, "GenerateProjectivePoints")]
            fn generate_projective_points(points: *mut G1Projective, size: usize);
            #[link_name = concat!($curve_prefix, "GenerateAffinePoints")]
            fn generate_affine_points(points: *mut G1Affine, size: usize);
        }

        impl Curve for $curve {
            type BaseField = $base_field;
            type ScalarField = $scalar_field;

            fn eq_proj(point1: *const G1Projective, point2: *const G1Projective) -> c_uint {
                unsafe { eq(point1, point2) }
            }

            fn to_affine(point: *const Projective<$curve>, point_out: *mut Affine<$curve>) {
                unsafe { proj_to_affine(point, point_out) };
            }

            fn generate_random_projective_points(size: usize) -> Vec<G1Projective> {
                let mut res = vec![G1Projective::zero(); size];
                unsafe { generate_projective_points(&mut res[..] as *mut _ as *mut G1Projective, size) };
                res
            }

            fn generate_random_affine_points(size: usize) -> Vec<G1Affine> {
                let mut res = vec![G1Affine::zero(); size];
                unsafe { generate_affine_points(&mut res[..] as *mut _ as *mut G1Affine, size) };
                res
            }

            #[cfg(feature = "arkworks")]
            type ArkSWConfig = ArkG1Config;
        }
    };
}

#[macro_export]
macro_rules! impl_curve_tests {
    (
        $base_limbs:ident,
        $curve:ident
    ) => {
        #[test]
        fn test_scalar_equality() {
            check_scalar_equality::<<$curve as Curve>::ScalarField>()
        }

        #[test]
        fn test_affine_projective_convert() {
            check_affine_projective_convert::<$curve>()
        }

        #[test]
        fn test_point_equality() {
            check_point_equality::<$base_limbs, <<$curve as Curve>::BaseField as FieldImpl>::Config, $curve>()
        }

        #[test]
        fn test_ark_scalar_convert() {
            check_ark_scalar_convert::<<$curve as Curve>::ScalarField>()
        }

        #[test]
        fn test_ark_point_convert() {
            check_ark_point_convert::<$curve>()
        }
    };
}
