#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::traits::FieldImpl;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective, SWCurveConfig};
use std::ffi::c_uint;
use std::fmt::Debug;

pub trait CurveConfig: Debug + PartialEq + Copy + Clone {
    type BaseField: FieldImpl;
    type ScalarField: FieldImpl;

    fn eq_proj(point1: *const Projective<Self>, point2: *const Projective<Self>) -> c_uint;
    fn to_affine(point: *const Projective<Self>, point_aff: *mut Affine<Self>);
    fn generate_random_projective_points(size: usize) -> Vec<Projective<Self>>;
    fn generate_random_affine_points(size: usize) -> Vec<Affine<Self>>;
    fn scalar_from_montgomery(scalars: &mut [Self::ScalarField]);
    fn scalar_to_montgomery(scalars: &mut [Self::ScalarField]);
    fn affine_from_montgomery(points: &mut [Affine<Self>]);
    fn affine_to_montgomery(points: &mut [Affine<Self>]);
    fn projective_from_montgomery(points: &mut [Projective<Self>]);
    fn projective_to_montgomery(points: &mut [Projective<Self>]);

    #[cfg(feature = "arkworks")]
    type ArkSWConfig: SWCurveConfig;
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Projective<C: CurveConfig> {
    pub x: C::BaseField,
    pub y: C::BaseField,
    pub z: C::BaseField,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Affine<C: CurveConfig> {
    pub x: C::BaseField,
    pub y: C::BaseField,
}

impl<C: CurveConfig> Affine<C> {
    // While this is not a true zero point and not even a valid point, it's still useful
    // both as a handy default as well as a representation of zero points in other codebases
    pub fn zero() -> Self {
        Affine {
            x: C::BaseField::zero(),
            y: C::BaseField::zero(),
        }
    }

    pub fn set_limbs(x: &[u32], y: &[u32]) -> Self {
        Affine {
            x: C::BaseField::set_limbs(x),
            y: C::BaseField::set_limbs(y),
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

impl<C: CurveConfig> From<Affine<C>> for Projective<C> {
    fn from(item: Affine<C>) -> Self {
        Self {
            x: item.x,
            y: item.y,
            z: C::BaseField::one(),
        }
    }
}

impl<C: CurveConfig> Projective<C> {
    pub fn zero() -> Self {
        Projective {
            x: C::BaseField::zero(),
            y: C::BaseField::one(),
            z: C::BaseField::zero(),
        }
    }

    pub fn set_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Projective {
            x: C::BaseField::set_limbs(x),
            y: C::BaseField::set_limbs(y),
            z: C::BaseField::set_limbs(z),
        }
    }
}

impl<C: CurveConfig> PartialEq for Projective<C> {
    fn eq(&self, other: &Self) -> bool {
        C::eq_proj(self as *const _, other as *const _) != 0
    }
}

impl<C: CurveConfig> From<Projective<C>> for Affine<C> {
    fn from(item: Projective<C>) -> Self {
        let mut aff = Self::zero();
        C::to_affine(&item as *const _, &mut aff as *mut _);
        aff
    }
}

#[cfg(feature = "arkworks")]
impl<C: CurveConfig> ArkConvertible for Affine<C>
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
impl<C: CurveConfig> ArkConvertible for Projective<C>
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
        $scalar_field:ident,
        $base_field:ident
    ) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct CurveCfg {}

        pub type G1Affine = Affine<CurveCfg>;
        pub type G1Projective = Projective<CurveCfg>;

        extern "C" {
            fn Eq(point1: *const G1Projective, point2: *const G1Projective) -> c_uint;
            fn ToAffine(point: *const G1Projective, point_out: *mut G1Affine);
            fn GenerateProjectivePoints(points: *mut G1Projective, size: usize);
            fn GenerateAffinePoints(points: *mut G1Affine, size: usize);
            #[link_name = concat!($curve_prefix, "ScalarConvertMontgomery")]
            fn ScalarConvertMontgomery(points: *mut $scalar_field, size: usize, is_into: u8, ctx: *const DeviceContext);
            #[link_name = concat!($curve_prefix, "AffineConvertMontgomery")]
            fn AffineConvertMontgomery(points: *mut G1Affine, size: usize, is_into: u8, ctx: *const DeviceContext);
            #[link_name = concat!($curve_prefix, "ProjectiveConvertMontgomery")]
            fn ProjectiveConvertMontgomery(
                points: *mut G1Projective,
                size: usize,
                is_into: u8,
                ctx: *const DeviceContext,
            );
        }

        impl CurveConfig for CurveCfg {
            type BaseField = $base_field;
            type ScalarField = $scalar_field;

            fn eq_proj(point1: *const G1Projective, point2: *const G1Projective) -> c_uint {
                unsafe { Eq(point1, point2) }
            }

            fn to_affine(point: *const Projective<CurveCfg>, point_out: *mut Affine<CurveCfg>) {
                unsafe { ToAffine(point, point_out) };
            }

            fn generate_random_projective_points(size: usize) -> Vec<G1Projective> {
                let mut res = vec![G1Projective::zero(); size];
                unsafe { GenerateProjectivePoints(&mut res[..] as *mut _ as *mut G1Projective, size) };
                res
            }

            fn generate_random_affine_points(size: usize) -> Vec<G1Affine> {
                let mut res = vec![G1Affine::zero(); size];
                unsafe { GenerateAffinePoints(&mut res[..] as *mut _ as *mut G1Affine, size) };
                res
            }

            fn scalar_from_montgomery(scalars: &mut [$scalar_field]) {
                unsafe {
                    ScalarConvertMontgomery(
                        scalars as *mut _ as *mut $scalar_field,
                        scalars.len(),
                        0,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            fn scalar_to_montgomery(scalars: &mut [$scalar_field]) {
                unsafe {
                    ScalarConvertMontgomery(
                        scalars as *mut _ as *mut $scalar_field,
                        scalars.len(),
                        1,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            fn affine_from_montgomery(points: &mut [G1Affine]) {
                unsafe {
                    AffineConvertMontgomery(
                        points as *mut _ as *mut G1Affine,
                        points.len(),
                        0,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            fn affine_to_montgomery(points: &mut [G1Affine]) {
                unsafe {
                    AffineConvertMontgomery(
                        points as *mut _ as *mut G1Affine,
                        points.len(),
                        1,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            fn projective_from_montgomery(points: &mut [G1Projective]) {
                unsafe {
                    ProjectiveConvertMontgomery(
                        points as *mut _ as *mut G1Projective,
                        points.len(),
                        0,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            fn projective_to_montgomery(points: &mut [G1Projective]) {
                unsafe {
                    ProjectiveConvertMontgomery(
                        points as *mut _ as *mut G1Projective,
                        points.len(),
                        1,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
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
        $curve_config:ident
    ) => {
        #[test]
        fn test_scalar_equality() {
            check_scalar_equality::<<$curve_config as CurveConfig>::ScalarField>()
        }

        #[test]
        fn test_affine_projective_convert() {
            check_affine_projective_convert::<$curve_config>()
        }

        #[test]
        fn test_point_equality() {
            check_point_equality::<$base_limbs, $curve_config>()
        }
    };
}

#[macro_export]
macro_rules! impl_curve_ark_tests {
    (
        $curve_config:ident,
        $ark_affine:ident,
        $scalar_config:ident
    ) => {
        #[test]
        fn test_ark_scalar_convert() {
            let size = 1 << 10;
            let scalars = $scalar_config::generate_random(size);
            for scalar in scalars {
                assert_eq!(scalar.to_ark(), scalar.to_ark())
            }
        }

        #[test]
        fn test_ark_point_convert() {
            let size = 1 << 10;
            let affine_points = $curve_config::generate_random_affine_points(size);
            for affine_point in affine_points {
                let ark_projective = Into::<Projective<$curve_config>>::into(affine_point).to_ark();
                let ark_affine: $ark_affine = ark_projective.into();
                assert!(ark_affine.is_on_curve());
                assert!(ark_affine.is_in_correct_subgroup_assuming_on_curve());
                let affine_after_conversion = Affine::<$curve_config>::from_ark(ark_affine).into();
                assert_eq!(affine_point, affine_after_conversion);
            }
        }
    };
}
