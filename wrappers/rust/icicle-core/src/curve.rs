#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::traits::{FieldImpl, MontgomeryConvertible};
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective, SWCurveConfig};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use std::fmt::Debug;

pub trait Curve: Debug + PartialEq + Copy + Clone {
    type BaseField: FieldImpl;
    type ScalarField: FieldImpl;

    #[doc(hidden)]
    fn eq_proj(point1: *const Projective<Self>, point2: *const Projective<Self>) -> bool;
    #[doc(hidden)]
    fn to_affine(point: *const Projective<Self>, point_aff: *mut Affine<Self>);
    #[doc(hidden)]
    fn generate_random_projective_points(size: usize) -> Vec<Projective<Self>>;
    #[doc(hidden)]
    fn generate_random_affine_points(size: usize) -> Vec<Affine<Self>>;
    #[doc(hidden)]
    fn convert_affine_montgomery(points: &mut HostOrDeviceSlice<Affine<Self>>, is_into: bool) -> CudaError;
    #[doc(hidden)]
    fn convert_projective_montgomery(points: &mut HostOrDeviceSlice<Projective<Self>>, is_into: bool) -> CudaError;

    #[cfg(feature = "arkworks")]
    type ArkSWConfig: SWCurveConfig;
}

/// A [projective](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) elliptic curve point.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Projective<C: Curve> {
    pub x: C::BaseField,
    pub y: C::BaseField,
    pub z: C::BaseField,
}

/// An [affine](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) elliptic curve point.
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
        C::eq_proj(self as *const _, other as *const _)
    }
}

impl<C: Curve> From<Projective<C>> for Affine<C> {
    fn from(proj: Projective<C>) -> Self {
        let mut aff = Self::zero();
        C::to_affine(&proj as *const _, &mut aff as *mut _);
        aff
    }
}

impl<C: Curve> MontgomeryConvertible for Affine<C> {
    fn to_mont(values: &mut HostOrDeviceSlice<Self>) -> CudaError {
        C::convert_affine_montgomery(values, true)
    }

    fn from_mont(values: &mut HostOrDeviceSlice<Self>) -> CudaError {
        C::convert_affine_montgomery(values, false)
    }
}

impl<C: Curve> MontgomeryConvertible for Projective<C> {
    fn to_mont(values: &mut HostOrDeviceSlice<Self>) -> CudaError {
        C::convert_projective_montgomery(values, true)
    }

    fn from_mont(values: &mut HostOrDeviceSlice<Self>) -> CudaError {
        C::convert_projective_montgomery(values, false)
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
        $curve_prefix_ident:ident,
        $curve:ident,
        $scalar_field:ident,
        $base_field:ident,
        $ark_config:ident,
        $affine_type:ident,
        $projective_type:ident
    ) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $curve {}

        pub type $affine_type = Affine<$curve>;
        pub type $projective_type = Projective<$curve>;

        mod $curve_prefix_ident {
            use super::{$affine_type, $projective_type, CudaError, DeviceContext};

            extern "C" {
                #[link_name = concat!($curve_prefix, "Eq")]
                pub(crate) fn eq(point1: *const $projective_type, point2: *const $projective_type) -> bool;
                #[link_name = concat!($curve_prefix, "ToAffine")]
                pub(crate) fn proj_to_affine(point: *const $projective_type, point_out: *mut $affine_type);
                #[link_name = concat!($curve_prefix, "GenerateProjectivePoints")]
                pub(crate) fn generate_projective_points(points: *mut $projective_type, size: usize);
                #[link_name = concat!($curve_prefix, "GenerateAffinePoints")]
                pub(crate) fn generate_affine_points(points: *mut $affine_type, size: usize);
                #[link_name = concat!($curve_prefix, "AffineConvertMontgomery")]
                pub(crate) fn _convert_affine_montgomery(
                    points: *mut $affine_type,
                    size: usize,
                    is_into: bool,
                    ctx: *const DeviceContext,
                ) -> CudaError;
                #[link_name = concat!($curve_prefix, "ProjectiveConvertMontgomery")]
                pub(crate) fn _convert_projective_montgomery(
                    points: *mut $projective_type,
                    size: usize,
                    is_into: bool,
                    ctx: *const DeviceContext,
                ) -> CudaError;
            }
        }

        impl Curve for $curve {
            type BaseField = $base_field;
            type ScalarField = $scalar_field;

            fn eq_proj(point1: *const $projective_type, point2: *const $projective_type) -> bool {
                unsafe { $curve_prefix_ident::eq(point1, point2) }
            }

            fn to_affine(point: *const $projective_type, point_out: *mut $affine_type) {
                unsafe { $curve_prefix_ident::proj_to_affine(point, point_out) };
            }

            fn generate_random_projective_points(size: usize) -> Vec<$projective_type> {
                let mut res = vec![$projective_type::zero(); size];
                unsafe {
                    $curve_prefix_ident::generate_projective_points(
                        &mut res[..] as *mut _ as *mut $projective_type,
                        size,
                    )
                };
                res
            }

            fn generate_random_affine_points(size: usize) -> Vec<$affine_type> {
                let mut res = vec![$affine_type::zero(); size];
                unsafe {
                    $curve_prefix_ident::generate_affine_points(&mut res[..] as *mut _ as *mut $affine_type, size)
                };
                res
            }

            fn convert_affine_montgomery(points: &mut HostOrDeviceSlice<$affine_type>, is_into: bool) -> CudaError {
                unsafe {
                    $curve_prefix_ident::_convert_affine_montgomery(
                        points.as_mut_ptr(),
                        points.len(),
                        is_into,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            fn convert_projective_montgomery(
                points: &mut HostOrDeviceSlice<$projective_type>,
                is_into: bool,
            ) -> CudaError {
                unsafe {
                    $curve_prefix_ident::_convert_projective_montgomery(
                        points.as_mut_ptr(),
                        points.len(),
                        is_into,
                        &get_default_device_context() as *const _ as *const DeviceContext,
                    )
                }
            }

            #[cfg(feature = "arkworks")]
            type ArkSWConfig = $ark_config;
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

        #[test]
        fn test_points_convert_montgomery() {
            check_points_convert_montgomery::<$curve>()
        }
    };
}
