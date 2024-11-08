use crate::traits::{FieldImpl, MontgomeryConvertible};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};
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
    fn convert_affine_montgomery(
        points: *mut Affine<Self>,
        len: usize,
        is_into: bool,
        stream: &IcicleStream,
    ) -> eIcicleError;
    #[doc(hidden)]
    fn convert_projective_montgomery(
        points: *mut Projective<Self>,
        len: usize,
        is_into: bool,
        stream: &IcicleStream,
    ) -> eIcicleError;

    #[doc(hidden)]
    fn add(
        point1: Projective<Self>,
        point2: Projective<Self>,
    ) -> Projective<Self>;
    #[doc(hidden)]
    fn sub(
        point1: Projective<Self>,
        point2: Projective<Self>,
    ) -> Projective<Self>;
    #[doc(hidden)]
    fn mul_scalar(
        point1: Projective<Self>,
        scalar: Self::ScalarField,
    ) -> Projective<Self>;
    #[doc(hidden)]
    fn mul_two_scalar(
        scalar1: Self::ScalarField,
        scalar2: Self::ScalarField,
    ) -> Self::ScalarField;
    #[doc(hidden)]
    fn add_two_scalar(
        scalar1: Self::ScalarField,
        scalar2: Self::ScalarField,
    ) -> Self::ScalarField;
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
        if *self == (Affine::<C>::zero()) {
            return Projective::<C>::zero();
        }

        Projective {
            x: self.x,
            y: self.y,
            z: C::BaseField::one(),
        }
    }
}

impl<C: Curve> From<Affine<C>> for Projective<C> {
    fn from(item: Affine<C>) -> Self {
        if item == (Affine::<C>::zero()) {
            return Self::zero();
        }
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
        C::eq_proj(self as *const Self, other as *const Self)
    }
}

impl<C: Curve> From<Projective<C>> for Affine<C> {
    fn from(proj: Projective<C>) -> Self {
        let mut aff = Self::zero();
        C::to_affine(&proj as *const Projective<C>, &mut aff as *mut Self);
        aff
    }
}

impl<C: Curve> MontgomeryConvertible for Affine<C> {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
        if !values.is_on_active_device() {
            panic!("values not allocated on an inactive device");
        }
        C::convert_affine_montgomery(unsafe { values.as_mut_ptr() }, values.len(), true, stream)
    }

    fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
        if !values.is_on_active_device() {
            panic!("values not allocated on an inactive device");
        }
        C::convert_affine_montgomery(unsafe { values.as_mut_ptr() }, values.len(), false, stream)
    }
}

impl<C: Curve> MontgomeryConvertible for Projective<C> {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
        if !values.is_on_active_device() {
            panic!("values not allocated on an inactive device");
        }
        C::convert_projective_montgomery(unsafe { values.as_mut_ptr() }, values.len(), true, stream)
    }

    fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
        if !values.is_on_active_device() {
            panic!("values not allocated on an inactive device");
        }
        C::convert_projective_montgomery(unsafe { values.as_mut_ptr() }, values.len(), false, stream)
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
        $affine_type:ident,
        $projective_type:ident
    ) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $curve {}

        pub type $affine_type = Affine<$curve>;
        pub type $projective_type = Projective<$curve>;

        mod $curve_prefix_ident {
            use super::{eIcicleError, $affine_type, $projective_type, IcicleStream, VecOpsConfig};

            extern "C" {
                #[link_name = concat!($curve_prefix, "_eq")]
                pub(crate) fn eq(point1: *const $projective_type, point2: *const $projective_type) -> bool;
                #[link_name = concat!($curve_prefix, "_to_affine")]
                pub(crate) fn proj_to_affine(point: *const $projective_type, point_out: *mut $affine_type);
                #[link_name = concat!($curve_prefix, "_generate_projective_points")]
                pub(crate) fn generate_projective_points(points: *mut $projective_type, size: usize);
                #[link_name = concat!($curve_prefix, "_generate_affine_points")]
                pub(crate) fn generate_affine_points(points: *mut $affine_type, size: usize);
                #[link_name = concat!($curve_prefix, "_affine_convert_montgomery")]
                #[link_name = concat!($curve_prefix, "_add")]
                pub(crate) fn add(
                    point1: *const $projective_type,
                    point2: *const $projective_type, 
                    result: *mut $projective_type,
                );
                #[link_name = concat!($curve_prefix, "_sub")]
                pub(crate) fn sub(
                    point1: *const $projective_type,
                    point2: *const $projective_type, 
                    result: *mut $projective_type,
                );
                #[link_name = concat!($curve_prefix, "_mul_scalar")]
                pub(crate) fn mul_scalar(
                    point1: *const $projective_type,
                    scalar: *const $scalar_field, 
                    result: *mut $projective_type,
                );
                #[link_name = concat!($curve_prefix, "_mul_two_scalar")]
                pub(crate) fn mul_two_scalar(
                    scalar1: *const $scalar_field,
                    scalar2: *const $scalar_field, 
                    result: *mut $scalar_field,
                );
                #[link_name = concat!($curve_prefix, "_add_two_scalar")]
                pub(crate) fn add_two_scalar(
                    scalar1: *const $scalar_field,
                    scalar2: *const $scalar_field, 
                    result: *mut $scalar_field,
                );
                pub(crate) fn _convert_affine_montgomery(
                    input: *const $affine_type,
                    size: usize,
                    is_into: bool,
                    config: &VecOpsConfig,
                    output: *mut $affine_type,
                ) -> eIcicleError;
                #[link_name = concat!($curve_prefix, "_projective_convert_montgomery")]
                pub(crate) fn _convert_projective_montgomery(
                    input: *const $projective_type,
                    size: usize,
                    is_into: bool,
                    config: &VecOpsConfig,
                    output: *mut $projective_type,
                ) -> eIcicleError;
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

            fn add(point1: $projective_type, point2: $projective_type) -> $projective_type {
                let mut result = $projective_type::zero();

                unsafe {
                    $curve_prefix_ident::add(
                        &point1 as *const $projective_type,
                        &point2 as *const $projective_type,
                        &mut result as *mut _ as *mut $projective_type
                    );
                };

                result
            }

            fn sub(point1: $projective_type, point2: $projective_type) -> $projective_type {
                let mut result = $projective_type::zero();

                unsafe {
                    $curve_prefix_ident::sub(
                        &point1 as *const $projective_type,
                        &point2 as *const $projective_type,
                        &mut result as *mut _ as *mut $projective_type
                    );
                };

                result
            }

            fn mul_scalar(point1: $projective_type, scalar: $scalar_field) -> $projective_type {
                let mut result = $projective_type::zero();

                unsafe {
                    $curve_prefix_ident::mul_scalar(
                        &point1 as *const $projective_type,
                        &scalar as *const $scalar_field,
                        &mut result as *mut _ as *mut $projective_type
                    );
                };

                result
            }

            fn mul_two_scalar(scalar1: $scalar_field, scalar2: $scalar_field) -> $scalar_field {
                let mut result = $scalar_field::zero();

                unsafe {
                    $curve_prefix_ident::mul_two_scalar(
                        &scalar1 as *const $scalar_field,
                        &scalar2 as *const $scalar_field,
                        &mut result as *mut _ as *mut $scalar_field
                    );
                };

                result
            }

            fn add_two_scalar(scalar1: $scalar_field, scalar2: $scalar_field) -> $scalar_field {
                let mut result = $scalar_field::zero();

                unsafe {
                    $curve_prefix_ident::add_two_scalar(
                        &scalar1 as *const $scalar_field,
                        &scalar2 as *const $scalar_field,
                        &mut result as *mut _ as *mut $scalar_field
                    );
                };

                result
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

            fn convert_affine_montgomery(
                points: *mut $affine_type,
                len: usize,
                is_into: bool,
                stream: &IcicleStream,
            ) -> eIcicleError {
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = true;
                config.is_result_on_device = true;
                config.is_async = false;
                config.stream_handle = (&*stream).into();
                unsafe { $curve_prefix_ident::_convert_affine_montgomery(points, len, is_into, &config, points) }
            }

            fn convert_projective_montgomery(
                points: *mut $projective_type,
                len: usize,
                is_into: bool,
                stream: &IcicleStream,
            ) -> eIcicleError {
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = true;
                config.is_result_on_device = true;
                config.is_async = false;
                config.stream_handle = (&*stream).into();
                unsafe { $curve_prefix_ident::_convert_projective_montgomery(points, len, is_into, &config, points) }
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
                check_point_equality::<$base_limbs, <<$curve as Curve>::BaseField as FieldImpl>::Config, $curve>()
            }

            #[test]
            fn test_points_convert_montgomery() {
                initialize();
                check_points_convert_montgomery::<$curve>()
            }
        }
    };
}
