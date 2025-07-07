use icicle_runtime::errors::IcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::ffi::c_void;
use std::ops::{Add, Mul, Sub};

#[doc(hidden)]
pub trait GenerateRandom: Sized {
    fn generate_random(size: usize) -> Vec<Self>;
}

#[macro_export]
macro_rules! impl_generate_random {
    (
        $obj:ident,
        $generate_random_function_name:expr
    ) => {
        impl icicle_core::traits::GenerateRandom for $obj {
            fn generate_random(size: usize) -> Vec<$obj> {
                extern "C" {
                    #[link_name = $generate_random_function_name]
                    pub(crate) fn generate_random_ffi(elements: *mut $obj, size: usize);
                }

                let mut res = vec![$obj::zero(); size];
                unsafe { generate_random_ffi(&mut res[..] as *mut _ as *mut $obj, size) };
                res
            }
        }
    };
}

pub trait MontgomeryConvertible: Sized {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> Result<(), IcicleError>;
    fn from_mont(
        values: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        stream: &IcicleStream,
    ) -> Result<(), IcicleError>;
}

#[macro_export]
macro_rules! impl_montgomery_convertible {
    (
        $obj:ident,
        $convert_montgomery_function_name:expr
    ) => {
        impl $obj {
            fn convert_montgomery(
                values: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                stream: &IcicleStream,
                is_into: bool,
            ) -> Result<(), icicle_runtime::IcicleError> {
                extern "C" {
                    #[link_name = $convert_montgomery_function_name]
                    fn convert_montgomery_ffi(
                        values: *const $obj,
                        size: u64,
                        is_into: bool,
                        config: &icicle_core::vec_ops::VecOpsConfig,
                        output: *mut $obj,
                    ) -> eIcicleError;
                }

                // check device slice is on active device
                if values.is_on_device() && !values.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = icicle_core::vec_ops::VecOpsConfig::default();
                config.is_a_on_device = values.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                unsafe {
                    convert_montgomery_ffi(
                        values.as_ptr(),
                        values.len() as u64,
                        is_into,
                        &config,
                        values.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }

        impl icicle_core::traits::MontgomeryConvertible for $obj {
            fn to_mont(
                values: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                stream: &IcicleStream,
            ) -> Result<(), icicle_runtime::IcicleError> {
                $obj::convert_montgomery(values, stream, true)
            }

            fn from_mont(
                values: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                stream: &IcicleStream,
            ) -> Result<(), icicle_runtime::IcicleError> {
                $obj::convert_montgomery(values, stream, false)
            }
        }
    };
}

pub trait Arithmetic: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn sqr(&self) -> Self;
    fn pow(&self, exp: usize) -> Self;
}

#[macro_export]
macro_rules! impl_arithmetic {
    (
        $obj:ident,
        $obj_prefix:literal
    ) => {
        impl std::ops::Add for $obj {
            type Output = Self;

            fn add(self, second: Self) -> Self {
                extern "C" {
                    #[link_name = concat!($obj_prefix, "_add")]
                    pub(crate) fn add(a: *const $obj, b: *const $obj, result: *mut $obj);
                }

                let mut result = Self::zero();
                unsafe {
                    add(
                        &self as *const $obj,
                        &second as *const $obj,
                        &mut result as *mut $obj,
                    );
                }
                result
            }
        }

        impl std::ops::Sub for $obj {
            type Output = Self;

            fn sub(self, second: Self) -> Self {
                extern "C" {
                    #[link_name = concat!($obj_prefix, "_sub")]
                    pub(crate) fn sub(a: *const $obj, b: *const $obj, result: *mut $obj);
                }

                let mut result = Self::zero();
                unsafe {
                    sub(
                        &self as *const $obj,
                        &second as *const $obj,
                        &mut result as *mut $obj,
                    );
                }
                result
            }
        }

        impl std::ops::Mul for $obj {
            type Output = Self;

            fn mul(self, second: Self) -> Self {
                extern "C" {
                    #[link_name = concat!($obj_prefix, "_mul")]
                    pub(crate) fn mul(a: *const $obj, b: *const $obj, result: *mut $obj);
                }

                let mut result = Self::zero();
                unsafe {
                    mul(
                        &self as *const $obj,
                        &second as *const $obj,
                        &mut result as *mut $obj,
                    );
                }
                result
            }
        }

        impl icicle_core::traits::Arithmetic for $obj {
            fn sqr(&self) -> Self {
                extern "C" {
                    #[link_name = concat!($obj_prefix, "_sqr")]
                    pub(crate) fn sqr(a: *const $obj, result: *mut $obj);
                }

                let mut result = Self::zero();
                unsafe {
                    sqr(self as *const $obj, &mut result as *mut $obj);
                }
                result
            }

            fn pow(&self, exp: usize) -> Self {
                extern "C" {
                    #[link_name = concat!($obj_prefix, "_pow")]
                    pub(crate) fn pow(a: *const $obj, exp: usize, result: *mut $obj);
                }

                let mut result = Self::zero();
                unsafe {
                    pow(self as *const $obj, exp, &mut result as *mut $obj);
                }
                result
            }
        }
    };
}

pub trait Invertible: Sized {
    fn inv(&self) -> Self;
}

#[macro_export]
macro_rules! impl_invertible {
    (
        $obj:ident,
        $obj_prefix:literal
    ) => {
        impl icicle_core::traits::Invertible for $obj {
            fn inv(&self) -> Self {
                extern "C" {
                    #[link_name = concat!($obj_prefix, "_inv")]
                    pub(crate) fn inverse(a: *const $obj, result: *mut $obj);
                }
                let mut result = Self::zero();
                unsafe { inverse(self as *const $obj, &mut result as *mut $obj) };
                result
            }
        }
    };
}

pub trait Handle {
    fn handle(&self) -> *const c_void;
}
