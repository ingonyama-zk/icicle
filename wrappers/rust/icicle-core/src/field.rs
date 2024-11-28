use crate::traits::{FieldConfig, FieldImpl, MontgomeryConvertible, Arithmetic};
use hex::FromHex;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul};

#[derive(PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize, F: FieldConfig> {
    limbs: [u32; NUM_LIMBS],
    p: PhantomData<F>,
}

unsafe impl<const NUM_LIMBS: usize, F: FieldConfig> Send for Field<NUM_LIMBS, F> {}
unsafe impl<const NUM_LIMBS: usize, F: FieldConfig> Sync for Field<NUM_LIMBS, F> {}

impl<const NUM_LIMBS: usize, F: FieldConfig> Display for Field<NUM_LIMBS, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "0x")?;
        for &b in self
            .limbs
            .iter()
            .rev()
        {
            write!(f, "{:08x}", b)?;
        }
        Ok(())
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Debug for Field<NUM_LIMBS, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Into<[u32; NUM_LIMBS]> for Field<NUM_LIMBS, F> {
    fn into(self) -> [u32; NUM_LIMBS] {
        self.limbs
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> From<[u32; NUM_LIMBS]> for Field<NUM_LIMBS, F> {
    fn from(limbs: [u32; NUM_LIMBS]) -> Self {
        Self { limbs, p: PhantomData }
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> FieldImpl for Field<NUM_LIMBS, F> {
    type Config = F;
    type Repr = [u32; NUM_LIMBS];

    fn to_bytes_le(&self) -> Vec<u8> {
        self.limbs
            .iter()
            .map(|limb| {
                limb.to_le_bytes()
                    .to_vec()
            })
            .flatten()
            .collect::<Vec<_>>()
    }

    // please note that this function zero-pads if there are not enough bytes
    // and only takes the first bytes in there are too many of them
    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut limbs: [u32; NUM_LIMBS] = [0; NUM_LIMBS];
        for (i, chunk) in bytes
            .chunks(4)
            .take(NUM_LIMBS)
            .enumerate()
        {
            let mut chunk_array: [u8; 4] = [0; 4];
            chunk_array[..chunk.len()].clone_from_slice(chunk);
            limbs[i] = u32::from_le_bytes(chunk_array);
        }
        Self::from(limbs)
    }

    fn from_hex(s: &str) -> Self {
        let mut bytes = Vec::from_hex(if s.starts_with("0x") { &s[2..] } else { s }).expect("Invalid hex string");
        bytes.reverse();
        Self::from_bytes_le(&bytes)
    }

    fn zero() -> Self {
        FieldImpl::from_u32(0)
    }

    fn one() -> Self {
        FieldImpl::from_u32(1)
    }

    fn from_u32(val: u32) -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[0] = val;
        Field { limbs, p: PhantomData }
    }
}

#[doc(hidden)]
pub trait MontgomeryConvertibleField<F: FieldImpl> {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<F> + ?Sized), stream: &IcicleStream) -> eIcicleError;
    fn from_mont(values: &mut (impl HostOrDeviceSlice<F> + ?Sized), stream: &IcicleStream) -> eIcicleError;
}

#[doc(hidden)]
pub trait FieldArithmetic<F: FieldImpl> {
    fn add(first: F, second: F) -> F;
    fn sub(first: F, second: F) -> F;
    fn mul(first: F, second: F) -> F;
    fn square(first: F) -> F;
    fn inv(first: F) -> F;
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Arithmetic for Field<NUM_LIMBS, F> where F: FieldArithmetic<Self> {
    fn square(self) -> Self {
        F::square(self)
    }

    fn inv(self) -> Self {
        F::inv(self)
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Add for Field<NUM_LIMBS, F>
where
    F: FieldArithmetic<Self>,
{
    type Output = Self;

    fn add(self, second: Self) -> Self {
        F::add(self, second)
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Sub for Field<NUM_LIMBS, F>
where
    F: FieldArithmetic<Self>,
{
    type Output = Self;

    fn sub(self, second: Self) -> Self {
        F::sub(self, second)
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Mul for Field<NUM_LIMBS, F>
where
    F: FieldArithmetic<Self>,
{
    type Output = Self;

    fn mul(self, second: Self) -> Self {
        F::mul(self, second)
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> MontgomeryConvertible for Field<NUM_LIMBS, F>
where
    F: MontgomeryConvertibleField<Self>,
{
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
        F::to_mont(values, stream)
    }

    fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError {
        F::from_mont(values, stream)
    }
}

#[macro_export]
macro_rules! impl_field {
    (
        $num_limbs:ident,
        $field_name:ident,
        $field_cfg:ident
    ) => {
        #[doc(hidden)]
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $field_cfg {}

        impl FieldConfig for $field_cfg {}
        pub type $field_name = Field<$num_limbs, $field_cfg>;
    };
}

#[macro_export]
macro_rules! impl_scalar_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $num_limbs:ident,
        $field_name:ident,
        $field_cfg:ident
    ) => {
        impl_field!($num_limbs, $field_name, $field_cfg);

        mod $field_prefix_ident {
            use super::{$field_name, HostOrDeviceSlice};
            use icicle_core::{vec_ops::VecOpsConfig, traits::FieldImpl};
            use icicle_runtime::errors::eIcicleError;
            use icicle_runtime::stream::{IcicleStream, IcicleStreamHandle};

            extern "C" {
                #[link_name = concat!($field_prefix, "_generate_scalars")]
                pub(crate) fn generate_scalars(scalars: *mut $field_name, size: usize);

                #[link_name = concat!($field_prefix, "_scalar_convert_montgomery")]
                fn _convert_scalars_montgomery(
                    scalars: *const $field_name,
                    size: u64,
                    is_into: bool,
                    config: &VecOpsConfig,
                    output: *mut $field_name,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_add")]
                pub(crate) fn add(
                    a: *const $field_name,
                    b: *const $field_name,
                    result: *mut $field_name,
                );

                #[link_name = concat!($field_prefix, "_sub")]
                pub(crate) fn sub(
                    a: *const $field_name,
                    b: *const $field_name,
                    result: *mut $field_name,
                );

                #[link_name = concat!($field_prefix, "_mul")]
                pub(crate) fn mul(
                    a: *const $field_name,
                    b: *const $field_name,
                    result: *mut $field_name,
                );
            }

            pub(crate) fn convert_scalars_montgomery(
                scalars: *mut $field_name,
                len: usize,
                is_into: bool,
                config: &VecOpsConfig,
            ) -> eIcicleError {
                unsafe { _convert_scalars_montgomery(scalars, len as u64, is_into, &config, scalars) }
            }
        }

        impl icicle_core::field::FieldArithmetic<$field_name> for $field_cfg {
            fn add(
                first: $field_name,
                second: $field_name,
            ) -> $field_name {
                let mut result = $field_name::zero();
                unsafe {
                    $field_prefix_ident::add(
                        &first as *const $field_name,
                        &second as *const $field_name,
                        &mut result as *mut $field_name,
                    );
                }

                result
            }

            fn sub(
                first: $field_name,
                second: $field_name,
            ) -> $field_name {
                let mut result = $field_name::zero();
                unsafe {
                    $field_prefix_ident::sub(
                        &first as *const $field_name,
                        &second as *const $field_name,
                        &mut result as *mut $field_name,
                    );
                }

                result
            }

            fn mul(
                first: $field_name,
                second: $field_name,
            ) -> $field_name {
                let mut result = $field_name::zero();
                unsafe {
                    $field_prefix_ident::mul(
                        &first as *const $field_name,
                        &second as *const $field_name,
                        &mut result as *mut $field_name,
                    );
                }

                result
            }

            fn square(
                first: $field_name,
            ) -> $field_name {
                let mut result = $field_name::zero();
                unsafe {
                    $field_prefix_ident::mul(
                        &first as *const $field_name,
                        &first as *const $field_name,
                        &mut result as *mut $field_name,
                    );
                }

                result
            }

            //TODO: emirsoyturk
            fn inv(
                first: $field_name,
            ) -> $field_name {
                let mut result = $field_name::zero();
                unsafe {
                    $field_prefix_ident::mul(
                        &first as *const $field_name,
                        &first as *const $field_name,
                        &mut result as *mut $field_name,
                    );
                }

                result
            }
        }

        impl GenerateRandom<$field_name> for $field_cfg {
            fn generate_random(size: usize) -> Vec<$field_name> {
                let mut res = vec![$field_name::zero(); size];
                unsafe { $field_prefix_ident::generate_scalars(&mut res[..] as *mut _ as *mut $field_name, size) };
                res
            }
        }

        impl MontgomeryConvertibleField<$field_name> for $field_cfg {
            fn to_mont(
                values: &mut (impl HostOrDeviceSlice<$field_name> + ?Sized),
                stream: &IcicleStream,
            ) -> eIcicleError {
                use icicle_core::vec_ops::VecOpsConfig;
                // check device slice is on active device
                if values.is_on_device() && !values.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = values.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                $field_prefix_ident::convert_scalars_montgomery(
                    unsafe { values.as_mut_ptr() },
                    values.len(),
                    true,
                    &config,
                )
            }

            fn from_mont(
                values: &mut (impl HostOrDeviceSlice<$field_name> + ?Sized),
                stream: &IcicleStream,
            ) -> eIcicleError {
                use icicle_core::vec_ops::VecOpsConfig;
                // check device slice is on active device
                if values.is_on_device() && !values.is_on_active_device() {
                    panic!("input not allocated on the active device");
                }
                let mut config = VecOpsConfig::default();
                config.is_a_on_device = values.is_on_device();
                config.is_async = !stream.is_null();
                config.stream_handle = (&*stream).into();
                $field_prefix_ident::convert_scalars_montgomery(
                    unsafe { values.as_mut_ptr() },
                    values.len(),
                    false,
                    &config,
                )
            }
        }
    };
}

#[macro_export]
macro_rules! impl_field_tests {
    (
        $field_name:ident
    ) => {
        pub mod test_field {
            use super::*;
            use icicle_runtime::test_utilities;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_field_convert_montgomery() {
                initialize();
                check_field_convert_montgomery::<$field_name>()
            }

            #[test]
            fn test_field_equality() {
                initialize();
                check_field_equality::<$field_name>()
            }

            #[test]
            fn test_field_arithmetic() {
                initialize();
                check_field_arithmetic::<$field_name>()
            }
        }
    };
}
