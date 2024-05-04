#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::traits::{FieldConfig, FieldImpl, MontgomeryConvertible};
#[cfg(feature = "arkworks")]
use ark_ff::{BigInteger, Field as ArkField, PrimeField};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::DeviceSlice;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use hex::FromHex;

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
        let mut bytes = Vec::from_hex(&s[2..]).expect("Invalid hex string");
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
pub trait MontgomeryConvertibleField<'a, F: FieldImpl> {
    fn to_mont(values: &mut DeviceSlice<F>, ctx: &DeviceContext<'a>) -> CudaError;
    fn from_mont(values: &mut DeviceSlice<F>, ctx: &DeviceContext<'a>) -> CudaError;
}

impl<'a, const NUM_LIMBS: usize, F: FieldConfig> MontgomeryConvertible<'a> for Field<NUM_LIMBS, F>
where
    F: MontgomeryConvertibleField<'a, Self>,
{
    fn to_mont(values: &mut DeviceSlice<Self>, ctx: &DeviceContext<'a>) -> CudaError {
        F::to_mont(values, ctx)
    }

    fn from_mont(values: &mut DeviceSlice<Self>, ctx: &DeviceContext<'a>) -> CudaError {
        F::from_mont(values, ctx)
    }
}

#[cfg(feature = "arkworks")]
impl<const NUM_LIMBS: usize, F: FieldConfig> ArkConvertible for Field<NUM_LIMBS, F> {
    type ArkEquivalent = F::ArkField;

    fn to_ark(&self) -> Self::ArkEquivalent {
        F::ArkField::from_random_bytes(&self.to_bytes_le()).unwrap()
    }

    fn from_ark(ark: Self::ArkEquivalent) -> Self {
        let ark_bytes: Vec<u8> = ark
            .to_base_prime_field_elements()
            .map(|x| {
                x.into_bigint()
                    .to_bytes_le()
            })
            .flatten()
            .collect();
        Self::from_bytes_le(&ark_bytes)
    }
}

#[macro_export]
macro_rules! impl_field {
    (
        $num_limbs:ident,
        $field_name:ident,
        $field_cfg:ident,
        $ark_equiv:ident
    ) => {
        #[doc(hidden)]
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $field_cfg {}

        impl FieldConfig for $field_cfg {
            #[cfg(feature = "arkworks")]
            type ArkField = $ark_equiv;
        }
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
        $field_cfg:ident,
        $ark_equiv:ident
    ) => {
        impl_field!($num_limbs, $field_name, $field_cfg, $ark_equiv);

        mod $field_prefix_ident {
            use super::{$field_name, CudaError, DeviceContext, HostOrDeviceSlice};

            extern "C" {
                #[link_name = concat!($field_prefix, "_generate_scalars")]
                pub(crate) fn generate_scalars(scalars: *mut $field_name, size: usize);

                #[link_name = concat!($field_prefix, "_scalar_convert_montgomery")]
                fn _convert_scalars_montgomery(
                    scalars: *mut $field_name,
                    size: usize,
                    is_into: bool,
                    ctx: *const DeviceContext,
                ) -> CudaError;
            }

            pub(crate) fn convert_scalars_montgomery(
                scalars: *mut $field_name,
                len: usize,
                is_into: bool,
                ctx: &DeviceContext,
            ) -> CudaError {
                unsafe { _convert_scalars_montgomery(scalars, len, is_into, ctx as *const DeviceContext) }
            }
        }

        impl GenerateRandom<$field_name> for $field_cfg {
            fn generate_random(size: usize) -> Vec<$field_name> {
                let mut res = vec![$field_name::zero(); size];
                unsafe { $field_prefix_ident::generate_scalars(&mut res[..] as *mut _ as *mut $field_name, size) };
                res
            }
        }

        impl<'a> MontgomeryConvertibleField<'a, $field_name> for $field_cfg {
            fn to_mont(values: &mut DeviceSlice<$field_name>, ctx: &DeviceContext<'a>) -> CudaError {
                check_device(ctx.device_id);
                assert_eq!(
                    values
                        .device_id()
                        .unwrap(),
                    ctx.device_id,
                    "Device ids are different in slice and context"
                );
                $field_prefix_ident::convert_scalars_montgomery(unsafe { values.as_mut_ptr() }, values.len(), true, ctx)
            }

            fn from_mont(values: &mut DeviceSlice<$field_name>, ctx: &DeviceContext<'a>) -> CudaError {
                check_device(ctx.device_id);
                assert_eq!(
                    values
                        .device_id()
                        .unwrap(),
                    ctx.device_id,
                    "Device ids are different in slice and context"
                );
                $field_prefix_ident::convert_scalars_montgomery(
                    unsafe { values.as_mut_ptr() },
                    values.len(),
                    false,
                    ctx,
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
        #[test]
        fn test_field_convert_montgomery() {
            check_field_convert_montgomery::<$field_name>()
        }

        #[test]
        fn test_field_equality() {
            check_field_equality::<$field_name>()
        }
    };
}
