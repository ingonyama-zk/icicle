use crate::error::IcicleResult;
#[cfg(feature = "arkworks")]
use ark_ff::Field as ArkField;
use icicle_cuda_runtime::{device_context::DeviceContext, error::CudaError, memory::DeviceSlice};
use std::{
    fmt::{Debug, Display},
    mem::MaybeUninit,
};

#[doc(hidden)]
pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

#[doc(hidden)]
pub trait FieldConfig: Debug + PartialEq + Copy + Clone {
    #[cfg(feature = "arkworks")]
    type ArkField: ArkField;
}

pub trait FieldImpl:
    Display + Debug + PartialEq + Copy + Clone + Into<Self::Repr> + From<Self::Repr> + Send + Sync
{
    #[doc(hidden)]
    type Config: FieldConfig;
    type Repr;

    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_le(bytes: &[u8]) -> Self;
    fn from_hex(s: &str) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u32(val: u32) -> Self;
}

#[cfg(feature = "arkworks")]
pub trait ArkConvertible {
    type ArkEquivalent;

    fn to_ark(&self) -> Self::ArkEquivalent;
    fn from_ark(ark: Self::ArkEquivalent) -> Self;
}

pub trait MontgomeryConvertible<'a>: Sized {
    fn to_mont(values: &mut DeviceSlice<Self>, ctx: &DeviceContext<'a>) -> CudaError;
    fn from_mont(values: &mut DeviceSlice<Self>, ctx: &DeviceContext<'a>) -> CudaError;
}

pub trait IcicleResultWrap {
    fn wrap(self) -> IcicleResult<()>;
    fn wrap_value<T>(self, value: T) -> IcicleResult<T>;
    fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> IcicleResult<T>;
}
