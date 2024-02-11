use crate::error::IcicleResult;
#[cfg(feature = "arkworks")]
use ark_ff::Field as ArkField;
use icicle_cuda_runtime::{error::CudaError, memory::HostOrDeviceSlice};
use std::{fmt::Debug, mem::MaybeUninit};

#[doc(hidden)]
pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

#[doc(hidden)]
pub trait FieldConfig: Debug + PartialEq + Copy + Clone {
    #[cfg(feature = "arkworks")]
    type ArkField: ArkField;
}

pub trait FieldImpl: Debug + PartialEq + Copy + Clone + Into<Self::Repr> + From<Self::Repr> {
    #[doc(hidden)]
    type Config: FieldConfig;
    type Repr;

    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_le(bytes: &[u8]) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

#[cfg(feature = "arkworks")]
pub trait ArkConvertible {
    type ArkEquivalent;

    fn to_ark(&self) -> Self::ArkEquivalent;
    fn from_ark(ark: Self::ArkEquivalent) -> Self;
}

pub trait MontgomeryConvertible: Sized {
    fn to_mont(values: &mut HostOrDeviceSlice<Self>) -> CudaError;
    fn from_mont(values: &mut HostOrDeviceSlice<Self>) -> CudaError;
}

pub trait IcicleResultWrap {
    fn wrap(self) -> IcicleResult<()>;
    fn wrap_value<T>(self, value: T) -> IcicleResult<T>;
    fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> IcicleResult<T>;
}
