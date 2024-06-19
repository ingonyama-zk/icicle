use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::DeviceSlice;
use std::{
    fmt::{Debug, Display},
    mem::MaybeUninit,
};

#[doc(hidden)]
pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

#[doc(hidden)]
pub trait FieldConfig: Debug + PartialEq + Copy + Clone {}

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

// pub trait MontgomeryConvertible<'a>: Sized {
//     fn to_mont(values: &mut DeviceSlice<Self>, ctx: &DeviceContext<'a>) -> CudaError;
//     fn from_mont(values: &mut DeviceSlice<Self>, ctx: &DeviceContext<'a>) -> CudaError;
// }

// pub trait IcicleResultWrap {
//     fn wrap(self) -> IcicleResult<()>;
//     fn wrap_value<T>(self, value: T) -> IcicleResult<T>;
//     fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> IcicleResult<T>;
// }
