use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::{
    fmt::{Debug, Display},
    ops::{Mul, Add, Sub},
};

pub trait FieldImpl:
    Default +
    Display +
    Debug +
    PartialEq +
    Copy +
    Clone +
    Into<Self::Repr> +
    From<Self::Repr> +
    Send +
    Sync +
    Mul<Output = Self> +
    Sub<Output = Self> +
    Add<Output = Self>
{
    type Repr;

    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_le(bytes: &[u8]) -> Self;
    fn from_hex(s: &str) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u32(val: u32) -> Self;
    fn sqr(self) -> Self;
    fn inv(self) -> Self;
}

#[doc(hidden)]
pub trait GenerateRandom: Sized {
    fn generate_random(size: usize) -> Vec<Self>;
}

pub trait MontgomeryConvertible: Sized {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError;
    fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError;
}

pub trait ScalarImpl: FieldImpl + GenerateRandom + MontgomeryConvertible {}
