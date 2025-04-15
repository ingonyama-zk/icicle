use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::ffi::c_void;
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

#[doc(hidden)]
pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

#[doc(hidden)]
pub trait FieldConfig: Debug + PartialEq + Copy + Clone {
    fn from_u32<const NUM_LIMBS: usize>(val: u32) -> [u32; NUM_LIMBS];
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

pub trait MontgomeryConvertible: Sized {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError;
    fn from_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> eIcicleError;
}

pub trait Arithmetic: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn sqr(self) -> Self;
    fn inv(self) -> Self;
    fn pow(self, exp: usize) -> Self;
}

pub trait Handle {
    fn handle(&self) -> *const c_void;
}

pub trait Serialization where Self: Sized {
    fn get_serialized_size(&self) -> Result<usize, eIcicleError>;
    fn serialize(&self) -> Result<Vec<u8>, eIcicleError>;
    fn deserialize(buffer: &[u8]) -> Result<Self, eIcicleError>;
    fn serialize_to_file(&self, filename: &str) -> Result<(), eIcicleError>;
    fn deserialize_from_file(filename: &str) -> Result<Self, eIcicleError>;
}
