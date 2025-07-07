use icicle_runtime::errors::IcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStream;
use std::ffi::c_void;
use std::ops::{Add, Mul, Sub};

#[doc(hidden)]
pub trait GenerateRandom: Sized {
    fn generate_random(size: usize) -> Vec<Self>;
}

pub trait MontgomeryConvertible: Sized {
    fn to_mont(values: &mut (impl HostOrDeviceSlice<Self> + ?Sized), stream: &IcicleStream) -> Result<(), IcicleError>;
    fn from_mont(
        values: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        stream: &IcicleStream,
    ) -> Result<(), IcicleError>;
}

pub trait Arithmetic: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn sqr(&self) -> Self;
    fn inv(&self) -> Self;
    fn pow(&self, exp: usize) -> Self;
}

pub trait Handle {
    fn handle(&self) -> *const c_void;
}
