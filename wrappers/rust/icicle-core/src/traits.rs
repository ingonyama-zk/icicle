use std::{fmt::Debug, result::*, mem::MaybeUninit};
use crate::error::IcicleResult;

pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

pub trait FieldImpl: Debug + PartialEq + Copy + Clone {
    fn set_limbs(value: &[u32]) -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_le(bytes: &[u8]) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

pub trait GetLimbs<const NUM_LIMBS: usize> {
    fn get_limbs(&self) -> [u32; NUM_LIMBS];
}

#[cfg(feature = "arkworks")]
pub trait ArkConvertible {
    type ArkEquivalent;

    fn to_ark(&self) -> Self::ArkEquivalent;
    fn from_ark(ark: Self::ArkEquivalent) -> Self;
}

pub trait ResultWrap<T, TError>{
    fn wrap(self) -> Result<T, TError>;
    fn wrap_value(self, value: T) -> Result<T, TError>;
    fn wrap_maybe_uninit(self, value: MaybeUninit<T>) -> Result<T, TError>;
}

pub trait IcicleResultWrap {
    fn wrap(self) -> IcicleResult<()>;
    fn wrap_value<T>(self, value: T) -> IcicleResult<T>;
    fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> IcicleResult<T>;
}