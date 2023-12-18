use std::fmt::Debug;

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
