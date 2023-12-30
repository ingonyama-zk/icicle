#[cfg(feature = "arkworks")]
use ark_ff::PrimeField;
use std::fmt::Debug;

#[doc(hidden)]
pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

#[doc(hidden)]
pub trait FieldConfig: Debug + PartialEq + Copy + Clone {
    #[cfg(feature = "arkworks")]
    type ArkField: PrimeField;
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
