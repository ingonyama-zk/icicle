use std::fmt::Debug;

pub trait GenerateRandom<F> {
    fn generate_random(size: usize) -> Vec<F>;
}

pub trait FieldImpl: Debug + PartialEq + Copy + Clone + Into<Self::Repr> + From<Self::Repr> {
    #[doc(hidden)]
    type Config;
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
