use std::marker::PhantomData;
#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ff::{BigInteger, PrimeField};

#[cfg(feature = "arkworks")]
pub trait FieldConfig: PartialEq + Copy + Clone {
    type ArkField: PrimeField;
}
#[cfg(not(feature = "arkworks"))]
pub trait FieldConfig: PartialEq + Copy + Clone {}

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize, F: FieldConfig> {
    limbs: [u32; NUM_LIMBS],
    p: PhantomData<F>,
}

pub(crate) fn get_fixed_limbs<const NUM_LIMBS: usize>(val: &[u32]) -> [u32; NUM_LIMBS] {
    match val.len() {
        n if n < NUM_LIMBS => {
            let mut padded: [u32; NUM_LIMBS] = [0; NUM_LIMBS];
            padded[..val.len()].copy_from_slice(&val);
            padded
        }
        n if n == NUM_LIMBS => val
            .try_into()
            .unwrap(),
        _ => panic!("slice has too many elements"),
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Field<NUM_LIMBS, F> {
    pub fn get_limbs(&self) -> [u32; NUM_LIMBS] {
        self.limbs
    }

    pub fn set_limbs(value: &[u32]) -> Self {
        Self {
            limbs: get_fixed_limbs(value),
            p: PhantomData,
        }
    }

    pub fn to_bytes_le(&self) -> Vec<u8> {
        self.limbs
            .iter()
            .map(|limb| {
                limb.to_le_bytes()
                    .to_vec()
            })
            .flatten()
            .collect::<Vec<_>>()
    }

    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        let limbs = bytes
            .chunks(4)
            .map(|chunk| {
                u32::from_le_bytes(chunk.try_into().unwrap())
            })
            .collect::<Vec<_>>();
        Self::set_limbs(&limbs)
    }

    pub fn zero() -> Self {
        Field {
            limbs: [0u32; NUM_LIMBS],
            p: PhantomData,
        }
    }

    pub fn one() -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[0] = 1;
        Field { limbs, p: PhantomData }
    }
}

#[cfg(feature = "arkworks")]
impl<const NUM_LIMBS: usize, F: FieldConfig> ArkConvertible for Field<NUM_LIMBS, F> {
    type ArkEquivalent = F::ArkField;

    fn to_ark(&self) -> Self::ArkEquivalent {
        F::ArkField::from_le_bytes_mod_order(&self.to_bytes_le())
    }

    fn from_ark(ark: Self::ArkEquivalent) -> Self {
        let ark_bigint: <Self::ArkEquivalent as PrimeField>::BigInt = ark.into();
        Self::from_bytes_le(&ark_bigint.to_bytes_le())
    }
}
