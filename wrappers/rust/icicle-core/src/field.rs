#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ff::{BigInteger, PrimeField};
use std::marker::PhantomData;

#[cfg(feature = "arkworks")]
pub trait FieldConfig: PartialEq + Copy + Clone {
    type ArkField: PrimeField;
}
#[cfg(not(feature = "arkworks"))]
pub trait FieldConfig: PartialEq + Copy + Clone {}

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize, F: FieldConfig> {
    limbs: [u64; NUM_LIMBS],
    p: PhantomData<F>,
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Field<NUM_LIMBS, F> {
    pub fn get_limbs(&self) -> [u64; NUM_LIMBS] {
        self.limbs
    }

    pub fn from_limbs(limbs: [u64; NUM_LIMBS]) -> Self {
        Self {
            limbs,
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

    // please note that this function zero-pads if there are not enough bytes
    // and only takes the first bytes in there are too many of them
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut limbs: [u64; NUM_LIMBS] = [0; NUM_LIMBS];
        for (i, chunk) in bytes.chunks(8).take(NUM_LIMBS).enumerate() {
            let mut chunk_array: [u8; 8] = [0; 8];
            chunk_array[..chunk.len()].clone_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(chunk_array);
        }
        Self::from_limbs(limbs)
    }

    pub fn zero() -> Self {
        Field {
            limbs: [0u64; NUM_LIMBS],
            p: PhantomData,
        }
    }

    pub fn one() -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
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
