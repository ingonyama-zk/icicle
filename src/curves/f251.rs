use std::ffi::c_uint;
use ark_ff::{BigInteger256, PrimeField};
use std::mem::transmute;
use ark_ff::Field;
use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};
use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field_F251<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for Field_F251<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> Default for Field_F251<NUM_LIMBS> {
    fn default() -> Self {
        Field_F251::zero()
    }
}

impl<const NUM_LIMBS: usize> Field_F251<NUM_LIMBS> {
    pub fn zero() -> Self {
        Field_F251 {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Field_F251 { s }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

pub const BASE_LIMBS_F251: usize = 8;
pub const SCALAR_LIMBS_F251: usize = 8;

pub type BaseField_F251 = Field_F251<BASE_LIMBS_F251>;
pub type ScalarField_F251 = Field_F251<SCALAR_LIMBS_F251>;

fn get_fixed_limbs<const NUM_LIMBS: usize>(val: &[u32]) -> [u32; NUM_LIMBS] {
    match val.len() {
        n if n < NUM_LIMBS => {
            let mut padded: [u32; NUM_LIMBS] = [0; NUM_LIMBS];
            padded[..val.len()].copy_from_slice(&val);
            padded
        }
        n if n == NUM_LIMBS => val.try_into().unwrap(),
        _ => panic!("slice has too many elements"),
    }
}

impl ScalarField_F251 {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS_F251] {
        self.s
    }

    pub fn to_ark(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger256) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }

    pub fn to_ark_transmute(&self) -> BigInteger256 {
        unsafe { transmute(*self) }
    }

    pub fn from_ark_transmute(v: BigInteger256) -> ScalarField_F251 {
        unsafe { transmute(v) }
    }
}

impl ScalarField_F251 {
    pub fn from_limbs(value: &[u32]) -> ScalarField_F251 {
        ScalarField_F251 {
            s: get_fixed_limbs(value),
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, curves::f251::ScalarField_F251};

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField_F251::from_limbs(&limbs);
        assert_eq!(
            scalar.to_ark(),
            scalar.to_ark_transmute(),
            "{:08X?} {:08X?}",
            scalar.to_ark(),
            scalar.to_ark_transmute()
        )
    }
}