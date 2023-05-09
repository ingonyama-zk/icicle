use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

pub struct Field<const NUM_LIMBS: usize>{
    pub prime: [u32; NUM_LIMBS], 
}


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


#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct ScalarT<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}


impl<const NUM_LIMBS: usize> Default for ScalarT<NUM_LIMBS> {
    fn default() -> Self {
        ScalarT::zero()
    }
}

impl<const NUM_LIMBS: usize> ScalarT<NUM_LIMBS> {
    pub fn zero() -> Self {
        ScalarT {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn from_limbs(value: &[u32]) -> Self {
        Self {
            s: get_fixed_limbs(value),
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        ScalarT { s }
    }

    pub fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }

    pub fn from_limbs_le(value: &[u32]) -> ScalarT<NUM_LIMBS> {
       Self::from_limbs(value)
    }

    pub fn from_limbs_be(value: &[u32]) -> ScalarT<NUM_LIMBS> {
        let mut value = value.to_vec();
        value.reverse();
        Self::from_limbs_le(&value)
    }

    pub fn limbs(&self) -> [u32; NUM_LIMBS] {
        self.s
    }
}