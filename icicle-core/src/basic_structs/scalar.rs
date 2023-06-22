use std::ffi::{c_int, c_uint};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;
use std::mem::transmute;
use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use rustacuda::memory::{DeviceBox, CopyDestination};

use crate::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};

use std::marker::PhantomData;
use std::convert::TryInto;

use super::field::{Field, self};

pub fn get_fixed_limbs<const NUM_LIMBS: usize>(val: &[u32]) -> [u32; NUM_LIMBS] {
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

pub trait ScalarTrait{
    fn base_limbs() -> usize;
    fn zero() -> Self;
    fn from_limbs(value: &[u32]) -> Self;
    fn one() -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
    fn limbs(&self) -> &[u32];
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct ScalarT<M, const NUM_LIMBS: usize> {
    pub(crate) phantom: PhantomData<M>,
    pub(crate) value : [u32; NUM_LIMBS]
}

impl<M, const NUM_LIMBS: usize> ScalarTrait for ScalarT<M, NUM_LIMBS>
where
    M: Field<NUM_LIMBS>,
{

    fn base_limbs() -> usize {
        return NUM_LIMBS; 
    }

    fn zero() -> Self {
        ScalarT {
            value: [0u32; NUM_LIMBS],
            phantom: PhantomData,
        }
    }

    fn from_limbs(value: &[u32]) -> Self {
        Self {
            value: get_fixed_limbs(value),
            phantom: PhantomData,
        }
    }

    fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        ScalarT { value: s, phantom: PhantomData }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.value
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }

    fn limbs(&self) -> &[u32] {
        &self.value
    }
}

impl<M, const NUM_LIMBS: usize> ScalarT<M, NUM_LIMBS> where M: field::Field<NUM_LIMBS>{
    pub fn from_limbs_le(value: &[u32]) -> ScalarT<M,NUM_LIMBS> {
        Self::from_limbs(value)
     }
 
    pub fn from_limbs_be(value: &[u32]) -> ScalarT<M,NUM_LIMBS> {
         let mut value = value.to_vec();
         value.reverse();
         Self::from_limbs_le(&value)
     }
 
     // Additional Functions
     pub fn add(&self, other:ScalarT<M, NUM_LIMBS>) -> ScalarT<M,NUM_LIMBS>{  // overload + 
         return ScalarT{value: [self.value[0] + other.value[0];NUM_LIMBS], phantom: PhantomData }; 
     }
}