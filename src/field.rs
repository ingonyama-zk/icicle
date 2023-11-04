use std::ffi::c_uint;
use std::marker::PhantomData;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize, P> {
    limbs: [u32; NUM_LIMBS],
    p: PhantomData<P>,
}

extern "C" {
    fn RandomScalars(scalars: *mut ScalarField, size: c_uint);
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

impl<const NUM_LIMBS: usize, P> Field<NUM_LIMBS, P> {
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

pub const SCALAR_LIMBS: usize = 8;

// TODO: `usize` is random junk, needs to be changed to some field-specific type in reality
pub type ScalarField = Field<SCALAR_LIMBS, usize>;
