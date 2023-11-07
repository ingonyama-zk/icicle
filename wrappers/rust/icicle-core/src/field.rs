use std::marker::PhantomData;

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

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ScalarCfg {}

impl FieldConfig for ScalarCfg {}

const SCALAR_LIMBS: usize = 8;

pub type ScalarField = Field<SCALAR_LIMBS, ScalarCfg>;

extern "C" {
    fn GenerateScalars(scalars: *mut ScalarField, size: usize);
}

pub(crate) fn generate_random_scalars(size: usize) -> Vec<ScalarField> {
    let mut res = vec![ScalarField::zero(); size];
    unsafe { GenerateScalars(&mut res[..] as *mut _ as *mut ScalarField, size) };
    res
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{generate_random_scalars, Field, FieldConfig, ScalarField};
    use crate::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};
    use ark_ff::BigInteger256;
    use std::mem::transmute_copy;

    type BigIntegerArk = BigInteger256;

    impl<const NUM_LIMBS: usize, F: FieldConfig> Field<NUM_LIMBS, F> {
        pub fn to_ark(&self) -> BigIntegerArk {
            BigIntegerArk::new(
                u32_vec_to_u64_vec(&self.get_limbs())
                    .try_into()
                    .unwrap(),
            )
        }

        pub fn from_ark(ark: BigIntegerArk) -> Self {
            Self::set_limbs(&u64_vec_to_u32_vec(&ark.0))
        }

        pub fn to_ark_transmute(&self) -> BigIntegerArk {
            unsafe { transmute_copy(self) }
        }

        pub fn from_ark_transmute(v: BigIntegerArk) -> Self {
            unsafe { transmute_copy(&v) }
        }
    }

    #[test]
    fn test_scalar_equality() {
        let left = ScalarField::zero();
        let right = ScalarField::one();
        assert_ne!(left, right);
        let left = ScalarField::set_limbs(&[1]);
        assert_eq!(left, right);
    }

    #[test]
    fn test_ark_scalar_convert() {
        let size = 1 << 10;
        let scalars = generate_random_scalars(size);
        for scalar in scalars {
            assert_eq!(
                scalar.to_ark(),
                scalar.to_ark_transmute(),
                "{:08X?} {:08X?}",
                scalar.to_ark(),
                scalar.to_ark_transmute()
            )
        }
    }
}
