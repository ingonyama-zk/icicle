#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::traits::{FieldConfig, FieldImpl};
#[cfg(feature = "arkworks")]
use ark_ff::{BigInteger, PrimeField};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

#[derive(PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize, F: FieldConfig> {
    limbs: [u64; NUM_LIMBS],
    p: PhantomData<F>,
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Display for Field<NUM_LIMBS, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "0x")?;
        for &b in self
            .limbs
            .iter()
            .rev()
        {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Debug for Field<NUM_LIMBS, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "0x")?;
        for &b in self
            .limbs
            .iter()
            .rev()
        {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Into<[u64; NUM_LIMBS]> for Field<NUM_LIMBS, F> {
    fn into(self) -> [u64; NUM_LIMBS] {
        self.limbs
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> From<[u64; NUM_LIMBS]> for Field<NUM_LIMBS, F> {
    fn from(limbs: [u64; NUM_LIMBS]) -> Self {
        Self { limbs, p: PhantomData }
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> FieldImpl for Field<NUM_LIMBS, F> {
    type Config = F;
    type Repr = [u64; NUM_LIMBS];

    fn to_bytes_le(&self) -> Vec<u8> {
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
    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut limbs: [u64; NUM_LIMBS] = [0; NUM_LIMBS];
        for (i, chunk) in bytes
            .chunks(8)
            .take(NUM_LIMBS)
            .enumerate()
        {
            let mut chunk_array: [u8; 8] = [0; 8];
            chunk_array[..chunk.len()].clone_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(chunk_array);
        }
        Self::from(limbs)
    }

    fn zero() -> Self {
        Field {
            limbs: [0u64; NUM_LIMBS],
            p: PhantomData,
        }
    }

    fn one() -> Self {
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

#[macro_export]
macro_rules! impl_field {
    (
        $num_limbs:ident,
        $field_name:ident,
        $field_cfg:ident,
        $ark_equiv:ident
    ) => {
        #[doc(hidden)]
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $field_cfg {}

        impl FieldConfig for $field_cfg {
            #[cfg(feature = "arkworks")]
            type ArkField = $ark_equiv;
        }
        pub type $field_name = Field<$num_limbs, $field_cfg>;
    };
}

#[macro_export]
macro_rules! impl_scalar_field {
    (
        $field_prefix:literal,
        $num_limbs:ident,
        $field_name:ident,
        $field_cfg:ident,
        $ark_equiv:ident
    ) => {
        impl_field!($num_limbs, $field_name, $field_cfg, $ark_equiv);

        extern "C" {
            #[link_name = concat!($field_prefix, "GenerateScalars")]
            fn generate_scalars(scalars: *mut $field_name, size: usize);
        }

        impl GenerateRandom<$field_name> for $field_cfg {
            fn generate_random(size: usize) -> Vec<$field_name> {
                let mut res = vec![$field_name::zero(); size];
                unsafe { generate_scalars(&mut res[..] as *mut _ as *mut $field_name, size) };
                res
            }
        }
    };
}
