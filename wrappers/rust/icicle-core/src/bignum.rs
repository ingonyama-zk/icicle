use std::fmt::{Debug, Display};

pub trait BigNum:
    Display + Debug + Default + Copy + PartialEq + Clone + Into<Self::Limbs> + From<Self::Limbs> + From<u32> + Send + Sync
{
    const LIMBS_SIZE: usize;
    type Limbs: AsRef<[u32]> + AsMut<[u32]> + Copy;

    fn limbs(&self) -> &Self::Limbs;
    fn limbs_mut(&mut self) -> &mut Self::Limbs;

    fn to_bytes_le(&self) -> Vec<u8> {
        self.limbs()
            .as_ref()
            .iter()
            .flat_map(|&x| {
                x.to_le_bytes()
                    .to_vec()
            })
            .collect()
    }

    // please note that this function zero-pads if there are not enough bytes
    // and only takes the first bytes in there are too many of them
    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut result = Self::zero();
        let limbs = result
            .limbs_mut()
            .as_mut();
        for (i, chunk) in bytes
            .chunks(4)
            .take(Self::LIMBS_SIZE)
            .enumerate()
        {
            let mut chunk_array: [u8; 4] = [0; 4];
            chunk_array[..chunk.len()].clone_from_slice(chunk);
            limbs[i] = u32::from_le_bytes(chunk_array);
        }
        result
    }

    fn from_u32(val: u32) -> Self {
        Self::from(val)
    }

    fn zero() -> Self {
        Self::from(0)
    }

    fn one() -> Self {
        Self::from(1)
    }

    fn from_hex(s: &str) -> Self {
        let s = if s.starts_with("0x") { &s[2..] } else { s };
        let mut bytes = hex::decode(s).expect("Invalid hex string");
        bytes.reverse();
        Self::from_bytes_le(&bytes)
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "0x")?;
        for &b in self
            .limbs()
            .as_ref()
            .iter()
            .rev()
        {
            write!(f, "{:08x}", b)?;
        }
        Ok(())
    }

    fn debug_fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[macro_export]
macro_rules! impl_bignum {
    (
        $bignum:ident,
        $bignum_prefix:literal,
        $num_limbs:ident,
        $use_ffi_for_eq:expr,
        $use_ffi_for_from_u32:expr
    ) => {
        #[derive(Copy, Clone)]
        #[repr(C)]
        pub struct $bignum {
            limbs: [u32; $num_limbs],
        }

        impl Default for $bignum {
            fn default() -> Self {
                Self {
                    limbs: [0u32; $num_limbs],
                }
            }
        }

        impl Into<[u32; $num_limbs]> for $bignum {
            fn into(self) -> [u32; $num_limbs] {
                self.limbs
            }
        }

        impl From<[u32; $num_limbs]> for $bignum {
            fn from(limbs: [u32; $num_limbs]) -> Self {
                Self { limbs }
            }
        }

        impl icicle_core::bignum::BigNum for $bignum {
            const LIMBS_SIZE: usize = $num_limbs;
            type Limbs = [u32; $num_limbs];

            fn limbs(&self) -> &Self::Limbs {
                &self.limbs
            }

            fn limbs_mut(&mut self) -> &mut Self::Limbs {
                &mut self.limbs
            }
        }

        impl std::fmt::Display for $bignum {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                <Self as icicle_core::bignum::BigNum>::fmt(self, f)
            }
        }

        impl std::fmt::Debug for $bignum {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                <Self as icicle_core::bignum::BigNum>::debug_fmt(self, f)
            }
        }

        impl PartialEq for $bignum {
            fn eq(&self, other: &Self) -> bool {
                if $use_ffi_for_eq {
                    extern "C" {
                        #[link_name = concat!($bignum_prefix, "_eq")]
                        pub(crate) fn _eq(left: *const $bignum, right: *const $bignum, result: *mut bool);
                    }
                    let mut result = false;
                    unsafe {
                        _eq(
                            self.limbs
                                .as_ptr() as *const $bignum,
                            other
                                .limbs
                                .as_ptr() as *const $bignum,
                            &mut result,
                        );
                    }
                    result
                } else {
                    self.limbs == other.limbs
                }
            }
        }

        impl From<u32> for $bignum {
            fn from(val: u32) -> Self {
                if $use_ffi_for_from_u32 {
                    extern "C" {
                        #[link_name = concat!($bignum_prefix, "_from_u32")]
                        pub(crate) fn from_u32(val: u32, result: *mut $bignum);
                    }

                    let mut limbs = [0u32; $num_limbs];

                    unsafe {
                        from_u32(val, limbs.as_mut_ptr() as *mut Self);
                    }

                    Self { limbs }
                } else {
                    let mut limbs = [0u32; $num_limbs];
                    limbs[0] = val;
                    Self { limbs }
                }
            }
        }
    };
}
