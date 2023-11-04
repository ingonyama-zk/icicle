use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;
use std::ffi::c_uint;
use std::marker::PhantomData;

use crate::field::*;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Projective<const NUM_LIMBS: usize, F: FieldConfig> {
    pub x: Field<NUM_LIMBS, F>,
    pub y: Field<NUM_LIMBS, F>,
    pub z: Field<NUM_LIMBS, F>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Affine<const NUM_LIMBS: usize, F: FieldConfig> {
    pub x: Field<NUM_LIMBS, F>,
    pub y: Field<NUM_LIMBS, F>,
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Projective<NUM_LIMBS, F> {
    pub fn zero() -> Self {
        Projective {
            x: Field::<NUM_LIMBS, F>::zero(),
            y: Field::<NUM_LIMBS, F>::one(),
            z: Field::<NUM_LIMBS, F>::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Affine<NUM_LIMBS, F> {
    pub fn set_limbs(x: &[u32], y: &[u32]) -> Self {
        Affine {
            x: Field::<NUM_LIMBS, F>::set_limbs(x),
            y: Field::<NUM_LIMBS, F>::set_limbs(y),
        }
    }

    pub fn get_limbs(&self) -> Vec<u32> {
        [
            self.x
                .get_limbs(),
            self.y
                .get_limbs(),
        ]
        .concat()
    }

    pub fn to_projective(&self) -> Projective<NUM_LIMBS, F> {
        Projective {
            x: self.x,
            y: self.y,
            z: Field::<NUM_LIMBS, F>::one(),
        }
    }
}

impl<const NUM_LIMBS: usize, F: FieldConfig> Projective<NUM_LIMBS, F> {
    pub fn set_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Projective {
            x: Field::<NUM_LIMBS, F>::set_limbs(x),
            y: Field::<NUM_LIMBS, F>::set_limbs(y),
            z: Field::<NUM_LIMBS, F>::set_limbs(z),
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Self {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS, "length must be 3 * {}", BASE_LIMBS);
        Projective {
            x: Field::<NUM_LIMBS, F>::set_limbs(&value[..BASE_LIMBS]),
            y: Field::<NUM_LIMBS, F>::set_limbs(&value[BASE_LIMBS..BASE_LIMBS * 2]),
            z: Field::<NUM_LIMBS, F>::set_limbs(&value[BASE_LIMBS * 2..]),
        }
    }

    // pub fn to_affine(&self) -> G1Affine {
    //     G1Affine::default() //TODO:
    // }
}

pub const BASE_LIMBS: usize = 8;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct BaseCfg {}

const NUM_LIMBS: usize = 8;

impl FieldConfig for BaseCfg {}

pub type BaseField = Field<BASE_LIMBS, BaseCfg>;
pub type G1Affine = Affine<BASE_LIMBS, BaseCfg>;
pub type G1Projective = Projective<BASE_LIMBS, BaseCfg>;

extern "C" {
    fn Eq(point1: *const G1Projective, point2: *const G1Projective) -> c_uint;
    fn Zero(point: *mut G1Projective);
    fn ToAffine(point: *const G1Projective) -> G1Affine;
    fn RandomPoints(points: *mut G1Projective, size: c_uint);
}

impl PartialEq for G1Projective {
    fn eq(&self, other: &Self) -> bool {
        unsafe { Eq(self, other) != 0 }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::transmute_copy;

    use crate::curve::{G1Projective, ScalarField};
    use crate::utils::*;
    // use ark_bls12_381::{Fq, G1Affine, G1Projective};
    use ark_bn254::{Fq, G1Affine as arkG1Affine, G1Projective as arkG1Projective};
    use ark_ec::AffineCurve;
    use ark_ff::Field as ArkField;
    use ark_ff::PrimeField;
    use ark_ff::{BigInteger256, BigInteger384};

    use super::*;

    type BigIntegerScalarArk = BigInteger256;
    type BigIntegerBaseArk = BigInteger384;

    // impl Field<12> {
    //     pub fn to_ark(&self) -> BigIntegerBaseArk {
    //         BigIntegerBaseArk::new(
    //             u32_vec_to_u64_vec(&self.limbs())
    //                 .try_into()
    //                 .unwrap(),
    //         )
    //     }

    //     pub fn from_ark(ark: BigIntegerBaseArk) -> Self {
    //         Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    //     }

    //     pub fn to_ark_transmute(&self) -> BigIntegerBaseArk {
    //         unsafe { transmute_copy(self) }
    //     }

    //     pub fn from_ark_transmute(v: BigIntegerBaseArk) -> Self {
    //         unsafe { transmute_copy(&v) }
    //     }
    // }

    impl<const NUM_LIMBS: usize, F: FieldConfig> Field<NUM_LIMBS, F> {
        pub fn to_ark(&self) -> BigIntegerScalarArk {
            BigIntegerScalarArk::new(
                u32_vec_to_u64_vec(&self.get_limbs())
                    .try_into()
                    .unwrap(),
            )
        }

        pub fn from_ark(ark: BigIntegerScalarArk) -> Self {
            Self::set_limbs(&u64_vec_to_u32_vec(&ark.0))
        }

        pub fn to_ark_transmute(&self) -> BigIntegerScalarArk {
            unsafe { transmute_copy(self) }
        }

        pub fn from_ark_transmute(v: BigIntegerScalarArk) -> Self {
            unsafe { transmute_copy(&v) }
        }
    }

    impl G1Projective {
        pub fn to_ark(&self) -> arkG1Projective {
            //TODO: generic conversion
            self.to_ark_affine()
                .into_projective()
        }

        pub fn to_ark_affine(&self) -> arkG1Affine {
            //TODO: generic conversion
            use std::ops::Mul;
            let proj_x_field = Fq::from_le_bytes_mod_order(
                &self
                    .x
                    .to_bytes_le(),
            );
            let proj_y_field = Fq::from_le_bytes_mod_order(
                &self
                    .y
                    .to_bytes_le(),
            );
            let proj_z_field = Fq::from_le_bytes_mod_order(
                &self
                    .z
                    .to_bytes_le(),
            );
            let inverse_z = proj_z_field
                .inverse()
                .unwrap();
            let aff_x = proj_x_field.mul(inverse_z);
            let aff_y = proj_y_field.mul(inverse_z);
            arkG1Affine::new(aff_x, aff_y, false)
        }

        pub fn from_ark(ark: arkG1Projective) -> G1Projective {
            let z_inv = ark
                .z
                .inverse()
                .unwrap();
            let z_invsq = z_inv * z_inv;
            let z_invq3 = z_invsq * z_inv;
            G1Projective {
                x: BaseField::from_ark((ark.x * z_invsq).into_repr()),
                y: BaseField::from_ark((ark.y * z_invq3).into_repr()),
                z: BaseField::one(),
            }
        }
    }

    impl G1Affine {
        pub fn to_ark(&self) -> arkG1Affine {
            arkG1Affine::new(
                Fq::new(
                    self.x
                        .to_ark(),
                ),
                Fq::new(
                    self.y
                        .to_ark(),
                ),
                false,
            )
        }

        pub fn to_ark_repr(&self) -> arkG1Affine {
            arkG1Affine::new(
                Fq::from_repr(
                    self.x
                        .to_ark(),
                )
                .unwrap(),
                Fq::from_repr(
                    self.y
                        .to_ark(),
                )
                .unwrap(),
                false,
            )
        }

        pub fn from_ark(p: &arkG1Affine) -> Self {
            G1Affine {
                x: BaseField::from_ark(p.x.into_repr()),
                y: BaseField::from_ark(p.y.into_repr()),
            }
        }
    }

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField::set_limbs(&limbs);
        assert_eq!(
            scalar.to_ark(),
            scalar.to_ark_transmute(),
            "{:08X?} {:08X?}",
            scalar.to_ark(),
            scalar.to_ark_transmute()
        )
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_point_equality() {
        let left = G1Projective::zero();
        let right = G1Projective::zero();
        assert_eq!(left, right);
        let right = G1Projective::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &[0; BASE_LIMBS]);
        assert_eq!(left, right);
        let right = G1Projective::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &get_fixed_limbs::<BASE_LIMBS>(&[1]));
        assert_ne!(left, right);
        let left = G1Projective::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &get_fixed_limbs::<BASE_LIMBS>(&[1]));
        assert_eq!(left, right);
    }
}
