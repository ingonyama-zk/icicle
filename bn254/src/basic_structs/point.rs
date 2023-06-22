use std::ffi::c_uint;

use ark_bn254::{Fq as Fq_BN254, Fr as Fr_BN254, G1Affine as G1Affine_BN254, G1Projective as G1Projective_BN254};

use ark_ec::AffineCurve;
use ark_ff::{BigInteger256, PrimeField};
use std::mem::transmute;
use ark_ff::Field;
use icicle_core::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

use super::scalar::{get_fixed_limbs, self};


#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointT<BF: scalar::ScalarTrait> {
    pub x: BF,
    pub y: BF,
    pub z: BF,
}

impl<BF: DeviceCopy + scalar::ScalarTrait> Default for PointT<BF> {
    fn default() -> Self {
        PointT::zero()
    }
}

impl<BF: DeviceCopy + scalar::ScalarTrait> PointT<BF> {
    pub fn zero() -> Self {
        PointT {
            x: BF::zero(),
            y: BF::one(),
            z: BF::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinityT<BF> {
    pub x: BF,
    pub y: BF,
}

impl<BF: scalar::ScalarTrait> Default for PointAffineNoInfinityT<BF> {
    fn default() -> Self {
        PointAffineNoInfinityT {
            x: BF::zero(),
            y: BF::zero(),
        }
    }
}

impl<BF: Copy + scalar::ScalarTrait> PointAffineNoInfinityT<BF> {
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinityT {
            x: BF::from_limbs(x),
            y: BF::from_limbs(y)
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> PointT<BF> {
        PointT {
            x: self.x,
            y: self.y,
            z: BF::one(),
        }
    }
}

impl<BF: Copy + scalar::ScalarTrait> PointT<BF>  {
    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        PointT {
            x: BF::from_limbs(x),
            y: BF::from_limbs(y),
            z: BF::from_limbs(z)
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> PointT<BF> {
        let l = value.len();
        assert_eq!(l, 3 * BF::base_limbs(), "length must be 3 * {}", BF::base_limbs());
        PointT {
            x: BF::from_limbs(value[..BF::base_limbs()].try_into().unwrap()),
            y: BF::from_limbs(value[BF::base_limbs()..BF::base_limbs() * 2].try_into().unwrap()),
            z: BF::from_limbs(value[BF::base_limbs() * 2..].try_into().unwrap())
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinityT<BF> {
        PointAffineNoInfinityT {
            x: self.x,
            y: self.y,
        }
    }
}
