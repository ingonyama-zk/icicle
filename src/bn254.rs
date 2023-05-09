use std::ffi::c_uint;

use ark_bn254::{Fq as Fq_BN254, Fr as Fr_BN254, G1Affine as G1Affine_BN254, G1Projective as G1Projective_BN254};

use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};
use std::mem::transmute;

use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};

use crate::scalar_t::*;

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

pub const BASE_LIMBS_BN_254: usize = 8;
pub const SCALAR_LIMBS_BN_254: usize = 8;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct BaseField_BN_254{
    pub inner_scalar : ScalarT<BASE_LIMBS_BN_254>
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ScalarField_BN_254{
    pub inner_scalar : ScalarT<SCALAR_LIMBS_BN_254>
}

impl BaseField_BN_254 {
    pub fn to_ark(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger256) -> Self {
        Self {
            inner_scalar: ScalarT::from_limbs(&u64_vec_to_u32_vec(&ark.0))
        }
    }
}


#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point_BN_254 {
    pub x: BaseField_BN_254,
    pub y: BaseField_BN_254,
    pub z: BaseField_BN_254,
}

impl Default for Point_BN_254 {
    fn default() -> Self {
        Point_BN_254::zero()
    }
}

impl Point_BN_254 {
    pub fn zero() -> Self {
        Point_BN_254 {
            x: BaseField_BN_254{ inner_scalar: ScalarT::zero()},
            y: BaseField_BN_254{ inner_scalar: ScalarT::one()},
            z: BaseField_BN_254{ inner_scalar: ScalarT::zero()},
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn to_ark(&self) -> G1Projective_BN254 {
        //TODO: generic conversion
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> G1Affine_BN254 {
        //TODO: generic conversion
        use ark_ff::Field;
        use std::ops::Mul;
        let proj_x_field = Fq_BN254::from_le_bytes_mod_order(&self.x.inner_scalar.to_bytes_le());
        let proj_y_field = Fq_BN254::from_le_bytes_mod_order(&self.y.inner_scalar.to_bytes_le());
        let proj_z_field = Fq_BN254::from_le_bytes_mod_order(&self.z.inner_scalar.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine_BN254::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective_BN254) -> Point_BN_254 {
        use ark_ff::Field;
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point_BN_254 {
            x: BaseField_BN_254::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField_BN_254::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField_BN_254{inner_scalar : ScalarT::one()},
        }
    }

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point_BN_254 {
            x: BaseField_BN_254{ inner_scalar: ScalarT::from_limbs(x)},
            y: BaseField_BN_254{ inner_scalar: ScalarT::from_limbs(y)},
            z: BaseField_BN_254{ inner_scalar: ScalarT::from_limbs(z)}
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point_BN_254 {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS_BN_254, "length must be 3 * {}", BASE_LIMBS_BN_254);
        Point_BN_254 {
            x: BaseField_BN_254 {
                inner_scalar: ScalarT {s : value[..BASE_LIMBS_BN_254].try_into().unwrap()},
            },
            y: BaseField_BN_254 {
                inner_scalar: ScalarT {s : value[BASE_LIMBS_BN_254..BASE_LIMBS_BN_254 * 2].try_into().unwrap()},
            },
            z: BaseField_BN_254 {
                inner_scalar: ScalarT {s : value[BASE_LIMBS_BN_254 * 2..].try_into().unwrap()},
            },
        }
    }
}

extern "C" {
    fn eq_bn254(point1: *const Point_BN_254, point2: *const Point_BN_254) -> c_uint;
}

impl PartialEq for Point_BN_254 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_bn254(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity_BN_254 {
    pub x: BaseField_BN_254,
    pub y: BaseField_BN_254,
}

impl Default for PointAffineNoInfinity_BN_254 {
    fn default() -> Self {
        PointAffineNoInfinity_BN_254 {
            x: BaseField_BN_254{ inner_scalar: ScalarT::zero()},
            y: BaseField_BN_254{ inner_scalar: ScalarT::zero()},
        }
    }
}

impl PointAffineNoInfinity_BN_254 {
    // TODO: generics
    ///Fr_BN254om u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity_BN_254 {
            x: BaseField_BN_254{ inner_scalar: ScalarT::from_limbs(x)},
            y: BaseField_BN_254{ inner_scalar: ScalarT::from_limbs(y)},
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.inner_scalar.limbs(), self.y.inner_scalar.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point_BN_254 {
        Point_BN_254 {
            x: self.x,
            y: self.y,
            z: BaseField_BN_254{inner_scalar: ScalarT::one()},
        }
    }

    pub fn to_ark(&self) -> G1Affine_BN254 {
        G1Affine_BN254::new(Fq_BN254::new(self.x.to_ark()), Fq_BN254::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> G1Affine_BN254 {
        G1Affine_BN254::new(
            Fq_BN254::from_repr(self.x.to_ark()).unwrap(),
            Fq_BN254::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &G1Affine_BN254) -> Self {
        PointAffineNoInfinity_BN_254 {
            x: BaseField_BN_254::from_ark(p.x.into_repr()),
            y: BaseField_BN_254::from_ark(p.y.into_repr()),
        }
    }
}

impl Point_BN_254 {
    // TODO: generics

    pub fn to_affine(&self) -> PointAffineNoInfinity_BN_254 {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity_BN_254 {
            x: BaseField_BN_254::from_ark(ark_affine.x.into_repr()),
            y: BaseField_BN_254::from_ark(ark_affine.y.into_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity_BN_254 {
        PointAffineNoInfinity_BN_254 {
            x: self.x,
            y: self.y,
        }
    }
}


impl ScalarField_BN_254 {

    pub fn to_biginteger254(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap())
    }

    pub fn to_ark(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap())
    }

    pub fn from_biginteger256(ark: BigInteger256) -> Self {
        Self{ inner_scalar: ScalarT::from_limbs(&u64_vec_to_u32_vec(&ark.0))}
    }

    pub fn to_biginteger256_transmute(&self) -> BigInteger256 {
        unsafe { transmute(*self) }
    }

    pub fn from_biginteger_transmute(v: BigInteger256) -> ScalarField_BN_254 {
        ScalarField_BN_254{ inner_scalar: unsafe { transmute(v) }}
    }

    pub fn to_ark_transmute(&self) -> Fr_BN254 {
        unsafe { std::mem::transmute(*self) }
    }

    pub fn from_ark_transmute(v: &Fr_BN254) -> ScalarField_BN_254 {
        unsafe { std::mem::transmute_copy(v) }
    }

    pub fn to_ark_mod_p(&self) -> Fr_BN254 {
        Fr_BN254::new(BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap()))
    }

    pub fn to_ark_repr(&self) -> Fr_BN254 {
        Fr_BN254::from_repr(BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap())).unwrap()
    }

    pub fn from_ark(v: BigInteger256) -> ScalarField_BN_254 {
        Self { inner_scalar: ScalarT::from_limbs(&u64_vec_to_u32_vec(&v.0))}
    }

}

unsafe impl DeviceCopy for BaseField_BN_254 {}
unsafe impl DeviceCopy for ScalarField_BN_254 {}
