use std::ffi::c_uint;

use ark_bls12_377::{Fq as Fq_BLS12_377, Fr as Fr_BLS12_377, G1Affine as G1Affine_BLS12_377, G1Projective as G1Projective_BLS12_377};

use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};
use std::mem::transmute;

use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};

use crate::scalar_t::*;

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

pub const BASE_LIMBS_BLS_12_377: usize = 12;
pub const SCALAR_LIMBS_BLS_12_377: usize = 8;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct BaseField_BLS_12_377{
    pub inner_scalar : ScalarT<BASE_LIMBS_BLS_12_377>
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ScalarField_BLS_12_377{
    pub inner_scalar : ScalarT<SCALAR_LIMBS_BLS_12_377>
}

impl BaseField_BLS_12_377 {
    pub fn to_ark(&self) -> BigInteger384 {
        BigInteger384::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger384) -> Self {
        Self {
            inner_scalar: ScalarT::from_limbs(&u64_vec_to_u32_vec(&ark.0))
        }
    }
}


#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point_BLS_12_377 {
    pub x: BaseField_BLS_12_377,
    pub y: BaseField_BLS_12_377,
    pub z: BaseField_BLS_12_377,
}

impl Default for Point_BLS_12_377 {
    fn default() -> Self {
        Point_BLS_12_377::zero()
    }
}

impl Point_BLS_12_377 {
    pub fn zero() -> Self {
        Point_BLS_12_377 {
            x: BaseField_BLS_12_377{ inner_scalar: ScalarT::zero()},
            y: BaseField_BLS_12_377{ inner_scalar: ScalarT::one()},
            z: BaseField_BLS_12_377{ inner_scalar: ScalarT::zero()},
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn to_ark(&self) -> G1Projective_BLS12_377 {
        //TODO: generic conversion
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> G1Affine_BLS12_377 {
        //TODO: generic conversion
        use ark_ff::Field;
        use std::ops::Mul;
        let proj_x_field = Fq_BLS12_377::from_le_bytes_mod_order(&self.x.inner_scalar.to_bytes_le());
        let proj_y_field = Fq_BLS12_377::from_le_bytes_mod_order(&self.y.inner_scalar.to_bytes_le());
        let proj_z_field = Fq_BLS12_377::from_le_bytes_mod_order(&self.z.inner_scalar.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine_BLS12_377::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective_BLS12_377) -> Point_BLS_12_377 {
        use ark_ff::Field;
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point_BLS_12_377 {
            x: BaseField_BLS_12_377::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField_BLS_12_377::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField_BLS_12_377{inner_scalar : ScalarT::one()},
        }
    }

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point_BLS_12_377 {
            x: BaseField_BLS_12_377{ inner_scalar: ScalarT::from_limbs(x)},
            y: BaseField_BLS_12_377{ inner_scalar: ScalarT::from_limbs(y)},
            z: BaseField_BLS_12_377{ inner_scalar: ScalarT::from_limbs(z)}
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point_BLS_12_377 {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS_BLS_12_377, "length must be 3 * {}", BASE_LIMBS_BLS_12_377);
        Point_BLS_12_377 {
            x: BaseField_BLS_12_377 {
                inner_scalar: ScalarT {s : value[..BASE_LIMBS_BLS_12_377].try_into().unwrap()},
            },
            y: BaseField_BLS_12_377 {
                inner_scalar: ScalarT {s : value[BASE_LIMBS_BLS_12_377..BASE_LIMBS_BLS_12_377 * 2].try_into().unwrap()},
            },
            z: BaseField_BLS_12_377 {
                inner_scalar: ScalarT {s : value[BASE_LIMBS_BLS_12_377 * 2..].try_into().unwrap()},
            },
        }
    }
}

extern "C" {
    fn eq_bls12_377(point1: *const Point_BLS_12_377, point2: *const Point_BLS_12_377) -> c_uint;
}

impl PartialEq for Point_BLS_12_377 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_bls12_377(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity_BLS_12_377 {
    pub x: BaseField_BLS_12_377,
    pub y: BaseField_BLS_12_377,
}

impl Default for PointAffineNoInfinity_BLS_12_377 {
    fn default() -> Self {
        PointAffineNoInfinity_BLS_12_377 {
            x: BaseField_BLS_12_377{ inner_scalar: ScalarT::zero()},
            y: BaseField_BLS_12_377{ inner_scalar: ScalarT::zero()},
        }
    }
}

impl PointAffineNoInfinity_BLS_12_377 {
    // TODO: generics
    ///Fr_BLS12_377om u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity_BLS_12_377 {
            x: BaseField_BLS_12_377{ inner_scalar: ScalarT::from_limbs(x)},
            y: BaseField_BLS_12_377{ inner_scalar: ScalarT::from_limbs(y)},
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.inner_scalar.limbs(), self.y.inner_scalar.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point_BLS_12_377 {
        Point_BLS_12_377 {
            x: self.x,
            y: self.y,
            z: BaseField_BLS_12_377{inner_scalar: ScalarT::one()},
        }
    }

    pub fn to_ark(&self) -> G1Affine_BLS12_377 {
        G1Affine_BLS12_377::new(Fq_BLS12_377::new(self.x.to_ark()), Fq_BLS12_377::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> G1Affine_BLS12_377 {
        G1Affine_BLS12_377::new(
            Fq_BLS12_377::from_repr(self.x.to_ark()).unwrap(),
            Fq_BLS12_377::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &G1Affine_BLS12_377) -> Self {
        PointAffineNoInfinity_BLS_12_377 {
            x: BaseField_BLS_12_377::from_ark(p.x.into_repr()),
            y: BaseField_BLS_12_377::from_ark(p.y.into_repr()),
        }
    }
}

impl Point_BLS_12_377 {
    // TODO: generics

    pub fn to_affine(&self) -> PointAffineNoInfinity_BLS_12_377 {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity_BLS_12_377 {
            x: BaseField_BLS_12_377::from_ark(ark_affine.x.into_repr()),
            y: BaseField_BLS_12_377::from_ark(ark_affine.y.into_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity_BLS_12_377 {
        PointAffineNoInfinity_BLS_12_377 {
            x: self.x,
            y: self.y,
        }
    }
}


impl ScalarField_BLS_12_377 {

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

    pub fn from_biginteger_transmute(v: BigInteger256) -> ScalarField_BLS_12_377 {
        ScalarField_BLS_12_377{ inner_scalar: unsafe { transmute(v) }}
    }

    pub fn to_ark_transmute(&self) -> Fr_BLS12_377 {
        unsafe { std::mem::transmute(*self) }
    }

    pub fn from_ark_transmute(v: &Fr_BLS12_377) -> ScalarField_BLS_12_377 {
        unsafe { std::mem::transmute_copy(v) }
    }

    pub fn to_ark_mod_p(&self) -> Fr_BLS12_377 {
        Fr_BLS12_377::new(BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap()))
    }

    pub fn to_ark_repr(&self) -> Fr_BLS12_377 {
        Fr_BLS12_377::from_repr(BigInteger256::new(u32_vec_to_u64_vec(&self.inner_scalar.limbs()).try_into().unwrap())).unwrap()
    }

    pub fn from_ark(v: BigInteger256) -> ScalarField_BLS_12_377 {
        Self { inner_scalar: ScalarT::from_limbs(&u64_vec_to_u32_vec(&v.0))}
    }

}

unsafe impl DeviceCopy for BaseField_BLS_12_377 {}
unsafe impl DeviceCopy for ScalarField_BLS_12_377 {}
