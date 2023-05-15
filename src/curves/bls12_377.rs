use std::ffi::c_uint;

use ark_bls12_377::{Fq as Fq_BLS12_377, Fr as Fr_BLS12_377, G1Affine as G1Affine_BLS12_377, G1Projective as G1Projective_BLS12_377};

use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};
use std::mem::transmute;
use ark_ff::Field;
use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field_BLS12_377<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for Field_BLS12_377<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> Default for Field_BLS12_377<NUM_LIMBS> {
    fn default() -> Self {
        Field_BLS12_377::zero()
    }
}

impl<const NUM_LIMBS: usize> Field_BLS12_377<NUM_LIMBS> {
    pub fn zero() -> Self {
        Field_BLS12_377 {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Field_BLS12_377 { s }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

pub const BASE_LIMBS_BLS12_377: usize = 12;
pub const SCALAR_LIMBS_BLS12_377: usize = 8;

pub type BaseField_BLS12_377 = Field_BLS12_377<BASE_LIMBS_BLS12_377>;
pub type ScalarField_BLS12_377 = Field_BLS12_377<SCALAR_LIMBS_BLS12_377>;

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

impl BaseField_BLS12_377 {
    pub fn limbs(&self) -> [u32; BASE_LIMBS_BLS12_377] {
        self.s
    }

    pub fn from_limbs(value: &[u32]) -> Self {
        Self {
            s: get_fixed_limbs(value),
        }
    }

    pub fn to_ark(&self) -> BigInteger384 {
        BigInteger384::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger384) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }
}

impl ScalarField_BLS12_377 {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS_BLS12_377] {
        self.s
    }

    pub fn to_ark(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger256) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }

    pub fn to_ark_transmute(&self) -> BigInteger256 {
        unsafe { transmute(*self) }
    }

    pub fn from_ark_transmute(v: BigInteger256) -> ScalarField_BLS12_377 {
        unsafe { transmute(v) }
    }
}

#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point_BLS12_377 {
    pub x: BaseField_BLS12_377,
    pub y: BaseField_BLS12_377,
    pub z: BaseField_BLS12_377,
}

impl Default for Point_BLS12_377 {
    fn default() -> Self {
        Point_BLS12_377::zero()
    }
}

impl Point_BLS12_377 {
    pub fn zero() -> Self {
        Point_BLS12_377 {
            x: BaseField_BLS12_377::zero(),
            y: BaseField_BLS12_377::one(),
            z: BaseField_BLS12_377::zero(),
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
        let proj_x_field = Fq_BLS12_377::from_le_bytes_mod_order(&self.x.to_bytes_le());
        let proj_y_field = Fq_BLS12_377::from_le_bytes_mod_order(&self.y.to_bytes_le());
        let proj_z_field = Fq_BLS12_377::from_le_bytes_mod_order(&self.z.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine_BLS12_377::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective_BLS12_377) -> Point_BLS12_377 {
        use ark_ff::Field;
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point_BLS12_377 {
            x: BaseField_BLS12_377::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField_BLS12_377::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField_BLS12_377::one(),
        }
    }
}

extern "C" {
    fn eq_bls12_377(point1: *const Point_BLS12_377, point2: *const Point_BLS12_377) -> c_uint;
}

impl PartialEq for Point_BLS12_377 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_bls12_377(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity_BLS12_377 {
    pub x: BaseField_BLS12_377,
    pub y: BaseField_BLS12_377,
}

impl Default for PointAffineNoInfinity_BLS12_377 {
    fn default() -> Self {
        PointAffineNoInfinity_BLS12_377 {
            x: BaseField_BLS12_377::zero(),
            y: BaseField_BLS12_377::zero(),
        }
    }
}

impl PointAffineNoInfinity_BLS12_377 {
    // TODO: generics
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity_BLS12_377 {
            x: BaseField_BLS12_377 {
                s: get_fixed_limbs(x),
            },
            y: BaseField_BLS12_377 {
                s: get_fixed_limbs(y),
            },
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point_BLS12_377 {
        Point_BLS12_377 {
            x: self.x,
            y: self.y,
            z: BaseField_BLS12_377::one(),
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
        PointAffineNoInfinity_BLS12_377 {
            x: BaseField_BLS12_377::from_ark(p.x.into_repr()),
            y: BaseField_BLS12_377::from_ark(p.y.into_repr()),
        }
    }
}

impl Point_BLS12_377 {
    // TODO: generics

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point_BLS12_377 {
            x: BaseField_BLS12_377 {
                s: get_fixed_limbs(x),
            },
            y: BaseField_BLS12_377 {
                s: get_fixed_limbs(y),
            },
            z: BaseField_BLS12_377 {
                s: get_fixed_limbs(z),
            },
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point_BLS12_377 {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS_BLS12_377, "length must be 3 * {}", BASE_LIMBS_BLS12_377);
        Point_BLS12_377 {
            x: BaseField_BLS12_377 {
                s: value[..BASE_LIMBS_BLS12_377].try_into().unwrap(),
            },
            y: BaseField_BLS12_377 {
                s: value[BASE_LIMBS_BLS12_377..BASE_LIMBS_BLS12_377 * 2].try_into().unwrap(),
            },
            z: BaseField_BLS12_377 {
                s: value[BASE_LIMBS_BLS12_377 * 2..].try_into().unwrap(),
            },
        }
    }

    pub fn to_affine(&self) -> PointAffineNoInfinity_BLS12_377 {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity_BLS12_377 {
            x: BaseField_BLS12_377::from_ark(ark_affine.x.into_repr()),
            y: BaseField_BLS12_377::from_ark(ark_affine.y.into_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity_BLS12_377 {
        PointAffineNoInfinity_BLS12_377 {
            x: self.x,
            y: self.y,
        }
    }
}

impl ScalarField_BLS12_377 {
    pub fn from_limbs(value: &[u32]) -> ScalarField_BLS12_377 {
        ScalarField_BLS12_377 {
            s: get_fixed_limbs(value),
        }
    }
}


#[cfg(test)]
mod tests {
    use ark_bls12_377::{Fr as Fr_BLS12_377};

    use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, curves::bls12_377::{Point_BLS12_377, ScalarField_BLS12_377}};

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField_BLS12_377::from_limbs(&limbs);
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
        let left = Point_BLS12_377::zero();
        let right = Point_BLS12_377::zero();
        assert_eq!(left, right);
        let right = Point_BLS12_377::from_limbs(&[0; 12], &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0; 12]);
        assert_eq!(left, right);
        let right = Point_BLS12_377::from_limbs(
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0; 12],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert!(left != right);
    }
}