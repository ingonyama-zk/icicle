use std::ffi::c_uint;

use ark_CURVE_NAME_L::{Fq as Fq_CURVE_NAME_U, Fr as Fr_CURVE_NAME_U, G1Affine as G1Affine_CURVE_NAME_U, G1Projective as G1Projective_CURVE_NAME_U};

use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};
use std::mem::transmute;
use ark_ff::Field;
use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field_CURVE_NAME_U<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for Field_CURVE_NAME_U<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> Default for Field_CURVE_NAME_U<NUM_LIMBS> {
    fn default() -> Self {
        Field_CURVE_NAME_U::zero()
    }
}

impl<const NUM_LIMBS: usize> Field_CURVE_NAME_U<NUM_LIMBS> {
    pub fn zero() -> Self {
        Field_CURVE_NAME_U {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Field_CURVE_NAME_U { s }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

pub const BASE_LIMBS_CURVE_NAME_U: usize = 12;
pub const SCALAR_LIMBS_CURVE_NAME_U: usize = 8;

pub type BaseField_CURVE_NAME_U = Field_CURVE_NAME_U<BASE_LIMBS_CURVE_NAME_U>;
pub type ScalarField_CURVE_NAME_U = Field_CURVE_NAME_U<SCALAR_LIMBS_CURVE_NAME_U>;

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

impl BaseField_CURVE_NAME_U {
    pub fn limbs(&self) -> [u32; BASE_LIMBS_CURVE_NAME_U] {
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

impl ScalarField_CURVE_NAME_U {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS_CURVE_NAME_U] {
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

    pub fn from_ark_transmute(v: BigInteger256) -> ScalarField_CURVE_NAME_U {
        unsafe { transmute(v) }
    }
}

#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point_CURVE_NAME_U {
    pub x: BaseField_CURVE_NAME_U,
    pub y: BaseField_CURVE_NAME_U,
    pub z: BaseField_CURVE_NAME_U,
}

impl Default for Point_CURVE_NAME_U {
    fn default() -> Self {
        Point_CURVE_NAME_U::zero()
    }
}

impl Point_CURVE_NAME_U {
    pub fn zero() -> Self {
        Point_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U::zero(),
            y: BaseField_CURVE_NAME_U::one(),
            z: BaseField_CURVE_NAME_U::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn to_ark(&self) -> G1Projective_CURVE_NAME_U {
        //TODO: generic conversion
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> G1Affine_CURVE_NAME_U {
        //TODO: generic conversion
        use ark_ff::Field;
        use std::ops::Mul;
        let proj_x_field = Fq_CURVE_NAME_U::from_le_bytes_mod_order(&self.x.to_bytes_le());
        let proj_y_field = Fq_CURVE_NAME_U::from_le_bytes_mod_order(&self.y.to_bytes_le());
        let proj_z_field = Fq_CURVE_NAME_U::from_le_bytes_mod_order(&self.z.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine_CURVE_NAME_U::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective_CURVE_NAME_U) -> Point_CURVE_NAME_U {
        use ark_ff::Field;
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField_CURVE_NAME_U::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField_CURVE_NAME_U::one(),
        }
    }
}

extern "C" {
    fn eq_CURVE_NAME_L(point1: *const Point_CURVE_NAME_U, point2: *const Point_CURVE_NAME_U) -> c_uint;
}

impl PartialEq for Point_CURVE_NAME_U {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_CURVE_NAME_L(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity_CURVE_NAME_U {
    pub x: BaseField_CURVE_NAME_U,
    pub y: BaseField_CURVE_NAME_U,
}

impl Default for PointAffineNoInfinity_CURVE_NAME_U {
    fn default() -> Self {
        PointAffineNoInfinity_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U::zero(),
            y: BaseField_CURVE_NAME_U::zero(),
        }
    }
}

impl PointAffineNoInfinity_CURVE_NAME_U {
    // TODO: generics
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U {
                s: get_fixed_limbs(x),
            },
            y: BaseField_CURVE_NAME_U {
                s: get_fixed_limbs(y),
            },
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point_CURVE_NAME_U {
        Point_CURVE_NAME_U {
            x: self.x,
            y: self.y,
            z: BaseField_CURVE_NAME_U::one(),
        }
    }

    pub fn to_ark(&self) -> G1Affine_CURVE_NAME_U {
        G1Affine_CURVE_NAME_U::new(Fq_CURVE_NAME_U::new(self.x.to_ark()), Fq_CURVE_NAME_U::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> G1Affine_CURVE_NAME_U {
        G1Affine_CURVE_NAME_U::new(
            Fq_CURVE_NAME_U::from_repr(self.x.to_ark()).unwrap(),
            Fq_CURVE_NAME_U::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &G1Affine_CURVE_NAME_U) -> Self {
        PointAffineNoInfinity_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U::from_ark(p.x.into_repr()),
            y: BaseField_CURVE_NAME_U::from_ark(p.y.into_repr()),
        }
    }
}

impl Point_CURVE_NAME_U {
    // TODO: generics

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U {
                s: get_fixed_limbs(x),
            },
            y: BaseField_CURVE_NAME_U {
                s: get_fixed_limbs(y),
            },
            z: BaseField_CURVE_NAME_U {
                s: get_fixed_limbs(z),
            },
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point_CURVE_NAME_U {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS_CURVE_NAME_U, "length must be 3 * {}", BASE_LIMBS_CURVE_NAME_U);
        Point_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U {
                s: value[..BASE_LIMBS_CURVE_NAME_U].try_into().unwrap(),
            },
            y: BaseField_CURVE_NAME_U {
                s: value[BASE_LIMBS_CURVE_NAME_U..BASE_LIMBS_CURVE_NAME_U * 2].try_into().unwrap(),
            },
            z: BaseField_CURVE_NAME_U {
                s: value[BASE_LIMBS_CURVE_NAME_U * 2..].try_into().unwrap(),
            },
        }
    }

    pub fn to_affine(&self) -> PointAffineNoInfinity_CURVE_NAME_U {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity_CURVE_NAME_U {
            x: BaseField_CURVE_NAME_U::from_ark(ark_affine.x.into_repr()),
            y: BaseField_CURVE_NAME_U::from_ark(ark_affine.y.into_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity_CURVE_NAME_U {
        PointAffineNoInfinity_CURVE_NAME_U {
            x: self.x,
            y: self.y,
        }
    }
}

impl ScalarField_CURVE_NAME_U {
    pub fn from_limbs(value: &[u32]) -> ScalarField_CURVE_NAME_U {
        ScalarField_CURVE_NAME_U {
            s: get_fixed_limbs(value),
        }
    }
}


#[cfg(test)]
mod tests {
    use ark_CURVE_NAME_L::{Fr as Fr_CURVE_NAME_U};

    use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, curves::CURVE_NAME_L::{Point_CURVE_NAME_U, ScalarField_CURVE_NAME_U}};

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField_CURVE_NAME_U::from_limbs(&limbs);
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
        let left = Point_CURVE_NAME_U::zero();
        let right = Point_CURVE_NAME_U::zero();
        assert_eq!(left, right);
        let right = Point_CURVE_NAME_U::from_limbs(&[0; 12], &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0; 12]);
        assert_eq!(left, right);
        let right = Point_CURVE_NAME_U::from_limbs(
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0; 12],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert!(left != right);
    }
}