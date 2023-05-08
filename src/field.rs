use std::ffi::c_uint;
use std::mem::transmute;

use ark_bls12_381::{Fq, G1Affine, G1Projective};
use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

use crate::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for Field<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> Default for Field<NUM_LIMBS> {
    fn default() -> Self {
        Field::zero()
    }
}

impl<const NUM_LIMBS: usize> Field<NUM_LIMBS> {
    pub fn zero() -> Self {
        Field {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Field { s }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

pub const BASE_LIMBS: usize = 12;
pub const SCALAR_LIMBS: usize = 8;

#[cfg(feature = "bn254")]
pub const BASE_LIMBS: usize = 8;
#[cfg(feature = "bn254")]
pub const SCALAR_LIMBS: usize = 8;

pub type BaseField = Field<BASE_LIMBS>;
pub type ScalarField = Field<SCALAR_LIMBS>;

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

impl BaseField {
    pub fn limbs(&self) -> [u32; BASE_LIMBS] {
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

impl ScalarField {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS] {
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

    pub fn from_ark_transmute(v: BigInteger256) -> ScalarField {
        unsafe { transmute(v) }
    }
}

#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point {
    pub x: BaseField,
    pub y: BaseField,
    pub z: BaseField,
}

impl Default for Point {
    fn default() -> Self {
        Point::zero()
    }
}

impl Point {
    pub fn zero() -> Self {
        Point {
            x: BaseField::zero(),
            y: BaseField::one(),
            z: BaseField::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn to_ark(&self) -> G1Projective {
        //TODO: generic conversion
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> G1Affine {
        //TODO: generic conversion
        use ark_ff::Field;
        use std::ops::Mul;
        let proj_x_field = Fq::from_le_bytes_mod_order(&self.x.to_bytes_le());
        let proj_y_field = Fq::from_le_bytes_mod_order(&self.y.to_bytes_le());
        let proj_z_field = Fq::from_le_bytes_mod_order(&self.z.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective) -> Point {
        use ark_ff::Field;
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point {
            x: BaseField::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField::one(),
        }
    }
}

extern "C" {
    fn eq(point1: *const Point, point2: *const Point) -> c_uint;
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity {
    pub x: BaseField,
    pub y: BaseField,
}

impl Default for PointAffineNoInfinity {
    fn default() -> Self {
        PointAffineNoInfinity {
            x: BaseField::zero(),
            y: BaseField::zero(),
        }
    }
}

impl PointAffineNoInfinity {
    // TODO: generics
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity {
            x: BaseField {
                s: get_fixed_limbs(x),
            },
            y: BaseField {
                s: get_fixed_limbs(y),
            },
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point {
        Point {
            x: self.x,
            y: self.y,
            z: BaseField::one(),
        }
    }

    pub fn to_ark(&self) -> G1Affine {
        G1Affine::new(Fq::new(self.x.to_ark()), Fq::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> G1Affine {
        G1Affine::new(
            Fq::from_repr(self.x.to_ark()).unwrap(),
            Fq::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &G1Affine) -> Self {
        PointAffineNoInfinity {
            x: BaseField::from_ark(p.x.into_repr()),
            y: BaseField::from_ark(p.y.into_repr()),
        }
    }
}

impl Point {
    // TODO: generics

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point {
            x: BaseField {
                s: get_fixed_limbs(x),
            },
            y: BaseField {
                s: get_fixed_limbs(y),
            },
            z: BaseField {
                s: get_fixed_limbs(z),
            },
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS, "length must be 3 * {}", BASE_LIMBS);
        Point {
            x: BaseField {
                s: value[..BASE_LIMBS].try_into().unwrap(),
            },
            y: BaseField {
                s: value[BASE_LIMBS..BASE_LIMBS * 2].try_into().unwrap(),
            },
            z: BaseField {
                s: value[BASE_LIMBS * 2..].try_into().unwrap(),
            },
        }
    }

    pub fn to_affine(&self) -> PointAffineNoInfinity {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity {
            x: BaseField::from_ark(ark_affine.x.into_repr()),
            y: BaseField::from_ark(ark_affine.y.into_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity {
        PointAffineNoInfinity {
            x: self.x,
            y: self.y,
        }
    }
}

impl ScalarField {
    pub fn from_limbs(value: &[u32]) -> ScalarField {
        ScalarField {
            s: get_fixed_limbs(value),
        }
    }
}


#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;

    use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, field::{Point, ScalarField}};

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField::from_limbs(&limbs);
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
        let left = Point::zero();
        let right = Point::zero();
        assert_eq!(left, right);
        let right = Point::from_limbs(&[0; 12], &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0; 12]);
        assert_eq!(left, right);
        let right = Point::from_limbs(
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0; 12],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert!(left != right);
    }
}
