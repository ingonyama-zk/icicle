use std::ffi::c_uint;
use ark_bw6_761::{Fq as Fq_BW6_761, Fr as Fr_BW6_761, G1Affine as G1Affine_BW6_761, G1Projective as G1Projective_BW6_761};
use ark_ec::AffineCurve;
use ark_ff::{BigInteger768, BigInteger384, PrimeField};
use std::mem::transmute;
use ark_ff::Field;
use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};
use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field_BW6_761<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for Field_BW6_761<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> Default for Field_BW6_761<NUM_LIMBS> {
    fn default() -> Self {
        Field_BW6_761::zero()
    }
}

impl<const NUM_LIMBS: usize> Field_BW6_761<NUM_LIMBS> {
    pub fn zero() -> Self {
        Field_BW6_761 {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Field_BW6_761 { s }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

pub const BASE_LIMBS_BW6_761: usize = 24;
pub const SCALAR_LIMBS_BW6_761: usize = 12;

pub type BaseField_BW6_761 = Field_BW6_761<BASE_LIMBS_BW6_761>;
pub type ScalarField_BW6_761 = Field_BW6_761<SCALAR_LIMBS_BW6_761>;

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

// 
impl BaseField_BW6_761 {
    pub fn limbs(&self) -> [u32; BASE_LIMBS_BW6_761] {
        self.s
    }

    pub fn from_limbs(value: &[u32]) -> Self {
        Self {
            s: get_fixed_limbs(value),
        }
    }

    pub fn to_ark(&self) -> BigInteger768 {
        BigInteger768::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger768) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }
}
//

impl ScalarField_BW6_761 {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS_BW6_761] {
        self.s
    }

    pub fn to_ark(&self) -> BigInteger384 {
        BigInteger384::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger384) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }

    pub fn to_ark_transmute(&self) -> BigInteger384 {
        unsafe { transmute(*self) }
    }

    pub fn from_ark_transmute(v: BigInteger384) -> ScalarField_BW6_761 {
        unsafe { transmute(v) }
    }
}

#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point_BW6_761 {
    pub x: BaseField_BW6_761,
    pub y: BaseField_BW6_761,
    pub z: BaseField_BW6_761,
}

impl Default for Point_BW6_761 {
    fn default() -> Self {
        Point_BW6_761::zero()
    }
}

impl Point_BW6_761 {
    pub fn zero() -> Self {
        Point_BW6_761 {
            x: BaseField_BW6_761::zero(),
            y: BaseField_BW6_761::one(),
            z: BaseField_BW6_761::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn to_ark(&self) -> G1Projective_BW6_761 {
        //TODO: generic conversion
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> G1Affine_BW6_761 {
        //TODO: generic conversion
        use std::ops::Mul;
        let proj_x_field = Fq_BW6_761::from_le_bytes_mod_order(&self.x.to_bytes_le());
        let proj_y_field = Fq_BW6_761::from_le_bytes_mod_order(&self.y.to_bytes_le());
        let proj_z_field = Fq_BW6_761::from_le_bytes_mod_order(&self.z.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine_BW6_761::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective_BW6_761) -> Point_BW6_761 {
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point_BW6_761 {
            x: BaseField_BW6_761::from_ark((ark.x * z_invsq).into_repr()),
            y: BaseField_BW6_761::from_ark((ark.y * z_invq3).into_repr()),
            z: BaseField_BW6_761::one(),
        }
    }
}

extern "C" {
    fn eq_bw6_761(point1: *const Point_BW6_761, point2: *const Point_BW6_761) -> c_uint;
}

impl PartialEq for Point_BW6_761 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_bw6_761(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity_BW6_761 {
    pub x: BaseField_BW6_761,
    pub y: BaseField_BW6_761,
}

impl Default for PointAffineNoInfinity_BW6_761 {
    fn default() -> Self {
        PointAffineNoInfinity_BW6_761 {
            x: BaseField_BW6_761::zero(),
            y: BaseField_BW6_761::zero(),
        }
    }
}

impl PointAffineNoInfinity_BW6_761 {
    // TODO: generics
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity_BW6_761 {
            x: BaseField_BW6_761 {
                s: get_fixed_limbs(x),
            },
            y: BaseField_BW6_761 {
                s: get_fixed_limbs(y),
            },
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point_BW6_761 {
        Point_BW6_761 {
            x: self.x,
            y: self.y,
            z: BaseField_BW6_761::one(),
        }
    }

    pub fn to_ark(&self) -> G1Affine_BW6_761 {
        G1Affine_BW6_761::new(Fq_BW6_761::new(self.x.to_ark()), Fq_BW6_761::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> G1Affine_BW6_761 {
        G1Affine_BW6_761::new(
            Fq_BW6_761::from_repr(self.x.to_ark()).unwrap(),
            Fq_BW6_761::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &G1Affine_BW6_761) -> Self {
        PointAffineNoInfinity_BW6_761 {
            x: BaseField_BW6_761::from_ark(p.x.into_repr()),
            y: BaseField_BW6_761::from_ark(p.y.into_repr()),
        }
    }
}

impl Point_BW6_761 {
    // TODO: generics

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point_BW6_761 {
            x: BaseField_BW6_761 {
                s: get_fixed_limbs(x),
            },
            y: BaseField_BW6_761 {
                s: get_fixed_limbs(y),
            },
            z: BaseField_BW6_761 {
                s: get_fixed_limbs(z),
            },
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point_BW6_761 {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS_BW6_761, "length must be 3 * {}", BASE_LIMBS_BW6_761);
        Point_BW6_761 {
            x: BaseField_BW6_761 {
                s: value[..BASE_LIMBS_BW6_761].try_into().unwrap(),
            },
            y: BaseField_BW6_761 {
                s: value[BASE_LIMBS_BW6_761..BASE_LIMBS_BW6_761 * 2].try_into().unwrap(),
            },
            z: BaseField_BW6_761 {
                s: value[BASE_LIMBS_BW6_761 * 2..].try_into().unwrap(),
            },
        }
    }

    pub fn to_affine(&self) -> PointAffineNoInfinity_BW6_761 {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity_BW6_761 {
            x: BaseField_BW6_761::from_ark(ark_affine.x.into_repr()),
            y: BaseField_BW6_761::from_ark(ark_affine.y.into_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity_BW6_761 {
        PointAffineNoInfinity_BW6_761 {
            x: self.x,
            y: self.y,
        }
    }
}

impl ScalarField_BW6_761 {
    pub fn from_limbs(value: &[u32]) -> ScalarField_BW6_761 {
        ScalarField_BW6_761 {
            s: get_fixed_limbs(value),
        }
    }
}


#[cfg(test)]
mod tests {
    use ark_bw6_761::{Fr as Fr_BW6_761};

    use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, curves::bw6_761::{Point_BW6_761, ScalarField_BW6_761}};

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField_BW6_761::from_limbs(&limbs);
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
        let left = Point_BW6_761::zero();
        let right = Point_BW6_761::zero();
        assert_eq!(left, right);
        let right = Point_BW6_761::from_limbs(&[0; 12], &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0; 12]);
        assert_eq!(left, right);
        let right = Point_BW6_761::from_limbs(
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0; 12],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert!(left != right);
    }
}