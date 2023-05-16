use std::ffi::c_uint;
use std::ops::Mul;

// TODO: change curve here based on the features
use ark_bls12_381::{Fq, Fq2, G1Affine, G1Projective, G2Affine, G2Projective,
    g1::Parameters as G1Parameters, g2::Parameters as G2Parameters};
use ark_ec::{AffineCurve, ProjectiveCurve, SWModelParameters,
    models::short_weierstrass_jacobian::{GroupAffine, GroupProjective}};
use ark_ff::{BigInteger384, BigInteger256, PrimeField, Field};

use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

use crate::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};


fn bytes_to_u32_vector(bytes: &[u8]) -> Vec<u32> {
    // Ensure the byte array length is divisible by 4
    assert_eq!(bytes.len() % 4, 0);
    let mut result = Vec::with_capacity(bytes.len() / 4);
    for i in (0..bytes.len()).step_by(4) {
        let value = (bytes[i] as u32) << 24
            | (bytes[i + 1] as u32) << 16
            | (bytes[i + 2] as u32) << 8
            | (bytes[i + 3] as u32);
        result.push(value);
    }
    result
}

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

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct FiniteField<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for FiniteField<NUM_LIMBS> {}

pub trait LimbsField {
    fn zero() -> Self;
    fn one() -> Self;
    fn limbs(&self) -> Vec<u32>;
    fn from_limbs(value: &[u32]) -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
}

impl<const NUM_LIMBS: usize> LimbsField for FiniteField<NUM_LIMBS> {
    fn zero() -> Self {
        FiniteField {
            s: [0u32; NUM_LIMBS],
        }
    }

    fn one() -> Self {
        let mut s: [u32; NUM_LIMBS] = [0u32; NUM_LIMBS];
        s[0] = 1;
        FiniteField { s }
    }

    fn limbs(&self) -> Vec<u32> {
        self.s.to_vec()
    }

    fn from_limbs(value: &[u32]) -> Self {
        Self {
            s: get_fixed_limbs::<NUM_LIMBS>(value),
        }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

impl<const NUM_LIMBS: usize> Default for FiniteField<NUM_LIMBS> {
    fn default() -> Self {
        FiniteField::zero()
    }
}

pub const BASE_LIMBS: usize = 12;
pub const SCALAR_LIMBS: usize = 8;
pub const EXTENSION_LIMBS: usize = 2 * BASE_LIMBS;

#[cfg(feature = "bn254")]
pub const BASE_LIMBS: usize = 8;
#[cfg(feature = "bn254")]
pub const SCALAR_LIMBS: usize = 8;
#[cfg(feature = "bn254")]
pub const EXTENSION_LIMBS: usize = 2 * BASE_LIMBS;

pub type BaseField = FiniteField<BASE_LIMBS>;
pub type ScalarField = FiniteField<SCALAR_LIMBS>;
pub type ExtensionField = FiniteField<EXTENSION_LIMBS>;

pub trait ArkConvertible {
    type ArkEquivalent;

    fn to_ark(&self) -> Self::ArkEquivalent;
    fn from_ark(ark: &Self::ArkEquivalent) -> Self;
}

impl ArkConvertible for BaseField {
    type ArkEquivalent = Fq;

    fn to_ark(&self) -> Fq {
        Fq::from_repr(BigInteger384::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())).unwrap()
    }

    fn from_ark(ark: &Fq) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0.0))
    }
}

impl ArkConvertible for ScalarField {
    type ArkEquivalent = BigInteger256;

    fn to_ark(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    fn from_ark(ark: &BigInteger256) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }
}

impl ArkConvertible for ExtensionField {
    type ArkEquivalent = Fq2;

    fn to_ark(&self) -> Fq2 {
        let c0 = BaseField::from_limbs(&self.s[..BASE_LIMBS]).to_ark();
        let c1 = BaseField::from_limbs(&self.s[BASE_LIMBS..]).to_ark();
        Fq2::new(c0, c1)
    }

    fn from_ark(ark: &Fq2) -> Self {
        let re_part = get_fixed_limbs::<BASE_LIMBS>(&u64_vec_to_u32_vec(&ark.c0.0.0));
        let im_part = get_fixed_limbs::<BASE_LIMBS>(&u64_vec_to_u32_vec(&ark.c1.0.0));
        Self::from_limbs(&[re_part, im_part].concat())
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity<F> {
    pub x: F,
    pub y: F,
}

impl<F: LimbsField> Default for PointAffineNoInfinity<F> {
    fn default() -> Self {
        PointAffineNoInfinity {
            x: F::zero(),
            y: F::zero(),
        }
    }
}

impl<F: ArkConvertible + LimbsField + Copy> PointAffineNoInfinity<F> {
    // From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity {
            x: F::from_limbs(x),
            y: F::from_limbs(y),
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point<F> {
        Point {
            x: self.x,
            y: self.y,
            z: F::one(),
        }
    }

    fn to_ark_internal<ArkParameters: SWModelParameters<BaseField = F::ArkEquivalent>>(&self) -> GroupAffine<ArkParameters> {
        GroupAffine::<ArkParameters>::new(self.x.to_ark(), self.y.to_ark(), false)
    }
    
    fn from_ark_internal<ArkParameters: SWModelParameters<BaseField = F::ArkEquivalent>>(aff_ark: &GroupAffine<ArkParameters>) -> Self {
        PointAffineNoInfinity {
            x: F::from_ark(&aff_ark.x),
            y: F::from_ark(&aff_ark.y),
        }
    }
}

impl ArkConvertible for PointAffineNoInfinity<BaseField> {
    type ArkEquivalent = G1Affine;

    fn to_ark(&self) -> Self::ArkEquivalent {
        self.to_ark_internal::<G1Parameters>()
    }

    fn from_ark(ark: &Self::ArkEquivalent) -> Self {
        Self::from_ark_internal::<G1Parameters>(ark)
    }
}

impl ArkConvertible for PointAffineNoInfinity<ExtensionField> {
    type ArkEquivalent = G2Affine;

    fn to_ark(&self) -> Self::ArkEquivalent {
        self.to_ark_internal::<G2Parameters>()
    }

    fn from_ark(ark: &Self::ArkEquivalent) -> Self {
        Self::from_ark_internal::<G2Parameters>(ark)
    }
}

#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point<F> {
    pub x: F,
    pub y: F,
    pub z: F,
}

impl<F: ArkConvertible<ArkEquivalent = impl Field> + LimbsField + Copy> Point<F> {
    pub fn zero() -> Self {
        Point {
            x: F::zero(),
            y: F::one(),
            z: F::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point {
            x: F::from_limbs(x),
            y: F::from_limbs(y),
            z: F::from_limbs(z),
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Self {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS, "length must be 3 * {}", BASE_LIMBS);
        Point {
            x: F::from_limbs(&value[..BASE_LIMBS]),
            y: F::from_limbs(&value[BASE_LIMBS..BASE_LIMBS * 2]),
            z: F::from_limbs(&value[BASE_LIMBS * 2..]),
        }
    }

    fn to_ark_affine_internal<ArkParameters: SWModelParameters<BaseField = F::ArkEquivalent>>(&self)-> GroupAffine<ArkParameters> {
        let proj_x_field = self.x.to_ark();
        let proj_y_field = self.y.to_ark();
        let proj_z_field = self.z.to_ark();
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        GroupAffine::<ArkParameters>::new(aff_x, aff_y, false)
    }
    
    fn to_ark_internal<ArkParameters: SWModelParameters<BaseField = F::ArkEquivalent>>(&self) -> GroupProjective<ArkParameters> {
        self.to_ark_affine_internal::<ArkParameters>().into_projective()
    }
    
    fn from_ark_internal<ArkParameters: SWModelParameters<BaseField = F::ArkEquivalent>>(ark_proj: &GroupProjective<ArkParameters>) -> Self {
        let ark_affine = ark_proj.into_affine();
        Self {
            x: F::from_ark(&ark_affine.x),
            y: F::from_ark(&ark_affine.y),
            z: F::one(),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity<F> {
        PointAffineNoInfinity {
            x: self.x,
            y: self.y,
        }
    }
}

impl ArkConvertible for Point<BaseField> {
    type ArkEquivalent = G1Projective;

    fn to_ark(&self) -> Self::ArkEquivalent {
        self.to_ark_internal::<G1Parameters>()
    }

    fn from_ark(ark: &Self::ArkEquivalent) -> Self {
        Self::from_ark_internal::<G1Parameters>(ark)
    }
}

impl<F: ArkConvertible<ArkEquivalent = impl Field> + LimbsField + Copy> Default for Point<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl Point<BaseField> {
    pub fn to_ark_affine(&self) -> G1Affine {
        self.to_ark_affine_internal::<G1Parameters>()
    }
}

impl ArkConvertible for Point<ExtensionField> {
    type ArkEquivalent = G2Projective;

    fn to_ark(&self) -> Self::ArkEquivalent {
        self.to_ark_internal::<G2Parameters>()
    }

    fn from_ark(ark_proj: &Self::ArkEquivalent) -> Self {
        Self::from_ark_internal::<G2Parameters>(ark_proj)
    }
}

impl Point<ExtensionField> {
    pub fn to_ark_affine(&self) -> G2Affine {
        self.to_ark_affine_internal::<G2Parameters>()
    }
}

extern "C" {
    fn eq(point1: *const Point<BaseField>, point2: *const Point<BaseField>) -> c_uint;
}

impl PartialEq for Point<BaseField> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq(self, other) != 0 }
    }
}


#[cfg(test)]
pub(crate) mod tests {
    use ark_bls12_381::Fr;
    use ark_ff::BigInteger256;
    use std::mem::transmute;

    use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, field::{Point, ScalarField, LimbsField, ArkConvertible}};

    fn to_ark_transmute(v: &ScalarField) -> BigInteger256 {
        unsafe { transmute(*v) }
    }
    
    pub fn from_ark_transmute(v: BigInteger256) -> ScalarField {
        unsafe { transmute(v) }
    }

    #[test]
    fn test_ark_scalar_convert() {
        let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
        let scalar = ScalarField::from_limbs(&limbs);
        assert_eq!(
            scalar.to_ark(),
            to_ark_transmute(&scalar),
            "{:08X?} {:08X?}",
            scalar.to_ark(),
            to_ark_transmute(&scalar)
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
