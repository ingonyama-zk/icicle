use std::ffi::c_uint;
use std::fmt::Debug;
use std::marker;
use std::marker::PhantomData;
use std::mem::transmute;


use halo2curves::{
    bn256::{Fq as Fq_BN254_PSE, Fq, G1, G1Affine as G1Affine_BN254_PSE},
    CurveAffine,
    group::{ff::{Field, PrimeField}, Group, prime::PrimeCurveAffine},
    serde::SerdeObject
};
use halo2curves::bn256::G1Compressed;
use halo2curves::group::{Curve, GroupEncoding, UncompressedEncoding};
use halo2curves::pairing::Engine;
use rustacuda_core::DeviceCopy;
use rustacuda_derive::DeviceCopy;

use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};
use crate::test_bn254_pse::{generate_random_points_bn254, msm_bn254};
use crate::utils::get_rng;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field_BN254_PSE<const NUM_LIMBS: usize> {
    pub s: [u32; NUM_LIMBS],
}

unsafe impl<const NUM_LIMBS: usize> DeviceCopy for Field_BN254_PSE<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> Default for Field_BN254_PSE<NUM_LIMBS> {
    fn default() -> Self {
        Field_BN254_PSE::zero()
    }
}

impl<const NUM_LIMBS: usize> Field_BN254_PSE<NUM_LIMBS> {
    pub fn zero() -> Self {
        Field_BN254_PSE {
            s: [0u32; NUM_LIMBS],
        }
    }

    pub fn one() -> Self {
        let mut s = [0u32; NUM_LIMBS];
        s[0] = 1;
        Field_BN254_PSE { s }
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        self.s
            .iter()
            .map(|s| s.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<_>>()
    }
}

pub const BASE_LIMBS_BN254_PSE: usize = 8;
pub const SCALAR_LIMBS_BN254_PSE: usize = 8;

#[allow(non_camel_case_types)]
pub type BaseField_BN254_PSE = Field_BN254_PSE<BASE_LIMBS_BN254_PSE>;
#[allow(non_camel_case_types)]
pub type ScalarField_BN254_PSE = Field_BN254_PSE<SCALAR_LIMBS_BN254_PSE>;

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

impl ScalarField_BN254_PSE {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS_BN254_PSE] {
        self.s
    }

    /// Result is little-end sort
    pub fn to_pse(&self) -> [u8; 32] {
        Fq::from_raw(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap()).to_bytes()
    }

    pub fn from_pse(fq_bytes: [u8; 32])->Self  {
        let mut u64_array: [u64; 4] = [0; 4];
        for i in 0..4 {
            for j in 0..8 {
                u64_array[i] |= (fq_bytes[i * 8 + j] as u64) << (j * 8);
            }
        }
        Self::from_limbs(&u64_vec_to_u32_vec(&u64_array))
    }

    pub fn to_pse_transmute(&self) -> [u8;32] {
        unsafe { transmute(*self) }
    }

    pub fn from_pse_transmute(v: [u8;32]) -> ScalarField_BN254_PSE {
        unsafe { transmute(v) }
    }
}

#[derive(Debug, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct Point_BN254_PSE {
    pub x: BaseField_BN254_PSE,
    pub y: BaseField_BN254_PSE,
    pub z: BaseField_BN254_PSE,
}

impl Default for Point_BN254_PSE {
    fn default() -> Self {
        Point_BN254_PSE::zero()
    }
}

impl Point_BN254_PSE {
    pub fn zero() -> Self {
        Point_BN254_PSE {
            x: BaseField_BN254_PSE::zero(),
            y: BaseField_BN254_PSE::one(),
            z: BaseField_BN254_PSE::zero(),
        }
    }

    pub fn infinity() -> Self {
        Self::zero()
    }

    pub fn to_pse(&self) -> G1 {
        self.to_pse_affine().to_curve()
    }

    pub fn to_pse_affine(&self) -> G1Affine_BN254_PSE {
        use std::ops::Mul;
        let proj_x_field = Fq_BN254_PSE::from_raw_bytes(&self.x.to_bytes_le()).unwrap();
        let proj_y_field = Fq_BN254_PSE::from_raw_bytes(&self.y.to_bytes_le()).unwrap();
        let proj_z_field = Fq_BN254_PSE::from_raw_bytes(&self.z.to_bytes_le()).unwrap();
        let inverse_z = proj_z_field.invert().unwrap();
        let aff_x = proj_x_field.mul(&inverse_z);
        let aff_y = proj_y_field.mul(&inverse_z);
        G1Affine_BN254_PSE::from_xy(aff_x,aff_y).unwrap()
    }

    pub fn from_pse(projective_point: G1) -> Point_BN254_PSE {
        let z_inv = projective_point.z.invert().unwrap();
        let z_invsq = z_inv* z_inv;
        let z_invq3 = z_invsq * z_inv;

        Point_BN254_PSE {
            x: BaseField_BN254_PSE::from_pse((projective_point.x * z_invsq).to_repr()),
            y: BaseField_BN254_PSE::from_pse((projective_point.y * z_invq3).to_repr()),
            z: BaseField_BN254_PSE::one(),
        }
    }
}

extern "C" {
    fn eq_bn254(point1: *const Point_BN254_PSE, point2: *const Point_BN254_PSE) -> c_uint;
}

impl PartialEq for Point_BN254_PSE {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_bn254(self, other) != 0 }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, DeviceCopy)]
#[repr(C)]
pub struct PointAffineNoInfinity_BN254_PSE {
    pub x: BaseField_BN254_PSE,
    pub y: BaseField_BN254_PSE,
}

impl Default for PointAffineNoInfinity_BN254_PSE {
    fn default() -> Self {
        PointAffineNoInfinity_BN254_PSE {
            x: BaseField_BN254_PSE::zero(),
            y: BaseField_BN254_PSE::zero(),
        }
    }
}

impl PointAffineNoInfinity_BN254_PSE {
    ///From u32 limbs x,y
    pub fn from_limbs(x: &[u32], y: &[u32]) -> Self {
        PointAffineNoInfinity_BN254_PSE {
            x: BaseField_BN254_PSE {
                s: get_fixed_limbs(x),
            },
            y: BaseField_BN254_PSE {
                s: get_fixed_limbs(y),
            },
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point_BN254_PSE {
        Point_BN254_PSE {
            x: self.x,
            y: self.y,
            z: BaseField_BN254_PSE::one(),
        }
    }

    pub fn to_pse(&self) -> G1Affine_BN254_PSE {
        G1Affine_BN254_PSE::from_xy(Fq_BN254_PSE::from_bytes(&self.x.to_pse()).unwrap(), Fq_BN254_PSE::from_bytes(&self.y.to_pse()).unwrap()).unwrap()
    }

    pub fn to_pse_repr(&self) -> G1Affine_BN254_PSE {
        G1Affine_BN254_PSE::from_xy(
            Fq_BN254_PSE::from_repr(self.x.to_pse()).unwrap(),
            Fq_BN254_PSE::from_repr(self.y.to_pse()).unwrap(),
        ).unwrap()
    }

    pub fn from_pse(p: &G1Affine_BN254_PSE) -> Self {
        PointAffineNoInfinity_BN254_PSE {
            x: BaseField_BN254_PSE::from_pse(p.x.to_repr()),
            y: BaseField_BN254_PSE::from_pse(p.y.to_repr()),
        }
    }
}

impl Point_BN254_PSE {
    // TODO: generics

    pub fn from_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Point_BN254_PSE {
            x: BaseField_BN254_PSE {
                s: get_fixed_limbs(x),
            },
            y: BaseField_BN254_PSE {
                s: get_fixed_limbs(y),
            },
            z: BaseField_BN254_PSE {
                s: get_fixed_limbs(z),
            },
        }
    }

    pub fn from_xy_limbs(value: &[u32]) -> Point_BN254_PSE {
        let l = value.len();
        assert_eq!(l, 3 * BASE_LIMBS_BN254_PSE, "length must be 3 * {}", BASE_LIMBS_BN254_PSE);
        Point_BN254_PSE {
            x: BaseField_BN254_PSE {
                s: value[..BASE_LIMBS_BN254_PSE].try_into().unwrap(),
            },
            y: BaseField_BN254_PSE {
                s: value[BASE_LIMBS_BN254_PSE..BASE_LIMBS_BN254_PSE * 2].try_into().unwrap(),
            },
            z: BaseField_BN254_PSE {
                s: value[BASE_LIMBS_BN254_PSE * 2..].try_into().unwrap(),
            },
        }
    }

    pub fn to_affine(&self) -> PointAffineNoInfinity_BN254_PSE {
        let pse_affine = self.to_pse_affine();
        PointAffineNoInfinity_BN254_PSE {
            x: BaseField_BN254_PSE::from_pse(pse_affine.x.to_repr()),
            y: BaseField_BN254_PSE::from_pse(pse_affine.y.to_repr()),
        }
    }

    pub fn to_xy_strip_z(&self) -> PointAffineNoInfinity_BN254_PSE {
        PointAffineNoInfinity_BN254_PSE {
            x: self.x,
            y: self.y,
        }
    }
}

impl ScalarField_BN254_PSE {
    pub fn from_limbs(value: &[u32]) -> ScalarField_BN254_PSE {
        ScalarField_BN254_PSE {
            s: get_fixed_limbs(value),
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct MSMENTRY<E: Engine> {
    pub(crate) scalars: Vec<ScalarField_BN254_PSE>,
    pub(crate) bases: Vec<PointAffineNoInfinity_BN254_PSE>,
    _marker:PhantomData<E>
}

impl<E: Engine + Debug> MSMENTRY<E>{
    pub fn new(bases: &Vec<E::G1Affine>,scalars: &Vec<E::Scalar>)->Self{
        let bases:Vec<_> = bases.iter().map(|x|{
            let mut encoding = <G1Affine_BN254_PSE as GroupEncoding>::Repr::default();
            encoding.as_mut().copy_from_slice(x.to_bytes().as_ref());
            let affine_point = G1Affine_BN254_PSE::from_bytes(&encoding).unwrap();
            PointAffineNoInfinity_BN254_PSE::from_pse(&affine_point)
        }).collect();
        let scalars:Vec<_> = scalars.iter().map(|x|{
            let scalar_ref:[u8;32] = x.to_repr().as_ref().try_into().unwrap();
            ScalarField_BN254_PSE::from_pse(scalar_ref)
        }).collect();
        MSMENTRY{
            scalars,
            bases,
            _marker: PhantomData,
        }
    }

    pub fn msm_bn254(&self)-> E::G1{
    let projective_point = msm_bn254(&self.bases,&self.scalars,0).to_pse().to_bytes();
        let projective_point_ref = projective_point.as_ref();
        let mut encoding = <E::G1 as GroupEncoding>::Repr::default();
        encoding.as_mut().copy_from_slice(projective_point_ref);
        E::G1::from_bytes(&encoding).unwrap()
    }
}


// #[cfg(test)]
// mod tests {
//     use ark_bn254_pse::{Fr as Fr_BN254_PSE};
//
//     use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}, curves::bn254_pse::{Point_BN254_PSE, ScalarField_BN254_PSE}};
//
//     #[test]
//     fn test_ark_scalar_convert() {
//         let limbs = [0x0fffffff, 1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7];
//         let scalar = ScalarField_BN254_PSE::from_limbs(&limbs);
//         assert_eq!(
//             scalar.to_ark(),
//             scalar.to_ark_transmute(),
//             "{:08X?} {:08X?}",
//             scalar.to_ark(),
//             scalar.to_ark_transmute()
//         )
//     }
//
//     #[test]
//     #[allow(non_snake_case)]
//     fn test_point_equality() {
//         let left = Point_BN254_PSE::zero();
//         let right = Point_BN254_PSE::zero();
//         assert_eq!(left, right);
//         let right = Point_BN254_PSE::from_limbs(&[0; 8], &[2, 0, 0, 0, 0, 0, 0, 0], &[0; 8]);
//         assert_eq!(left, right);
//         let right = Point_BN254_PSE::from_limbs(
//             &[2, 0, 0, 0, 0, 0, 0, 0],
//             &[0; 8],
//             &[1, 0, 0, 0, 0, 0, 0, 0],
//         );
//         assert!(left != right);
//     }
// }

#[test]
fn test_g1_projective_point(){
    let seed = None;
    let projective_point = G1::random(get_rng(seed));
    let affine_point = projective_point.to_affine().to_bytes();
    let projective_point_rec = G1::from_bytes(&affine_point).unwrap();

    assert_eq!(projective_point,projective_point_rec);
}

#[test]
fn test_big(){
    let test = Fq::from_raw([1,0,0,0]);
    let u8_array = &test.to_bytes();
    let mut u64_array: [u64; 4] = [0; 4];
    for i in 0..4 {
        for j in 0..8 {
            u64_array[i] |= (u8_array[i * 8 + j] as u64) << (j * 8);
        }
    }
    let res = u64_vec_to_u32_vec(&u64_array);
    assert_eq!(res,vec![1, 0, 0, 0, 0, 0, 0, 0])
}

#[test]
fn test_singel(){
    let seed = None;
    let points = generate_random_points_bn254(1, get_rng(seed));
}