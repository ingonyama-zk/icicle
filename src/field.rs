use std::mem::transmute_copy;

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use rand::RngCore;

use crate::utils::get_fixed_limbs;

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(C)]
pub struct Field<const NUM_LIMBS: usize> {
    pub(crate) s: [u32; NUM_LIMBS],
}

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
}

pub const BASE_LIMBS: usize = 12;
pub const SCALAR_LIMBS: usize = 8;

#[cfg(feature = "bn254")]
pub const BASE_LIMBS: usize = 8;
#[cfg(feature = "bn254")]
pub const SCALAR_LIMBS: usize = 8;

pub type BaseField = Field<BASE_LIMBS>;
pub type ScalarField = Field<SCALAR_LIMBS>;

impl BaseField {
    pub fn limbs(&self) -> [u32; BASE_LIMBS] {
        self.s
    }
}

impl ScalarField {
    pub fn limbs(&self) -> [u32; SCALAR_LIMBS] {
        self.s
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Point {
    pub(crate) x: BaseField,
    pub(crate) y: BaseField,
    pub(crate) z: BaseField,
}

impl Default for Point {
    fn default() -> Self {
        Point {
            x: BaseField::zero(),
            y: BaseField::one(),
            z: BaseField::zero(),
        }
    }
}

impl Point {
    pub fn zero() -> Self {
        Point {
            x: BaseField::zero(),
            y: BaseField::zero(),
            z: BaseField::zero(),
        }
    }

    pub fn one() -> Self {
        //TODO: ??
        Point {
            x: BaseField::one(),
            y: BaseField::zero(),
            z: BaseField::zero(),
        }
    }

    fn to_ark(&self) -> G1Projective {
        G1Projective::new(
            unsafe { transmute_copy(&self.x) },
            unsafe { transmute_copy(&self.y) },
            unsafe { transmute_copy(&self.z) },
        )
    }

    pub(crate) fn from_ark(ark: G1Projective) -> Point {
        unsafe { transmute_copy(&ark) }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
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

    pub fn from_xy_limbs(value: &[u32]) -> PointAffineNoInfinity {
        let l = value.len();
        assert_eq!(l, 2 * BASE_LIMBS, "length must be 2 * {}", BASE_LIMBS);
        PointAffineNoInfinity {
            x: BaseField {
                s: value[..BASE_LIMBS].try_into().unwrap(),
            },
            y: BaseField {
                s: value[BASE_LIMBS..].try_into().unwrap(),
            },
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        [self.x.limbs(), self.y.limbs()].concat()
    }

    pub fn to_projective(&self) -> Point {
        //TODO: !!!review this!!!
        Point {
            x: self.x,
            y: self.y,
            z: BaseField::one(),
        }
    }

    fn to_ark(&self) -> G1Affine {
        G1Affine::new(
            unsafe { transmute_copy(&self.x) },
            unsafe { transmute_copy(&self.y) },
            false,
        )
    }
}

impl Point {
    // TODO: generics
    ///From u32 limbs x,y
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
        //assert_eq!(self.z, BaseField::one()); //TODO: !!!review this!!!
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

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Scalar {
    pub s: ScalarField, //TODO: do we need this wrapping struct?
}

impl Scalar {
    pub fn from_limbs(value: &[u32]) -> Scalar {
        Scalar {
            s: ScalarField::from_limbs(value),
        }
    }

    pub fn limbs(&self) -> Vec<u32> {
        self.s.limbs().to_vec()
    }

    pub fn one() -> Self {
        Scalar {
            s: ScalarField::one(),
        }
    }

    pub fn zero() -> Self {
        Scalar {
            s: ScalarField::zero(),
        }
    }
}

impl Scalar {
    pub fn to_ark(&self) -> Fr {
        unsafe { std::mem::transmute(*self) }
    }

    pub(crate) fn from_ark(v: Fr) -> Scalar {
        unsafe { std::mem::transmute(v) }
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Scalar {
            s: ScalarField::zero(),
        }
    }
}

pub fn generate_random_points(
    count: usize,
    mut rng: Box<dyn RngCore>,
) -> Vec<PointAffineNoInfinity> {
    (0..count)
        .map(|_| {
            //TODO: replace with CUDA?
            let mut var_name = [rng.next_u32(); 2 * BASE_LIMBS];
            var_name[0] = 0x01u32; //x first byte = 1
            var_name[BASE_LIMBS - 1] = 0xfau32;
            var_name[BASE_LIMBS] = 0x02u32; //x first byte = 2
            var_name[2 * BASE_LIMBS - 1] = 0xfbu32;
            PointAffineNoInfinity::from_xy_limbs(&var_name)
        })
        .collect()
}

pub fn generate_random_scalars(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Scalar> {
    (0..count)
        .map(|_| Scalar {
            s: ScalarField {
                s: [rng.next_u32(); SCALAR_LIMBS],
            },
        })
        .collect()
}
