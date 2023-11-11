use crate::field::{Field, FieldConfig};
use std::marker::PhantomData;
use std::ffi::{c_void, c_uint};
#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{SWCurveConfig, Projective as ArkProjective, Affine as ArkAffine};
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;

pub trait CurveConfig: PartialEq + Copy + Clone {
    fn eq_proj(point1: *const c_void, point2: *const c_void) -> c_uint;
    fn to_affine(point: *const c_void, point_aff: *mut c_void);

    #[cfg(feature = "arkworks")]
    type ArkSWConfig: SWCurveConfig;
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Projective<T, C: CurveConfig> {
    pub x: T,
    pub y: T,
    pub z: T,
    p: PhantomData<C>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub struct Affine<T, C: CurveConfig> {
    pub x: T,
    pub y: T,
    p: PhantomData<C>,
}

impl<const NUM_LIMBS: usize, F, C> Affine<Field<NUM_LIMBS, F>, C> 
where F: FieldConfig, C: CurveConfig {
    // While this is not a true zero point and not even a valid point, it's still useful 
    // both as a handy default as well as a representation of zero points in other codebases
    pub fn zero() -> Self {
        Affine {
            x: Field::<NUM_LIMBS, F>::zero(),
            y: Field::<NUM_LIMBS, F>::zero(),
            p: PhantomData,
        }
    }

    pub fn set_limbs(x: &[u32], y: &[u32]) -> Self {
        Affine {
            x: Field::<NUM_LIMBS, F>::set_limbs(x),
            y: Field::<NUM_LIMBS, F>::set_limbs(y),
            p: PhantomData,
        }
    }

    pub fn to_projective(&self) -> Projective<Field<NUM_LIMBS, F>, C> {
        Projective {
            x: self.x,
            y: self.y,
            z: Field::<NUM_LIMBS, F>::one(),
            p: PhantomData,
        }
    }
}

impl<const NUM_LIMBS: usize, F, C> From<Affine<Field<NUM_LIMBS, F>, C>> for Projective<Field<NUM_LIMBS, F>, C> 
where F: FieldConfig, C: CurveConfig {
    fn from(item: Affine<Field<NUM_LIMBS, F>, C>) -> Self {
        Self {
            x: item.x,
            y: item.y,
            z: Field::<NUM_LIMBS, F>::one(),
            p: PhantomData,
        }
    }
}

impl<const NUM_LIMBS: usize, F, C> Projective<Field<NUM_LIMBS, F>, C>
where F: FieldConfig, C: CurveConfig {
    pub fn zero() -> Self {
        Projective {
            x: Field::<NUM_LIMBS, F>::zero(),
            y: Field::<NUM_LIMBS, F>::one(),
            z: Field::<NUM_LIMBS, F>::zero(),
            p: PhantomData,
        }
    }

    pub fn set_limbs(x: &[u32], y: &[u32], z: &[u32]) -> Self {
        Projective {
            x: Field::<NUM_LIMBS, F>::set_limbs(x),
            y: Field::<NUM_LIMBS, F>::set_limbs(y),
            z: Field::<NUM_LIMBS, F>::set_limbs(z),
            p: PhantomData,
        }
    }
}

impl<const NUM_LIMBS: usize, F, C> PartialEq for Projective<Field<NUM_LIMBS, F>, C>
where F: FieldConfig, C: CurveConfig {
    fn eq(&self, other: &Self) -> bool {
        C::eq_proj(self as *const _ as *const c_void, other as *const _ as *const c_void) != 0
    }
}

impl<const NUM_LIMBS: usize, F, C> From<Projective<Field<NUM_LIMBS, F>, C>> for Affine<Field<NUM_LIMBS, F>, C>
where F: FieldConfig, C: CurveConfig {
    fn from(item: Projective<Field<NUM_LIMBS, F>, C>) -> Self {
        let mut aff = Self::zero();
        C::to_affine(&item as *const _ as *const c_void, &mut aff as *mut _ as *mut c_void);
        aff
    }
}

#[cfg(feature = "arkworks")]
impl<const NUM_LIMBS: usize, F, C> ArkConvertible for Affine<Field<NUM_LIMBS, F>, C>
where C: CurveConfig, F: FieldConfig<ArkField = <<C as CurveConfig>::ArkSWConfig as ArkCurveConfig>::BaseField> {
    type ArkEquivalent = ArkAffine<C::ArkSWConfig>;

    fn to_ark(&self) -> Self::ArkEquivalent {
        let proj_x = self.x.to_ark();
        let proj_y = self.y.to_ark();
        Self::ArkEquivalent::new_unchecked(proj_x, proj_y)
    }

    fn from_ark(ark: Self::ArkEquivalent) -> Self {
        Self {
            x: Field::<NUM_LIMBS, F>::from_ark(ark.x),
            y: Field::<NUM_LIMBS, F>::from_ark(ark.y),
            p: PhantomData,
        }
    }
}

#[cfg(feature = "arkworks")]
impl<const NUM_LIMBS: usize, F, C> ArkConvertible for Projective<Field<NUM_LIMBS, F>, C>
where C: CurveConfig, F: FieldConfig<ArkField = <<C as CurveConfig>::ArkSWConfig as ArkCurveConfig>::BaseField> {
    type ArkEquivalent = ArkProjective<C::ArkSWConfig>;

    fn to_ark(&self) -> Self::ArkEquivalent {
        let proj_x = self.x.to_ark();
        let proj_y = self.y.to_ark();
        let proj_z = self.z.to_ark();

        // conversion between projective used in icicle and Jacobian used in arkworks
        let proj_x = proj_x * proj_z;
        let proj_y = proj_y * proj_z * proj_z;
        Self::ArkEquivalent::new_unchecked(proj_x, proj_y, proj_z)
    }

    fn from_ark(ark: Self::ArkEquivalent) -> Self {
        // conversion between Jacobian used in arkworks and projective used in icicle
        let proj_x = ark.x * ark.z;
        let proj_z = ark.z * ark.z * ark.z;
        Self {
            x: Field::<NUM_LIMBS, F>::from_ark(proj_x),
            y: Field::<NUM_LIMBS, F>::from_ark(ark.y),
            z: Field::<NUM_LIMBS, F>::from_ark(proj_z),
            p: PhantomData,
        }
    }
}
