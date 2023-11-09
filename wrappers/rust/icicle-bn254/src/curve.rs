use icicle_core::curve::{Affine, Projective, CurveConfig};
use icicle_core::field::{Field, FieldConfig};
use std::ffi::{c_void, c_uint};
#[cfg(feature = "arkworks")]
use ark_bn254::{Fr, Fq, g1::Config as ArkG1Config};

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ScalarCfg {}

impl FieldConfig for ScalarCfg {
    #[cfg(feature = "arkworks")]
    type ArkField = Fr;
}

const SCALAR_LIMBS: usize = 8;

pub type ScalarField = Field<SCALAR_LIMBS, ScalarCfg>;

extern "C" {
    fn GenerateScalars(scalars: *mut ScalarField, size: usize);
}

pub(crate) fn generate_random_scalars(size: usize) -> Vec<ScalarField> {
    let mut res = vec![ScalarField::zero(); size];
    unsafe { GenerateScalars(&mut res[..] as *mut _ as *mut ScalarField, size) };
    res
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct BaseCfg {}

impl FieldConfig for BaseCfg {
    #[cfg(feature = "arkworks")]
    type ArkField = Fq;
}

pub const BASE_LIMBS: usize = 8;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct CurveCfg {}

extern "C" {
    fn Eq(point1: *const c_void, point2: *const c_void) -> c_uint;
    fn ToAffine(point: *const c_void, point_out: *mut c_void) -> c_uint;
}

impl CurveConfig for CurveCfg {
    fn eq_proj(point1: *const c_void, point2: *const c_void) -> c_uint {
        unsafe { Eq(point1, point2) }
    }

    fn to_affine(point: *const c_void, point_out: *mut c_void) {
        unsafe { ToAffine(point, point_out) };
    }

    #[cfg(feature = "arkworks")]
    type ArkSWConfig = ArkG1Config;
}

pub type BaseField = Field<BASE_LIMBS, BaseCfg>;
pub type G1Affine = Affine<BaseField, CurveCfg>;
pub type G1Projective = Projective<BaseField, CurveCfg>;

extern "C" {
    // fn Eq(point1: *const G1Projective, point2: *const G1Projective) -> c_uint;
    // fn ToAffine(point: *const G1Projective) -> G1Affine;
    fn GenerateProjectivePoints(points: *mut G1Projective, size: usize);
    fn GenerateAffinePoints(points: *mut G1Affine, size: usize);
}

pub(crate) fn generate_random_projective_points(size: usize) -> Vec<G1Projective> {
    let mut res = vec![G1Projective::zero(); size];
    unsafe { GenerateProjectivePoints(&mut res[..] as *mut _ as *mut G1Projective, size) };
    res
}

pub(crate) fn generate_random_affine_points(size: usize) -> Vec<G1Affine> {
    let mut res = vec![G1Affine::zero(); size];
    unsafe { GenerateAffinePoints(&mut res[..] as *mut _ as *mut G1Affine, size) };
    res
}

#[cfg(test)]
mod tests {
    use super::{
        generate_random_affine_points, generate_random_projective_points, generate_random_scalars, 
        ScalarField, BaseField, G1Affine, G1Projective, BASE_LIMBS,
    };
    use icicle_core::traits::ArkConvertible;

    use ark_bn254::{G1Affine as ArkG1Affine};

    #[test]
    fn test_scalar_equality() {
        let left = ScalarField::zero();
        let right = ScalarField::one();
        assert_ne!(left, right);
        let left = ScalarField::set_limbs(&[1]);
        assert_eq!(left, right);
    }

    #[test]
    fn test_ark_scalar_convert() {
        let size = 1 << 10;
        let scalars = generate_random_scalars(size);
        for scalar in scalars {
            assert_eq!(scalar.to_ark(), scalar.to_ark())
        }
    }

    #[test]
    fn test_affine_projective_convert() {
        let size = 1 << 10;
        let affine_points = generate_random_affine_points(size);
        let projective_points = generate_random_projective_points(size);
        for affine_point in affine_points {
            let projective_eqivalent: G1Projective = affine_point.into();
            assert_eq!(affine_point, projective_eqivalent.into());
        }
        for projective_point in projective_points {
            let affine_eqivalent: G1Affine = projective_point.into();
            assert_eq!(projective_point, affine_eqivalent.into());
        }
    }

    #[test]
    fn test_point_equality() {
        let left = G1Projective::zero();
        let right = G1Projective::zero();
        assert_eq!(left, right);
        let right = G1Projective::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &[0; BASE_LIMBS]);
        assert_eq!(left, right);
        let right = G1Projective::set_limbs(&[0; BASE_LIMBS], &[4; BASE_LIMBS], &BaseField::set_limbs(&[2]).get_limbs());
        assert_ne!(left, right);
        let left = G1Projective::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &BaseField::one().get_limbs());
        assert_eq!(left, right);
    }

    #[test]
    fn test_ark_point_convert() {
        let size = 1 << 10;
        let affine_points = generate_random_affine_points(size);
        for affine_point in affine_points {
            let ark_projective = Into::<G1Projective>::into(affine_point).to_ark();
            let ark_affine: ArkG1Affine = ark_projective.into();
            assert!(ark_affine.is_on_curve());
            assert!(ark_affine.is_in_correct_subgroup_assuming_on_curve());
            let affine_after_conversion: G1Affine = G1Projective::from_ark(ark_projective).into();
            assert_eq!(affine_point, affine_after_conversion);
        }
    }
}
