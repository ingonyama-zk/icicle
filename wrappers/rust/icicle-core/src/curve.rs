use crate::field::{Field, FieldConfig};
#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective, SWCurveConfig};
use std::ffi::{c_uint, c_void};
use std::marker::PhantomData;

pub trait CurveConfig: PartialEq + Copy + Clone {
    fn eq_proj(point1: *const c_void, point2: *const c_void) -> c_uint;
    fn to_affine(point: *const c_void, point_aff: *mut c_void);

    type BaseField;
    type ScalarField;
    type Projective;
    type Affine;

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
where
    F: FieldConfig,
    C: CurveConfig,
{
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
where
    F: FieldConfig,
    C: CurveConfig,
{
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
where
    F: FieldConfig,
    C: CurveConfig,
{
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
where
    F: FieldConfig,
    C: CurveConfig,
{
    fn eq(&self, other: &Self) -> bool {
        C::eq_proj(self as *const _ as *const c_void, other as *const _ as *const c_void) != 0
    }
}

impl<const NUM_LIMBS: usize, F, C> From<Projective<Field<NUM_LIMBS, F>, C>> for Affine<Field<NUM_LIMBS, F>, C>
where
    F: FieldConfig,
    C: CurveConfig,
{
    fn from(item: Projective<Field<NUM_LIMBS, F>, C>) -> Self {
        let mut aff = Self::zero();
        C::to_affine(&item as *const _ as *const c_void, &mut aff as *mut _ as *mut c_void);
        aff
    }
}

#[cfg(feature = "arkworks")]
impl<const NUM_LIMBS: usize, F, C> ArkConvertible for Affine<Field<NUM_LIMBS, F>, C>
where
    C: CurveConfig,
    F: FieldConfig<ArkField = <<C as CurveConfig>::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    type ArkEquivalent = ArkAffine<C::ArkSWConfig>;

    fn to_ark(&self) -> Self::ArkEquivalent {
        let proj_x = self
            .x
            .to_ark();
        let proj_y = self
            .y
            .to_ark();
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
where
    C: CurveConfig,
    F: FieldConfig<ArkField = <<C as CurveConfig>::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    type ArkEquivalent = ArkProjective<C::ArkSWConfig>;

    fn to_ark(&self) -> Self::ArkEquivalent {
        let proj_x = self
            .x
            .to_ark();
        let proj_y = self
            .y
            .to_ark();
        let proj_z = self
            .z
            .to_ark();

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

#[macro_export]
macro_rules! impl_curve {
    (
        $scalar_limbs:ident,
        $base_limbs:ident,
    ) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct ScalarCfg {}

        impl FieldConfig for ScalarCfg {
            #[cfg(feature = "arkworks")]
            type ArkField = Fr;
        }

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

        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct CurveCfg {}

        pub type BaseField = Field<$base_limbs, BaseCfg>;
        pub type G1Affine = Affine<BaseField, CurveCfg>;
        pub type G1Projective = Projective<BaseField, CurveCfg>;

        extern "C" {
            fn Eq(point1: *const c_void, point2: *const c_void) -> c_uint;
            fn ToAffine(point: *const c_void, point_out: *mut c_void);
            fn GenerateProjectivePoints(points: *mut G1Projective, size: usize);
            fn GenerateAffinePoints(points: *mut G1Affine, size: usize);
        }

        impl CurveConfig for CurveCfg {
            type BaseField = BaseField;
            type ScalarField = ScalarField;
            type Affine = G1Affine;
            type Projective = G1Projective;

            fn eq_proj(point1: *const c_void, point2: *const c_void) -> c_uint {
                unsafe { Eq(point1, point2) }
            }

            fn to_affine(point: *const c_void, point_out: *mut c_void) {
                unsafe { ToAffine(point, point_out) };
            }

            #[cfg(feature = "arkworks")]
            type ArkSWConfig = ArkG1Config;
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
    };
}

#[macro_export]
macro_rules! impl_curve_tests {
    () => {
        #[test]
        fn test_scalar_equality() {
            let left = ScalarField::zero();
            let right = ScalarField::one();
            assert_ne!(left, right);
            let left = ScalarField::set_limbs(&[1]);
            assert_eq!(left, right);
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
            let right = G1Projective::set_limbs(
                &[0; BASE_LIMBS],
                &[4; BASE_LIMBS],
                &BaseField::set_limbs(&[2]).get_limbs(),
            );
            assert_ne!(left, right);
            let left = G1Projective::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &BaseField::one().get_limbs());
            assert_eq!(left, right);
        }
    };
}

#[macro_export]
macro_rules! impl_curve_ark_tests {
    () => {
        #[test]
        fn test_ark_scalar_convert() {
            let size = 1 << 10;
            let scalars = generate_random_scalars(size);
            for scalar in scalars {
                assert_eq!(scalar.to_ark(), scalar.to_ark())
            }
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
    };
}
