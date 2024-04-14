#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::{
    curve::{Affine, Curve, Projective},
    field::Field,
    traits::{FieldConfig, FieldImpl, GenerateRandom, MontgomeryConvertible},
};
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective};
use icicle_cuda_runtime::{
    device_context::DeviceContext,
    error::CudaResultWrap,
    memory::{DeviceVec, HostSlice},
};

pub fn check_field_equality<F: FieldImpl>() {
    let left = F::zero();
    let right = F::one();
    assert_ne!(left, right);
    let left = F::from_bytes_le(&[1]);
    assert_eq!(left, right);
}

pub fn check_affine_projective_convert<C: Curve>() {
    let size = 1 << 10;
    let affine_points = C::generate_random_affine_points(size);
    let projective_points = C::generate_random_projective_points(size);
    for affine_point in affine_points {
        let projective_eqivalent: Projective<C> = affine_point.into();
        assert_eq!(affine_point, projective_eqivalent.into());
    }
    for projective_point in projective_points {
        let affine_eqivalent: Affine<C> = projective_point.into();
        assert_eq!(projective_point, affine_eqivalent.into());
    }
}

pub fn check_point_equality<const BASE_LIMBS: usize, F: FieldConfig, C>()
where
    C: Curve<BaseField = Field<BASE_LIMBS, F>>,
{
    let left = Projective::<C>::zero();
    let right = Projective::<C>::zero();
    assert_eq!(left, right);
    let right = Projective::<C>::from_limbs([0; BASE_LIMBS], [2; BASE_LIMBS], [0; BASE_LIMBS]);
    assert_eq!(left, right);
    let mut z = [0; BASE_LIMBS];
    z[0] = 2;
    let right = Projective::<C>::from_limbs([0; BASE_LIMBS], [4; BASE_LIMBS], z);
    assert_ne!(left, right);
    let left = Projective::<C>::from_limbs([0; BASE_LIMBS], [2; BASE_LIMBS], C::BaseField::one().into());
    assert_eq!(left, right);
}

#[cfg(feature = "arkworks")]
pub fn check_ark_scalar_convert<F: FieldImpl + ArkConvertible>()
where
    F::Config: GenerateRandom<F>,
    <F as ArkConvertible>::ArkEquivalent: PartialEq + std::fmt::Debug,
{
    let size = 1 << 10;
    let scalars = F::Config::generate_random(size);
    for scalar in scalars {
        assert_eq!(scalar.to_ark(), scalar.to_ark())
    }
}

#[cfg(feature = "arkworks")]
pub fn check_ark_point_convert<C: Curve>()
where
    Affine<C>: ArkConvertible<ArkEquivalent = ArkAffine<C::ArkSWConfig>>,
    Projective<C>: ArkConvertible<ArkEquivalent = ArkProjective<C::ArkSWConfig>>,
{
    let size = 1 << 10;
    let affine_points = C::generate_random_affine_points(size);
    for affine_point in affine_points {
        let ark_projective = Into::<Projective<C>>::into(affine_point).to_ark();
        let ark_affine: ArkAffine<C::ArkSWConfig> = ark_projective.into();
        assert!(ark_affine.is_on_curve());
        assert!(ark_affine.is_in_correct_subgroup_assuming_on_curve());
        let affine_after_conversion = Affine::<C>::from_ark(ark_affine).into();
        assert_eq!(affine_point, affine_after_conversion);
    }
}

pub fn check_field_convert_montgomery<F>()
where
    F: FieldImpl + MontgomeryConvertible<'static>,
    F::Config: GenerateRandom<F>,
{
    let size = 1 << 10;
    let scalars = F::Config::generate_random(size);
    let device_ctx = DeviceContext::default();

    let mut d_scalars = DeviceVec::cuda_malloc(size).unwrap();
    d_scalars
        .copy_from_host(HostSlice::from_slice(&scalars))
        .unwrap();

    F::to_mont(&mut d_scalars, &device_ctx)
        .wrap()
        .unwrap();
    F::from_mont(&mut d_scalars, &device_ctx)
        .wrap()
        .unwrap();

    let mut scalars_copy = vec![F::zero(); size];
    d_scalars
        .copy_to_host(HostSlice::from_mut_slice(&mut scalars_copy))
        .unwrap();

    for (s1, s2) in scalars
        .iter()
        .zip(scalars_copy.iter())
    {
        assert_eq!(s1, s2);
    }
}

pub fn check_points_convert_montgomery<C: Curve>()
where
    Affine<C>: MontgomeryConvertible<'static>,
    Projective<C>: MontgomeryConvertible<'static>,
{
    let size = 1 << 10;
    let device_ctx = DeviceContext::default();

    let affine_points = C::generate_random_affine_points(size);
    let mut d_affine = DeviceVec::cuda_malloc(size).unwrap();
    d_affine
        .copy_from_host(HostSlice::from_slice(&affine_points))
        .unwrap();

    Affine::<C>::to_mont(&mut d_affine, &device_ctx)
        .wrap()
        .unwrap();
    Affine::<C>::from_mont(&mut d_affine, &device_ctx)
        .wrap()
        .unwrap();

    let mut affine_copy = vec![Affine::<C>::zero(); size];
    d_affine
        .copy_to_host(HostSlice::from_mut_slice(&mut affine_copy))
        .unwrap();

    for (p1, p2) in affine_points
        .iter()
        .zip(affine_copy.iter())
    {
        assert_eq!(p1, p2);
    }

    let proj_points = C::generate_random_projective_points(size);
    let mut d_proj = DeviceVec::cuda_malloc(size).unwrap();
    d_proj
        .copy_from_host(HostSlice::from_slice(&proj_points))
        .unwrap();

    Projective::<C>::to_mont(&mut d_proj, &device_ctx)
        .wrap()
        .unwrap();
    Projective::<C>::from_mont(&mut d_proj, &device_ctx)
        .wrap()
        .unwrap();

    let mut projective_copy = vec![Projective::<C>::zero(); size];
    d_proj
        .copy_to_host(HostSlice::from_mut_slice(&mut projective_copy))
        .unwrap();

    for (p1, p2) in proj_points
        .iter()
        .zip(projective_copy.iter())
    {
        assert_eq!(p1, p2);
    }
}
