use crate::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    traits::{Arithmetic, GenerateRandom, MontgomeryConvertible},
};
use icicle_runtime::{
    memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
    stream::IcicleStream,
};

pub fn check_field_arithmetic<F>()
where
    F: PrimeField + Arithmetic + GenerateRandom,
{
    let size = 1 << 10;
    let scalars_a = F::generate_random(size);
    let scalars_b = F::generate_random(size);

    for i in 0..size {
        let result1 = scalars_a[i] + scalars_b[i];
        let result2 = result1 - scalars_b[i];
        assert_eq!(result2, scalars_a[i]);
    }

    // Test field multiplication API
    let scalar_a = scalars_a[0];
    let square = scalar_a.sqr();
    let mul_by_self = scalar_a.mul(scalar_a);
    assert_eq!(square, mul_by_self);

    // Test field pow API
    let pow_4 = scalar_a.pow(4);
    let mul_mul = mul_by_self.mul(mul_by_self);
    assert_eq!(pow_4, mul_mul);

    let inv = scalar_a.inv();
    let one = scalar_a.mul(inv);
    assert_eq!(one, F::one());
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
        println!("{:?}", projective_point);
        let affine_eqivalent: Affine<C> = projective_point.into();
        assert_eq!(projective_point, affine_eqivalent.into());
    }
}

pub fn check_point_arithmetic<C: Curve>() {
    let size = 1 << 10;
    let projective_points_a = C::generate_random_projective_points(size);
    let projective_points_b = C::generate_random_projective_points(size);

    for i in 0..size {
        let result1 = projective_points_a[i] + projective_points_b[i];
        let result2 = result1 - projective_points_b[i];
        assert_eq!(result2, projective_points_a[i]);
    }
}

pub fn check_point_equality<F: PrimeField, C>()
where
    C: Curve<BaseField = F>,
{
    let left = Projective::<C>::zero();
    let right = Projective::<C>::zero();
    assert_eq!(left, right);

    let x = F::zero();
    let y = F::from_u32(2);
    let z = F::zero();
    let right = Projective::<C>::from_limbs(x.into(), y.into(), z.into());
    assert_eq!(left, right);

    let z = F::from_u32(2);
    let right = Projective::<C>::from_limbs(F::zero().into(), F::from_u32(4).into(), z.into());
    assert_ne!(left, right);

    let left = Projective::<C>::from_limbs(F::zero().into(), F::from_u32(2).into(), C::BaseField::one().into());
    assert_eq!(left, right);
}

pub fn check_field_convert_montgomery<F>()
where
    F: PrimeField + MontgomeryConvertible + GenerateRandom,
{
    let mut stream = IcicleStream::create().unwrap();

    let size = 1 << 10;
    let scalars = F::generate_random(size);

    let mut d_scalars = DeviceVec::<F>::malloc(size);
    d_scalars
        .copy_from_host(scalars.into_slice())
        .unwrap();

    F::to_mont(&mut d_scalars, &stream);
    F::from_mont(&mut d_scalars, &stream);

    let mut scalars_copy = vec![F::zero(); size];
    d_scalars
        .copy_to_host_async(scalars_copy.into_slice_mut(), &stream)
        .unwrap();
    stream
        .synchronize()
        .unwrap();
    stream
        .destroy()
        .unwrap();

    assert_eq!(scalars_copy, scalars);
}

pub fn check_points_convert_montgomery<C: Curve>()
where
    Affine<C>: MontgomeryConvertible,
    Projective<C>: MontgomeryConvertible,
{
    let size = 1 << 10;

    let affine_points = C::generate_random_affine_points(size);
    let mut d_affine = DeviceVec::<Affine<C>>::malloc(size);
    d_affine
        .copy_from_host(affine_points.into_slice())
        .unwrap();

    Affine::<C>::to_mont(&mut d_affine, &IcicleStream::default())
        .wrap()
        .unwrap();
    Affine::<C>::from_mont(&mut d_affine, &IcicleStream::default())
        .wrap()
        .unwrap();

    let affine_copy = d_affine.to_host_vec();

    assert_eq!(affine_points, affine_copy);

    let proj_points = C::generate_random_projective_points(size);
    let mut d_proj = DeviceVec::<Projective<C>>::malloc(size);
    d_proj
        .copy_from_host(proj_points.into_slice())
        .unwrap();

    Projective::<C>::to_mont(&mut d_proj, &IcicleStream::default())
        .wrap()
        .unwrap();
    Projective::<C>::from_mont(&mut d_proj, &IcicleStream::default())
        .wrap()
        .unwrap();

    let projective_copy = d_proj.to_host_vec();

    assert_eq!(proj_points, projective_copy);
}

pub fn check_generator<C: Curve>() {
    let generator = C::get_generator();
    let zero = Projective::<C>::zero();
    assert_ne!(generator, zero);
    assert!(C::is_on_curve(generator));
}
