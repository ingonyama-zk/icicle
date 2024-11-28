use crate::{
    curve::{Affine, Curve, Projective},
    field::Field,
    traits::{FieldConfig, FieldImpl, GenerateRandom, MontgomeryConvertible, Arithmetic},
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};

pub fn check_field_equality<F: FieldImpl>() {
    let left = F::zero();
    let right = F::one();
    assert_ne!(left, right);
    let left = F::from_bytes_le(&[1]);
    assert_eq!(left, right);
}

pub fn check_field_arithmetic<F>()
where
    F: FieldImpl + Arithmetic,
    F::Config: GenerateRandom<F>,
{
    let size = 1 << 10;
    let scalars_a = F::Config::generate_random(size);
    let scalars_b = F::Config::generate_random(size);

    for i in 0..size {
        let result1 = scalars_a[i].add(scalars_b[i]);
        let result2 = result1.sub(scalars_b[i]);
        assert_eq!(result2, scalars_a[i]);
    }

    let scalar_a = scalars_a[0];
    let square = scalar_a.square();
    let mul = scalar_a.mul(scalar_a);

    assert_eq!(square, mul);
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

pub fn check_point_arithmetic<C: Curve>() {
    let size = 1 << 10;
    let projective_points_a = C::generate_random_projective_points(size);
    let projective_points_b = C::generate_random_projective_points(size);

    for i in 0..size {
        let result1 = projective_points_a[i] + projective_points_b[i];
        let result2 = result1 - projective_points_b[i]; //only X coordinate is correct Y and Z are zero
        assert_eq!(result2, projective_points_a[i]);
    }

    // let point = projective_points_a[0];
    // let scalar = FieldImpl::from_u32(3);

    // let mul = point * scalar;
    // let add = point + point + point;
    
    // assert_eq!(mul, point);
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

pub fn check_field_convert_montgomery<F>()
where
    F: FieldImpl + MontgomeryConvertible,
    F::Config: GenerateRandom<F>,
{
    let mut stream = IcicleStream::create().unwrap();

    let size = 1 << 10;
    let scalars = F::Config::generate_random(size);

    let mut d_scalars = DeviceVec::device_malloc(size).unwrap();
    d_scalars
        .copy_from_host(HostSlice::from_slice(&scalars))
        .unwrap();

    F::to_mont(&mut d_scalars, &stream);
    F::from_mont(&mut d_scalars, &stream);

    let mut scalars_copy = vec![F::zero(); size];
    d_scalars
        .copy_to_host_async(HostSlice::from_mut_slice(&mut scalars_copy), &stream)
        .unwrap();
    stream
        .synchronize()
        .unwrap();
    stream
        .destroy()
        .unwrap();

    assert_eq!(scalars_copy, scalars);
}


// pub fn check_field_arithmetic<F>()
// where 
//     F: FieldImpl,
//     F::Config: GenerateRandom<F>,
// {
//     let size = 1 << 10;
//     let scalars_a = F::Config::generate_random(size);
//     let scalars_b = F::Config::generate_random(size);

//     for i in 0..size {
//         let result1 = scalars_a[i] + scalars_b[i];
//         let result2 = result1 - scalars_b[i];
//         assert_eq!(result2, scalars_a[i]);
//     }
// }

pub fn check_points_convert_montgomery<C: Curve>()
where
    Affine<C>: MontgomeryConvertible,
    Projective<C>: MontgomeryConvertible,
{
    let size = 1 << 10;

    let affine_points = C::generate_random_affine_points(size);
    let mut d_affine = DeviceVec::device_malloc(size).unwrap();
    d_affine
        .copy_from_host(HostSlice::from_slice(&affine_points))
        .unwrap();

    Affine::<C>::to_mont(&mut d_affine, &IcicleStream::default())
        .wrap()
        .unwrap();
    Affine::<C>::from_mont(&mut d_affine, &IcicleStream::default())
        .wrap()
        .unwrap();

    let mut affine_copy = vec![Affine::<C>::zero(); size];
    d_affine
        .copy_to_host(HostSlice::from_mut_slice(&mut affine_copy))
        .unwrap();

    assert_eq!(affine_points, affine_copy);

    let proj_points = C::generate_random_projective_points(size);
    let mut d_proj = DeviceVec::device_malloc(size).unwrap();
    d_proj
        .copy_from_host(HostSlice::from_slice(&proj_points))
        .unwrap();

    Projective::<C>::to_mont(&mut d_proj, &IcicleStream::default())
        .wrap()
        .unwrap();
    Projective::<C>::from_mont(&mut d_proj, &IcicleStream::default())
        .wrap()
        .unwrap();

    let mut projective_copy = vec![Projective::<C>::zero(); size];
    d_proj
        .copy_to_host(HostSlice::from_mut_slice(&mut projective_copy))
        .unwrap();

    assert_eq!(proj_points, projective_copy);
}
