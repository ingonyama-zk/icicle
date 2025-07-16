use crate::bignum::BigNum;
use crate::polynomial_ring::{flatten_polyring_slice, PolynomialRing};
use crate::ring::IntegerRing;
use crate::{
    field::Field,
    projective::Projective,
    traits::{GenerateRandom, MontgomeryConvertible},
};
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::{
    memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
    stream::IcicleStream,
};
use std::fmt::Debug;

pub fn check_ring_arithmetic<R>()
where
    R: IntegerRing + GenerateRandom,
{
    let size = 1 << 10;
    let scalars_a = R::generate_random(size);
    let scalars_b = R::generate_random(size);

    for i in 0..size {
        let result1 = scalars_a[i] + scalars_b[i];
        let result2 = result1 - scalars_b[i];
        assert_eq!(result2, scalars_a[i]);

        // Test ring multiplication API
        let scalar_a = scalars_a[i];
        let square = scalar_a.sqr();
        let mul_by_self = scalar_a.mul(scalar_a);
        assert_eq!(square, mul_by_self);

        // Test ring pow API
        let pow_4 = scalar_a.pow(4);
        let mul_mul = mul_by_self.mul(mul_by_self);
        assert_eq!(pow_4, mul_mul);
    }
}

pub fn check_field_arithmetic<F>()
where
    F: Field + GenerateRandom,
{
    check_ring_arithmetic::<F>();
    let size = 1 << 10;
    let scalars_a = F::generate_random(size);

    for i in 0..size {
        // Test field inv API
        let inv = scalars_a[i].inv();
        let one = scalars_a[i].mul(inv);
        assert_eq!(one, F::one());
    }
}

pub fn check_affine_projective_convert<P: Projective>() {
    let size = 1 << 10;
    let affine_points = P::Affine::generate_random(size);
    let projective_points = P::generate_random(size);
    for affine_point in affine_points {
        let projective_eqivalent: P = affine_point.into();
        assert_eq!(affine_point, projective_eqivalent.into());
    }
    for projective_point in projective_points {
        println!("{:?}", projective_point);
        let affine_eqivalent: P::Affine = projective_point.into();
        assert_eq!(projective_point, affine_eqivalent.into());
    }
}

pub fn check_point_arithmetic<P: Projective>() {
    let size = 1 << 10;
    let projective_points_a = P::generate_random(size);
    let projective_points_b = P::generate_random(size);

    for i in 0..size {
        let result1 = projective_points_a[i] + projective_points_b[i];
        let result2 = result1 - projective_points_b[i];
        assert_eq!(result2, projective_points_a[i]);
    }
}

pub fn check_point_equality<P: Projective>() {
    let left = P::zero();
    let right = P::zero();
    assert_eq!(left, right);

    let x = P::BaseField::zero();
    let y = P::BaseField::from(2);
    let z = P::BaseField::zero();
    let right = P::from_limbs(*x.limbs(), *y.limbs(), *z.limbs());
    assert_eq!(left, right);

    let z = P::BaseField::from(2);
    let right = P::from_limbs(
        *P::BaseField::zero().limbs(),
        *P::BaseField::from(4).limbs(),
        *z.limbs(),
    );
    assert_ne!(left, right);

    let left = P::from_limbs(
        *P::BaseField::zero().limbs(),
        *P::BaseField::from(2).limbs(),
        *P::BaseField::one().limbs(),
    );
    assert_eq!(left, right);
}

pub fn check_montgomery_convert_host<T>()
where
    T: Debug + Clone + PartialEq + GenerateRandom + MontgomeryConvertible,
{
    let size = 1 << 10;
    let mut elements = T::generate_random(size);
    let expected = elements.clone();

    T::to_mont(elements.into_slice_mut(), &IcicleStream::default()).unwrap();
    T::from_mont(elements.into_slice_mut(), &IcicleStream::default()).unwrap();

    assert_eq!(expected, elements);
}

pub fn check_montgomery_convert_device<T>()
where
    T: Debug + Default + Clone + Copy + PartialEq + GenerateRandom + MontgomeryConvertible,
{
    let mut stream = IcicleStream::create().unwrap();

    let size = 1 << 10;
    let elements = T::generate_random(size);

    let mut d_elements = DeviceVec::from_host_slice(&elements);

    T::to_mont(&mut d_elements, &stream).unwrap();
    T::from_mont(&mut d_elements, &stream).unwrap();

    let mut elements_copy = vec![T::default(); size];
    d_elements
        .copy_to_host_async(elements_copy.into_slice_mut(), &stream)
        .unwrap();
    stream
        .synchronize()
        .unwrap();
    stream
        .destroy()
        .unwrap();

    assert_eq!(elements_copy, elements);
}

pub fn check_generator<P: Projective>() {
    let generator = P::get_generator();
    let zero = P::zero();
    assert_ne!(generator, zero);
    assert!(P::is_on_curve(generator));
}

pub fn check_zero_and_from_slice<P: PolynomialRing>() {
    let zero = P::zero();
    let expected = vec![P::Base::zero(); P::DEGREE];
    assert_eq!(zero.values(), expected.as_slice());

    let input = vec![P::Base::one(); P::DEGREE];
    let poly = P::from_slice(&input).unwrap();
    assert_eq!(poly.values(), input.as_slice());
}
/// Verifies that flattening a slice of polynomials yields a correctly sized,
/// reinterpreted slice of base field elements.
pub fn check_polyring_flatten_host_memory<P>()
where
    P: PolynomialRing + GenerateRandom,
{
    // Generate a vector of one random polynomial
    let polynomials = P::generate_random(5);
    let poly_slice = polynomials.into_slice();

    // Flatten the polynomial slice to a scalar slice
    let scalar_slice = flatten_polyring_slice(poly_slice);

    // Ensure the flattened slice has the correct number of base elements
    let expected_len = poly_slice.len() * P::DEGREE;
    assert_eq!(
        scalar_slice.len(),
        expected_len,
        "Expected flattened length {}, got {}",
        expected_len,
        scalar_slice.len()
    );

    // Ensure the underlying memory was reinterpreted, not copied
    unsafe {
        assert_eq!(
            poly_slice.as_ptr() as *const P::Base,
            scalar_slice.as_ptr(),
            "Pointer mismatch: flattening should preserve memory layout"
        );
    }
}

/// Verifies that flattening a device slice of polynomials yields a correctly sized,
/// reinterpreted device slice of base field elements without copying.
pub fn check_polyring_flatten_device_memory<P>()
where
    P: PolynomialRing + GenerateRandom,
{
    // Generate a single random polynomial on host and copy to device
    let size = 7;
    let host_polys = P::generate_random(size);
    let device_slice = host_polys.into_slice();

    // Flatten the device polynomial slice
    let scalar_slice = flatten_polyring_slice(device_slice);

    // Check length is DEGREE × num_polynomials
    let expected_len = device_slice.len() * P::DEGREE;
    assert_eq!(
        scalar_slice.len(),
        expected_len,
        "Flattened device slice has incorrect length"
    );

    // Check underlying memory is reinterpreted, not copied
    unsafe {
        assert_eq!(
            device_slice.as_ptr() as *const P::Base,
            scalar_slice.as_ptr(),
            "Flattened device slice does not share memory with original"
        );
    }
}
