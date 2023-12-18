use crate::{
    curve::{Affine, CurveConfig, Projective},
    traits::{FieldImpl, GetLimbs},
};

pub fn check_scalar_equality<F: FieldImpl>() {
    let left = F::zero();
    let right = F::one();
    assert_ne!(left, right);
    let left = F::set_limbs(&[1]);
    assert_eq!(left, right);
}

pub fn check_affine_projective_convert<C: CurveConfig>() {
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

pub fn check_point_equality<const BASE_LIMBS: usize, C: CurveConfig>()
where
    C::BaseField: GetLimbs<BASE_LIMBS>,
{
    let left = Projective::<C>::zero();
    let right = Projective::<C>::zero();
    assert_eq!(left, right);
    let right = Projective::<C>::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &[0; BASE_LIMBS]);
    assert_eq!(left, right);
    let right = Projective::<C>::set_limbs(
        &[0; BASE_LIMBS],
        &[4; BASE_LIMBS],
        &C::BaseField::set_limbs(&[2]).get_limbs(),
    );
    assert_ne!(left, right);
    let left = Projective::<C>::set_limbs(&[0; BASE_LIMBS], &[2; BASE_LIMBS], &C::BaseField::one().get_limbs());
    assert_eq!(left, right);
}
