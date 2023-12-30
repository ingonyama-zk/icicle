#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
use crate::{
    curve::{Affine, Curve, Projective},
    field::Field,
    traits::{FieldConfig, FieldImpl, GenerateRandom},
};
#[cfg(feature = "arkworks")]
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective};

pub fn check_scalar_equality<F: FieldImpl>() {
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
