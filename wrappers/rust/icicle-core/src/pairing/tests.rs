use crate::{
    curve::{Affine, Curve},
    traits::FieldImpl,
};

use super::{pairing, Pairing};

pub fn check_pairing_bilinearity<C1, C2, F>()
where
    C1: Curve,
    C2: Curve,
    F: FieldImpl,
    C1: Pairing<C1, C2, F>,
{
    let p = C1::generate_random_affine_points(1)[0];
    let q = C2::generate_random_affine_points(1)[0];
    let coeff = 42;
    let s1 = C1::ScalarField::from_u32(coeff);
    let s2 = C2::ScalarField::from_u32(coeff);

    let ps = Affine::<C1>::from(p.to_projective() * s1);
    let qs = Affine::<C2>::from(q.to_projective() * s2);

    let res1 = pairing(&ps, &q).unwrap();
    let res2 = pairing(&p, &qs).unwrap();

    assert_eq!(res1, res2);
}
