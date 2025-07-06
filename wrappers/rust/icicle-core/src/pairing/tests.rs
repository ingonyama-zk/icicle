use crate::{curve::Curve, field::Field, traits::GenerateRandom};

use super::{pairing, Pairing};

pub fn check_pairing_bilinearity<C1, C2, F>()
where
    C1: Curve,
    C2: Curve,
    F: Field,
    C1: Pairing<C1, C2, F>,
{
    let p = C1::Affine::generate_random(1)[0];
    let q = C2::Affine::generate_random(1)[0];
    let coeff = 42;
    let s1 = C1::ScalarField::from(coeff);
    let s2 = C2::ScalarField::from(coeff);

    let ps: C1::Affine = C1::Projective::into(C1::Projective::from(p) * s1);
    let qs: C2::Affine = C2::Projective::into(C2::Projective::from(q) * s2);

    let res1 = pairing::<C1, C2, F>(&ps, &q).unwrap();
    let res2 = pairing::<C1, C2, F>(&p, &qs).unwrap();

    assert_eq!(res1, res2);
}
