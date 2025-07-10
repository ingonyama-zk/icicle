use crate::{field::Field, projective::Projective, traits::GenerateRandom};

use super::{pairing, Pairing};

pub fn check_pairing_bilinearity<P1, P2, F>()
where
    P1: Projective,
    P2: Projective,
    F: Field,
    P1: Pairing<P1, P2, F>,
{
    let p = P1::Affine::generate_random(1)[0];
    let q = P2::Affine::generate_random(1)[0];
    let coeff = 42;
    let s1 = P1::ScalarField::from(coeff);
    let s2 = P2::ScalarField::from(coeff);

    let ps: P1::Affine = P1::into(P1::from(p) * s1);
    let qs: P2::Affine = P2::into(P2::from(q) * s2);

    let res1 = pairing::<P1, P2, F>(&ps, &q).unwrap();
    let res2 = pairing::<P1, P2, F>(&p, &qs).unwrap();

    assert_eq!(res1, res2);
}
