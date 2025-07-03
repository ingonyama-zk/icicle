use crate::polynomial_ring::PolynomialRing;
use crate::random_sampling::{
    challenge_space_polynomials_sampling, random_sampling, ChallengeSpacePolynomialsSampling, RandomSampling,
};
use crate::traits::{Arithmetic, FieldImpl};
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{memory::HostSlice, test_utilities};
use rand::Rng;

pub fn check_random_sampling<F>()
where
    F: FieldImpl + Arithmetic,
    F::Config: RandomSampling<F>,
{
    let output_size = 1 << 10;
    let cfg = VecOpsConfig::default();

    let mut output_a = vec![F::zero(); output_size];
    let mut output_b = vec![F::zero(); output_size];

    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    test_utilities::test_set_main_device();
    random_sampling(
        output_size,
        false,
        &seed,
        &cfg,
        HostSlice::from_mut_slice(&mut output_a),
    )
    .expect("random sampling failed");
    test_utilities::test_set_ref_device();
    random_sampling(
        output_size,
        false,
        &seed,
        &cfg,
        HostSlice::from_mut_slice(&mut output_b),
    )
    .expect("random sampling failed");

    for i in 0..output_size {
        assert_eq!(output_a[i], output_b[i], "random sampling mismatch at index {}", i);
    }
}

pub fn check_challenge_space_polynomials_sampling<P>()
where
    P: PolynomialRing,
    P::Base: FieldImpl + Arithmetic,
    P: ChallengeSpacePolynomialsSampling<P>,
{
    let output_size = 1 << 10;
    let cfg = VecOpsConfig::default();

    let mut output_a = vec![P::zero(); output_size];
    let mut output_b = vec![P::zero(); output_size];

    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    let ones = 31;
    let twos = 10;
    let norm = 15;

    test_utilities::test_set_main_device();
    challenge_space_polynomials_sampling(&seed, &cfg, ones, twos, norm, HostSlice::from_mut_slice(&mut output_a))
        .expect("challenge space polynomials sampling failed");

    test_utilities::test_set_ref_device();
    challenge_space_polynomials_sampling(&seed, &cfg, ones, twos, norm, HostSlice::from_mut_slice(&mut output_b))
        .expect("challenge space polynomials sampling failed");

    for i in 0..output_size {
        assert_eq!(
            output_a[i], output_b[i],
            "challenge space polynomials sampling mismatch at index {}",
            i
        );
    }
}
