use crate::hash::Hasher;
use crate::sumcheck::{
    PreDefinedProgram, ReturningValueProgramTrait, Sumcheck, SumcheckConfig, SumcheckProofOps, SumcheckTranscriptConfig,
};
use crate::traits::{FieldImpl, GenerateRandom};
use icicle_runtime::memory::HostSlice;

/// Tests the `SumcheckTranscriptConfig` struct with different constructors.
pub fn check_sumcheck_transcript_config<F: FieldImpl>(hash: &Hasher)
where
    <F as FieldImpl>::Config: GenerateRandom<F>,
{
    // Generate a random seed for the test.
    let seed_rng = F::Config::generate_random(1)[0];

    // Test `new` constructor
    let config1 = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    // Verify that the fields are correctly initialized
    assert_eq!(config1.domain_separator_label, b"DomainLabel");
    assert_eq!(config1.round_poly_label, b"PolyLabel");
    assert_eq!(config1.round_challenge_label, b"ChallengeLabel");
    assert!(config1.little_endian);
    assert_eq!(config1.seed_rng, seed_rng);

    // Test `from_string_labels` constructor
    let config2 = SumcheckTranscriptConfig::from_string_labels(
        hash,
        "DomainLabel",
        "PolyLabel",
        "ChallengeLabel",
        false, // big endian
        seed_rng,
    );

    // Verify that the fields are correctly initialized
    assert_eq!(config2.domain_separator_label, b"DomainLabel");
    assert_eq!(config2.round_poly_label, b"PolyLabel");
    assert_eq!(config2.round_challenge_label, b"ChallengeLabel");
    assert!(!config2.little_endian);
    assert_eq!(config2.seed_rng, seed_rng);
}

/// Tests the `Sumcheck` struct's basic functionality, including proving and verifying.
pub fn check_sumcheck_simple<SW, P>(hash: &Hasher)
where
    SW: Sumcheck,
    P: ReturningValueProgramTrait,
{
    let log_mle_poly_size = 13u64;
    let mle_poly_size = 1 << log_mle_poly_size;
    let nof_mle_poly = 4;
    // Generate a random seed for the test.
    let seed_rng = <<SW as Sumcheck>::FieldConfig>::generate_random(1)[0];

    // Create a transcript configuration.
    let config = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    let mut mle_polys = Vec::with_capacity(nof_mle_poly);
    for _ in 0..nof_mle_poly {
        let mle_poly_random = <<SW as Sumcheck>::FieldConfig>::generate_random(mle_poly_size);
        mle_polys.push(mle_poly_random);
    }

    // Create a Sumcheck instance using the transcript configuration.
    let mut claimed_sum = <<SW as Sumcheck>::Field as FieldImpl>::zero();
    for i in 0..mle_poly_size {
        let a = mle_polys[0][i];
        let b = mle_polys[1][i];
        let c = mle_polys[2][i];
        let eq = mle_polys[3][i];
        claimed_sum = claimed_sum + (a * b - c) * eq;
    }

    /****** Being Proof ******/
    let sumcheck = SW::new().unwrap();

    let mle_poly_ptrs: Vec<*const <SW as Sumcheck>::Field> = mle_polys
        .iter()
        .map(|poly| {
            let poly_ptr = poly.as_ptr();
            return poly_ptr;
        })
        .collect::<Vec<*const <SW as Sumcheck>::Field>>();

    let combine_func = P::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();
    let sumcheck_config = SumcheckConfig::default();
    // Generate a proof using the `prove` method.
    let proof = sumcheck.prove(
        HostSlice::from_slice(&mle_poly_ptrs[..]),
        mle_poly_size as u64,
        claimed_sum,
        combine_func,
        &config,
        &sumcheck_config,
    );
    /****** End Proof ******/

    /****** Obtain Proof Data ******/
    let proof_data = <<SW as Sumcheck>::Proof as SumcheckProofOps<<SW as Sumcheck>::Field>>::get_proof(&proof).unwrap();

    /********** Verifier deserializes proof data *********/
    let proof_as_sumcheck_proof: <SW as Sumcheck>::Proof = <SW as Sumcheck>::Proof::from(proof_data);

    // Verify the proof.
    let valid = sumcheck.verify(&proof_as_sumcheck_proof, claimed_sum, &config);

    assert!(valid);
}
