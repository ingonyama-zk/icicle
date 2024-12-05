use crate::hash::Hasher;
use crate::sumcheck::{Sumcheck, SumcheckConstructor, SumcheckOps, SumcheckTranscriptConfig};
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
pub fn check_sumcheck_simple<F: FieldImpl>(hash: &Hasher)
where
    <F as FieldImpl>::Config: GenerateRandom<F> + SumcheckConstructor<F>,
{
    // Generate a random seed for the test.
    let seed_rng = F::Config::generate_random(1)[0];

    // Create a transcript configuration.
    let config = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true, // little endian
        seed_rng,
    );

    // Create a Sumcheck instance using the transcript configuration.
    let sumcheck = Sumcheck::new::<F>(&config).unwrap();

    // Generate dummy input data.
    let dummy_input: Vec<F> = F::Config::generate_random(5);

    // Generate a proof using the `prove` method.
    let proof = sumcheck.prove(HostSlice::from_slice(&dummy_input));

    // Verify the proof using the `verify` method.
    let _valid = sumcheck.verify(&proof);
}
