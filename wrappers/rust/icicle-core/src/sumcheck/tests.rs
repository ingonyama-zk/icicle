use crate::hash::Hasher;
use crate::sumcheck::SumcheckTranscriptConfig;
use crate::traits::{FieldImpl, GenerateRandom};

pub fn check_sumcheck_transcript_config<F: FieldImpl>(hash: &Hasher)
where
    <F as FieldImpl>::Config: GenerateRandom<F>,
{
    let seed_rng = F::Config::generate_random(1)[0];

    // Test `new` constructor
    let config1 = SumcheckTranscriptConfig::new(
        hash,
        b"DomainLabel".to_vec(),
        b"PolyLabel".to_vec(),
        b"ChallengeLabel".to_vec(),
        true,
        seed_rng,
    );

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
        false,
        seed_rng,
    );

    assert_eq!(config2.domain_separator_label, b"DomainLabel");
    assert_eq!(config2.round_poly_label, b"PolyLabel");
    assert_eq!(config2.round_challenge_label, b"ChallengeLabel");
    assert!(!config2.little_endian);
    assert_eq!(config2.seed_rng, seed_rng);
}
