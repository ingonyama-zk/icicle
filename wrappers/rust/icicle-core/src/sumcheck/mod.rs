#[doc(hidden)]
pub mod tests;

use crate::hash::Hasher;

pub struct SumcheckTranscriptConfig<'a, S> {
    pub hash: &'a Hasher,
    pub domain_separator_label: Vec<u8>,
    pub round_poly_label: Vec<u8>,
    pub round_challenge_label: Vec<u8>,
    pub little_endian: bool,
    pub seed_rng: S,
}

impl<'a, S> SumcheckTranscriptConfig<'a, S> {
    /// Constructor for `SumcheckTranscriptConfig` with explicit parameters.
    pub fn new(
        hash: &'a Hasher,
        domain_separator_label: Vec<u8>,
        round_poly_label: Vec<u8>,
        round_challenge_label: Vec<u8>,
        little_endian: bool,
        seed_rng: S,
    ) -> Self {
        Self {
            hash,
            domain_separator_label,
            round_poly_label,
            round_challenge_label,
            little_endian,
            seed_rng,
        }
    }

    /// Convenience constructor for `SumcheckTranscriptConfig` using string labels.
    pub fn from_string_labels(
        hash: &'a Hasher,
        domain_separator_label: &str,
        round_poly_label: &str,
        round_challenge_label: &str,
        little_endian: bool,
        seed_rng: S,
    ) -> Self {
        Self {
            hash,
            domain_separator_label: domain_separator_label
                .as_bytes()
                .to_vec(),
            round_poly_label: round_poly_label
                .as_bytes()
                .to_vec(),
            round_challenge_label: round_challenge_label
                .as_bytes()
                .to_vec(),
            little_endian,
            seed_rng,
        }
    }
}

#[macro_export]
macro_rules! impl_sumcheck_tests {
    (
        $field:ident
      ) => {
        use icicle_core::sumcheck::tests::*;
        use icicle_hash::keccak::Keccak256;

        #[test]
        fn test_sumcheck_transcript_config() {
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_transcript_config::<$field>(&hash)
        }
    };
}
