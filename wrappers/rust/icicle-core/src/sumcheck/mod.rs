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

// TODO Yuval: add a Sumcheck trait and implement it in the macro per field

#[macro_export]
macro_rules! impl_sumcheck {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_cfg:ident
    ) => {
        use icicle_core::sumcheck::SumcheckTranscriptConfig;
        use icicle_core::traits::FieldImpl;
        use icicle_runtime::eIcicleError;
        use std::ffi::c_void;

        type SumcheckHandle = *const c_void;
        pub struct Sumcheck {
            handle: SumcheckHandle,
        }

        extern "C" {
            #[link_name = concat!($field_prefix, "_sumcheck_create")]
            fn create() -> SumcheckHandle;
        }

        impl Sumcheck {
            pub fn new<S: FieldImpl>(transcript_config: &SumcheckTranscriptConfig<S>) -> Result<Self, eIcicleError> {
                unsafe {
                    // TODO real call
                    create();
                }
                Err(eIcicleError::UnknownError)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sumcheck_tests {
    ($field:ident) => {
        use icicle_core::sumcheck::tests::*;
        use icicle_hash::keccak::Keccak256;

        #[test]
        fn test_sumcheck_transcript_config() {
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_transcript_config::<$field>(&hash)
        }
    };
}
