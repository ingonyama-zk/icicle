#[doc(hidden)]
pub mod tests;

use crate::hash::Hasher;
use crate::traits::FieldImpl;
use icicle_runtime::eIcicleError;

pub struct SumcheckTranscriptConfig<'a, S> {
    pub hash: &'a Hasher,
    pub domain_separator_label: Vec<u8>,
    pub round_poly_label: Vec<u8>,
    pub round_challenge_label: Vec<u8>,
    pub little_endian: bool,
    pub seed_rng: S,
}
// This trait is implemented on FieldConfig to enable Sumcheck struct to create a sumcheck prover
pub trait SumcheckConstructor<F> {
    fn new(transcript_config: &SumcheckTranscriptConfig<F>) -> Result<impl SumcheckOps<F>, eIcicleError>;
}

pub trait SumcheckOps<F> {
    // TODO replace with sumcheck proof type
    fn prove(&self) -> String;
    fn verify(&self, proof: &str) -> bool;
}

// This struct is used simply to construct a sumcheck instance that implements sumcheck ops in a generic way
pub struct Sumcheck;

/*******************/

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

impl Sumcheck {
    fn new<'a, F: FieldImpl>(
        transcript_config: &'a SumcheckTranscriptConfig<'a, F>,
    ) -> Result<impl SumcheckOps<F> + 'a, eIcicleError>
    where
        F: FieldImpl,
        F::Config: SumcheckConstructor<F>,
    {
        <<F as FieldImpl>::Config as SumcheckConstructor<F>>::new(&transcript_config)
    }
}

#[macro_export]
macro_rules! impl_sumcheck {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_cfg:ident
    ) => {
        use icicle_core::sumcheck::{SumcheckConstructor, SumcheckOps, SumcheckTranscriptConfig};
        use icicle_core::traits::FieldImpl;
        use icicle_runtime::eIcicleError;
        use std::ffi::c_void;

        pub type SumcheckHandle = *const c_void;
        pub struct SumcheckInternal {
            handle: SumcheckHandle,
        }

        extern "C" {
            #[link_name = concat!($field_prefix, "_sumcheck_create")]
            fn create() -> SumcheckHandle;
        }

        impl SumcheckConstructor<$field> for $field_cfg {
            fn new(
                transcript_config: &SumcheckTranscriptConfig<$field>,
            ) -> Result<impl SumcheckOps<$field>, eIcicleError> {
                let handle: SumcheckHandle = unsafe {
                    // TODO add params
                    create()
                };
                // if handle.is_null() {
                //     return Err(eIcicleError::UnknownError);
                // }
                Ok(SumcheckInternal { handle })
            }
        }

        impl SumcheckOps<$field> for SumcheckInternal {
            fn prove(&self) -> String {
                String::from("hello")
            }

            fn verify(&self, _: &str) -> bool {
                true
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

        #[test]
        fn test_sumcheck_simple() {
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_simple::<$field>(&hash)
        }
    };
}
