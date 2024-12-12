#[doc(hidden)]
pub mod tests;

use crate::hash::Hasher;
use crate::traits::FieldImpl;
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};
use std::ffi::c_void;

/// Configuration for the Sumcheck protocol's transcript.
pub struct SumcheckTranscriptConfig<'a, F> {
    pub hash: &'a Hasher,
    pub domain_separator_label: Vec<u8>,
    pub round_poly_label: Vec<u8>,
    pub round_challenge_label: Vec<u8>,
    pub little_endian: bool,
    pub seed_rng: F,
}

/// Trait for constructing a Sumcheck prover instance.
pub trait SumcheckConstructor<F> {
    /// Creates a new Sumcheck prover instance.
    /// Optionally, consider returning `Box<dyn SumcheckOps<F>>`
    fn new(transcript_config: &SumcheckTranscriptConfig<F>) -> Result<impl SumcheckOps<F>, eIcicleError>;
}

/// Trait for Sumcheck operations, including proving and verification.
pub trait SumcheckOps<F> {
    fn prove(&self, input: &(impl HostOrDeviceSlice<F> + ?Sized)) -> String; // TODO: Replace `String` with proof type.
    fn verify(&self, proof: &str) -> bool;
}

/// Empty struct used to represent Sumcheck operations generically.
pub struct Sumcheck;

impl Sumcheck {
    fn new<'a, F: FieldImpl>(
        transcript_config: &'a SumcheckTranscriptConfig<'a, F>,
    ) -> Result<impl SumcheckOps<F> + 'a, eIcicleError>
    where
        F: FieldImpl + SumcheckConstructor<F>,
    {
        <F as SumcheckConstructor<F>>::new(&transcript_config)
    }
}

impl<'a, F> SumcheckTranscriptConfig<'a, F> {
    /// Constructs a new `SumcheckTranscriptConfig` with explicit parameters.
    pub fn new(
        hash: &'a Hasher,
        domain_separator_label: Vec<u8>,
        round_poly_label: Vec<u8>,
        round_challenge_label: Vec<u8>,
        little_endian: bool,
        seed_rng: F,
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

    /// Convenience constructor using string labels.
    pub fn from_string_labels(
        hash: &'a Hasher,
        domain_separator_label: &str,
        round_poly_label: &str,
        round_challenge_label: &str,
        little_endian: bool,
        seed_rng: F,
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

/// Converts `SumcheckTranscriptConfig` to its FFI-compatible equivalent.
#[repr(C)]
pub struct FFISumcheckTranscriptConfig<F> {
    hash: *const c_void,
    domain_separator_label: *const u8,
    domain_separator_label_len: usize,
    round_poly_label: *const u8,
    round_poly_label_len: usize,
    round_challenge_label: *const u8,
    round_challenge_label_len: usize,
    little_endian: bool,
    pub seed_rng: *const F,
}

impl<'a, F> From<&SumcheckTranscriptConfig<'a, F>> for FFISumcheckTranscriptConfig<F>
where
    F: FieldImpl,
{
    fn from(config: &SumcheckTranscriptConfig<'a, F>) -> Self {
        FFISumcheckTranscriptConfig {
            hash: config
                .hash
                .handle,
            domain_separator_label: config
                .domain_separator_label
                .as_ptr(),
            domain_separator_label_len: config
                .domain_separator_label
                .len(),
            round_poly_label: config
                .round_poly_label
                .as_ptr(),
            round_poly_label_len: config
                .round_poly_label
                .len(),
            round_challenge_label: config
                .round_challenge_label
                .as_ptr(),
            round_challenge_label_len: config
                .round_challenge_label
                .len(),
            little_endian: config.little_endian,
            seed_rng: &config.seed_rng,
        }
    }
}

/// Macro to implement Sumcheck functionality for a specific field.
#[macro_export]
macro_rules! impl_sumcheck {
    (
        $field_prefix:literal, 
        $field_prefix_ident:ident, 
        $field:ident
    ) => {
        use icicle_core::sumcheck::{
            FFISumcheckTranscriptConfig, SumcheckConstructor, SumcheckOps, SumcheckTranscriptConfig,
        };
        use icicle_core::traits::FieldImpl;
        use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};
        use std::ffi::c_void;

        pub type SumcheckHandle = *const c_void;

        pub struct SumcheckInternal {
            handle: SumcheckHandle,
        }

        extern "C" {
            #[link_name = concat!($field_prefix, "_sumcheck_create")]
            fn icicle_sumcheck_create(config: *const FFISumcheckTranscriptConfig<$field>) -> SumcheckHandle;

            #[link_name = concat!($field_prefix, "_sumcheck_delete")]
            fn icicle_sumcheck_delete(handle: SumcheckHandle) -> eIcicleError;
        }

        impl SumcheckConstructor<$field> for $field {
            fn new(
                transcript_config: &SumcheckTranscriptConfig<$field>,
            ) -> Result<impl SumcheckOps<$field>, eIcicleError> {
                let handle = unsafe { icicle_sumcheck_create(&transcript_config.into()) };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(SumcheckInternal { handle })
            }
        }

        impl SumcheckOps<$field> for SumcheckInternal {
            fn prove(&self, _input: &(impl HostOrDeviceSlice<$field> + ?Sized)) -> String {
                String::from("Proof")
            }

            fn verify(&self, _proof: &str) -> bool {
                true
            }
        }

        impl Drop for SumcheckInternal {
            fn drop(&mut self) {
                if !self
                    .handle
                    .is_null()
                {
                    unsafe {
                        let _ = icicle_sumcheck_delete(self.handle);
                    }
                }
            }
        }
    };
}

/// Macro to define tests for a specific field.
#[macro_export]
macro_rules! impl_sumcheck_tests {
    ($field:ident) => {
        use icicle_core::sumcheck::tests::*;
        use icicle_hash::keccak::Keccak256;

        #[test]
        fn test_sumcheck_transcript_config() {
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_transcript_config::<$field>(&hash);
        }

        #[test]
        fn test_sumcheck_simple() {
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_simple::<$field>(&hash);
        }
    };
}
