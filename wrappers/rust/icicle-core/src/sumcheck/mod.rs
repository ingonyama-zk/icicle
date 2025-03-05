#[doc(hidden)]
pub mod tests;

use crate::field::FieldArithmetic;
use crate::hash::Hasher;
use crate::program::ReturningValueProgram;
use crate::traits::{Arithmetic, FieldConfig, FieldImpl, GenerateRandom};
use icicle_runtime::config::ConfigExtension;
use icicle_runtime::stream::IcicleStreamHandle;
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

/// Converts [SumcheckTranscriptConfig](SumcheckTranscriptConfig) to its FFI-compatible equivalent.
#[repr(C)]
#[derive(Debug)]
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

#[repr(C)]
#[derive(Debug, Clone)]
pub struct SumcheckConfig {
    /**< Stream for asynchronous execution. Default is nullptr. */
    pub stream: IcicleStreamHandle,
    /**< If true, then use extension field for the fiat shamir result. Recommended for small fields for security*/
    pub use_extension_field: bool,
    /**< Number of input chunks to hash in batch. Default is 1. */
    pub batch: u64,
    /**< True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false. */
    pub are_inputs_on_device: bool,
    /**< True to run the hash asynchronously, false to run synchronously. Default is false. */
    pub is_async: bool,
    /**< Pointer to backend-specific configuration extensions. Default is nullptr. */
    pub ext: ConfigExtension,
}

impl Default for SumcheckConfig {
    fn default() -> Self {
        Self {
            stream: std::ptr::null_mut(),
            use_extension_field: false,
            batch: 1,
            are_inputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

/// Trait for Sumcheck operations, including proving and verification.
pub trait Sumcheck {
    type Field: FieldImpl + Arithmetic;
    type FieldConfig: FieldConfig + GenerateRandom<Self::Field> + FieldArithmetic<Self::Field>;
    type Proof: SumcheckProofOps<Self::Field>;

    fn new() -> Result<Self, eIcicleError>
    where
        Self: Sized;

    fn prove(
        &self,
        mle_polys: &[&(impl HostOrDeviceSlice<Self::Field> + ?Sized)],
        mle_poly_size: u64,
        claimed_sum: Self::Field,
        combine_function: impl ReturningValueProgram,
        transcript_config: &SumcheckTranscriptConfig<Self::Field>,
        sumcheck_config: &SumcheckConfig,
    ) -> Self::Proof;

    fn verify(
        &self,
        proof: &Self::Proof,
        claimed_sum: Self::Field,
        transcript_config: &SumcheckTranscriptConfig<Self::Field>,
    ) -> Result<bool, eIcicleError>;
}

pub trait SumcheckProofOps<F>: From<Vec<Vec<F>>>
where
    F: FieldImpl,
{
    fn get_round_polys(&self) -> Result<Vec<Vec<F>>, eIcicleError>;
    fn print(&self) -> eIcicleError;
}

/// Macro to implement Sumcheck functionality for a specific field.
#[macro_export]
macro_rules! impl_sumcheck {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_cfg:ident) => {
        use icicle_core::program::{PreDefinedProgram, ReturningValueProgram, ProgramHandle};
        use icicle_core::sumcheck::{
            FFISumcheckTranscriptConfig, Sumcheck, SumcheckConfig, SumcheckProofOps, SumcheckTranscriptConfig,
        };
        use icicle_core::traits::{FieldImpl, Handle};
        use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};
        use std::ffi::c_void;
        use std::slice;
        use crate::symbol::$field_prefix_ident::FieldSymbol;

        extern "C" {
            #[link_name = concat!($field_prefix, "_sumcheck_create")]
            fn icicle_sumcheck_create() -> SumcheckHandle;

            #[link_name = concat!($field_prefix, "_sumcheck_delete")]
            fn icicle_sumcheck_delete(handle: SumcheckHandle) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_sumcheck_get_proof")]
            fn icicle_sumcheck_prove(
                handle: SumcheckHandle,
                mle_polys: *const *const $field,
                mle_poly_size: u64,
                num_mle_polys: u64,
                claimed_sum: &$field,
                combine_function_handle: ProgramHandle,
                transcript_config: &FFISumcheckTranscriptConfig<$field>,
                sumcheck_config: &SumcheckConfig,
            ) -> SumcheckProofHandle;

            #[link_name = concat!($field_prefix, "_sumcheck_verify")]
            fn icicle_sumcheck_verify(
                handle: SumcheckHandle,
                proof: SumcheckProofHandle,
                claimed_sum: *const $field,
                transcript_config: &FFISumcheckTranscriptConfig<$field>,
                is_verified: &mut bool,
            ) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_sumcheck_proof_create")]
            fn icicle_sumcheck_proof_create(
                polys: *const *const $field,
                nof_polys: u64,
                poly_size: u64,
            ) -> SumcheckProofHandle;

            #[link_name = concat!($field_prefix, "_sumcheck_proof_delete")]
            fn icicle_sumcheck_proof_delete(handle: SumcheckProofHandle) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_sumcheck_proof_get_poly_sizes")]
            fn icicle_sumcheck_proof_get_proof_sizes(
                handle: SumcheckProofHandle,
                poly_size: *mut u64,
                nof_polys: *mut u64,
            ) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_sumcheck_proof_get_round_poly_at")]
            fn icicle_sumcheck_proof_get_proof_at(handle: SumcheckProofHandle, index: u64) -> *const $field;

            #[link_name = concat!($field_prefix, "_sumcheck_proof_print")]
            fn icicle_sumcheck_proof_print(handle: SumcheckProofHandle) -> eIcicleError;
        }

        /***************** SumcheckWrapper *************************/
        pub type SumcheckHandle = *const c_void;

        pub struct SumcheckWrapper {
            handle: SumcheckHandle,
        }

        impl Sumcheck for SumcheckWrapper {
            type Field = $field;
            type FieldConfig = $field_cfg;
            type Proof = SumcheckProof;

            fn new() -> Result<Self, eIcicleError> {
                let handle = unsafe { icicle_sumcheck_create() };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }

                Ok(SumcheckWrapper { handle })
            }

            fn prove(
                &self,
                mle_polys: &[&(impl HostOrDeviceSlice<$field> + ?Sized)],
                mle_poly_size: u64,
                claimed_sum: $field,
                combine_function: impl ReturningValueProgram,
                transcript_config: &SumcheckTranscriptConfig<$field>,
                sumcheck_config: &SumcheckConfig,
            ) -> Self::Proof {
                let ffi_transcript_config = FFISumcheckTranscriptConfig::from(transcript_config);

                let mut cfg = sumcheck_config.clone();
                if mle_polys[0].is_on_device() {
                    for mle_poly in mle_polys {
                        assert!(mle_poly.is_on_active_device());
                    }
                    cfg.are_inputs_on_device = true;
                }

                unsafe {
                    let mle_polys_internal: Vec<*const $field> = mle_polys
                        .iter()
                        .map(|mle_poly| mle_poly.as_ptr())
                        .collect();

                    let proof_handle = icicle_sumcheck_prove(
                        self.handle,
                        mle_polys_internal.as_ptr() as *const *const $field,
                        mle_poly_size,
                        mle_polys.len() as u64,
                        &claimed_sum,
                        combine_function.handle(),
                        &ffi_transcript_config,
                        &cfg,
                    );

                    Self::Proof { handle: proof_handle }
                }
            }

            fn verify(
                &self,
                proof: &Self::Proof,
                claimed_sum: $field,
                transcript_config: &SumcheckTranscriptConfig<$field>,
            ) -> Result<bool, eIcicleError> {
                let ffi_transcript_config = FFISumcheckTranscriptConfig::from(transcript_config);
                let mut is_verified = false;
                let err = unsafe {
                    icicle_sumcheck_verify(
                        self.handle,
                        proof.handle,
                        &claimed_sum,
                        &ffi_transcript_config,
                        &mut is_verified,
                    )
                };

                if err != eIcicleError::Success {
                    return Err(err);
                }

                Ok(is_verified)
            }
        }

        impl Drop for SumcheckWrapper {
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

        /***************** SumcheckProof *************************/

        type SumcheckProofHandle = *const c_void;

        pub struct SumcheckProof {
            pub(crate) handle: SumcheckProofHandle,
        }

        impl SumcheckProofOps<$field> for SumcheckProof {
            fn get_round_polys(&self) -> Result<Vec<Vec<$field>>, eIcicleError> {
                let mut poly_size = 0;
                let mut num_polys = 0;
                unsafe {
                    let err = icicle_sumcheck_proof_get_proof_sizes(self.handle, &mut poly_size, &mut num_polys);

                    if err != eIcicleError::Success {
                        return Err(err);
                    }

                    let mut rounds: Vec<Vec<$field>> = Vec::with_capacity(num_polys as usize);

                    for i in 0..num_polys {
                        let round = icicle_sumcheck_proof_get_proof_at(self.handle, i as u64);
                        let round_slice = slice::from_raw_parts(round, poly_size as usize);
                        rounds.push(round_slice.to_vec());
                    }

                    Ok(rounds)
                }
            }

            fn print(&self) -> eIcicleError {
                unsafe { icicle_sumcheck_proof_print(self.handle) }
            }
        }

        impl From<Vec<Vec<$field>>> for SumcheckProof {
            fn from(value: Vec<Vec<$field>>) -> Self {
                let vec_of_pointers: Vec<*const $field> = value
                    .iter()
                    .map(|vec| vec.as_ptr())
                    .collect();
                unsafe {
                    let handle = icicle_sumcheck_proof_create(
                        vec_of_pointers.as_ptr() as *const *const $field,
                        value.len() as u64,
                        value[0].len() as u64,
                    );

                    if handle.is_null() {
                        panic!("Couldn't convert into SumcheckProof");
                    }

                    Self { handle }
                }
            }
        }

        impl Drop for SumcheckProof {
            fn drop(&mut self) {
                if !self
                    .handle
                    .is_null()
                {
                    unsafe {
                        let _ = icicle_sumcheck_proof_delete(self.handle);
                    }
                }
            }
        }
    };
}

/// Macro to define tests for a specific field.
#[macro_export]
macro_rules! impl_sumcheck_tests {
    (
        $field_prefix_ident: ident,
        $field:ident
    ) => {
        use super::SumcheckWrapper;
        use crate::program::$field_prefix_ident::FieldReturningValueProgram as Program;
        use icicle_core::sumcheck::tests::*;
        use icicle_hash::keccak::Keccak256;
        use icicle_runtime::{device::Device, runtime, test_utilities};
        use std::sync::Once;

        const MAX_SIZE: u64 = 1 << 18;
        static INIT: Once = Once::new();
        const FAST_TWIDDLES_MODE: bool = false;

        pub fn initialize() {
            INIT.call_once(move || {
                test_utilities::test_load_and_init_devices();
            });
            test_utilities::test_set_ref_device();
        }

        #[test]
        fn test_sumcheck_transcript_config() {
            initialize();
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_transcript_config::<$field>(&hash);
        }

        #[test]
        fn test_sumcheck_simple() {
            initialize();
            test_utilities::test_set_ref_device();
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_simple::<SumcheckWrapper, Program>(&hash);

            test_utilities::test_set_main_device();
            let device_hash = Keccak256::new(0).unwrap();
            check_sumcheck_simple_device::<SumcheckWrapper, Program>(&device_hash);
        }

        #[test]
        fn test_sumcheck_user_defined_combine() {
            initialize();
            test_utilities::test_set_ref_device();
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_user_defined_combine::<SumcheckWrapper, Program>(&hash);

            test_utilities::test_set_main_device();
            let device_hash = Keccak256::new(0).unwrap();
            check_sumcheck_user_defined_combine::<SumcheckWrapper, Program>(&device_hash);
        }
    };
}
