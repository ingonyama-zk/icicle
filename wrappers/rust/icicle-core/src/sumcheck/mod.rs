#[doc(hidden)]
pub mod tests;

use crate::field::FieldArithmetic;
use crate::hash::Hasher;
use crate::program::ReturningValueProgram;
use crate::traits::{Arithmetic, FieldConfig, FieldImpl, GenerateRandom};
use icicle_runtime::config::ConfigExtension;
use icicle_runtime::stream::IcicleStreamHandle;
use icicle_runtime::{memory::HostOrDeviceSlice, IcicleError};
use serde::{de::DeserializeOwned, Serialize};
use std::ffi::c_void;

/// Configuration for the Sumcheck protocol's transcript.
///
/// This structure holds the configuration parameters needed for the transcript
/// in the Sumcheck protocol, including hash function, domain labels, and RNG settings.
pub struct SumcheckTranscriptConfig<'a, F> {
    /// The hash function used for transcript generation
    pub hash: &'a Hasher,
    /// Label used for domain separation in the transcript
    pub domain_separator_label: Vec<u8>,
    /// Label used for round polynomials in the transcript
    pub round_poly_label: Vec<u8>,
    /// Label used for round challenges in the transcript
    pub round_challenge_label: Vec<u8>,
    /// Whether to use little-endian byte order
    pub little_endian: bool,
    /// Random number generator seed for transcript generation
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

/// Configuration for the Sumcheck protocol.
///
/// This structure holds runtime configuration parameters that control how the
/// Sumcheck protocol executes, including device settings and performance options.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SumcheckConfig {
    /// Stream for asynchronous execution. Default is nullptr.
    pub stream: IcicleStreamHandle,
    /// If true, use extension field for the fiat shamir result.
    /// Recommended for small fields for security.
    pub use_extension_field: bool,
    /// Number of input chunks to hash in batch. Default is 1.
    pub batch: u64,
    /// True if inputs reside on the device (e.g., GPU), false if on the host (CPU).
    /// Default is false.
    pub are_inputs_on_device: bool,
    /// True to run the hash asynchronously, false to run synchronously.
    /// Default is false.
    pub is_async: bool,
    /// Pointer to backend-specific configuration extensions.
    /// Default is nullptr.
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
///
/// This trait defines the core functionality of the Sumcheck protocol,
/// including proof generation and verification. It is generic over the
/// field type and configuration.
pub trait Sumcheck {
    /// The field type used in the protocol
    type Field: FieldImpl + Arithmetic;
    /// The field configuration type
    type FieldConfig: FieldConfig + GenerateRandom<Self::Field> + FieldArithmetic<Self::Field>;
    /// The proof type used in the protocol
    type Proof: SumcheckProofOps<Self::Field>;

    /// Creates a new instance of the Sumcheck protocol.
    fn new() -> Result<Self, IcicleError>
    where
        Self: Sized;

    /// Generates a proof for the given MLE polynomials and claimed sum.
    ///
    /// # Arguments
    /// * `mle_polys` - Slice of MLE polynomials to prove
    /// * `mle_poly_size` - Size of each MLE polynomial
    /// * `claimed_sum` - The claimed sum to prove
    /// * `combine_function` - Function to combine MLE polynomials
    /// * `transcript_config` - Configuration for the transcript
    /// * `sumcheck_config` - Runtime configuration
    fn prove(
        &self,
        mle_polys: &[&(impl HostOrDeviceSlice<Self::Field> + ?Sized)],
        mle_poly_size: u64,
        claimed_sum: Self::Field,
        combine_function: impl ReturningValueProgram,
        transcript_config: &SumcheckTranscriptConfig<Self::Field>,
        sumcheck_config: &SumcheckConfig,
    ) -> Self::Proof;

    /// Verifies a Sumcheck proof against a claimed sum.
    ///
    /// # Arguments
    /// * `proof` - The proof to verify
    /// * `claimed_sum` - The claimed sum to verify against
    /// * `transcript_config` - Configuration for the transcript
    ///
    /// # Returns
    /// * `Ok(true)` if verification succeeds
    /// * `Ok(false)` if verification fails
    /// * `Err(e)` if an error occurs during verification
    fn verify(
        &self,
        proof: &Self::Proof,
        claimed_sum: Self::Field,
        transcript_config: &SumcheckTranscriptConfig<Self::Field>,
    ) -> Result<bool, IcicleError>;

    /// Retrieves the challenge vector from the Sumcheck instance.
    ///
    /// The challenge vector contains the alpha values used in each round
    /// of the protocol. The first challenge is always zero (as per protocol
    /// design), while subsequent challenges are derived from the previous
    /// round's polynomial.
    ///
    /// # Returns
    /// * `Ok(Vec<Field>)` containing the challenge vector
    /// * `Err(e)` if an error occurs
    fn get_challenge_vector(&self) -> Result<Vec<Self::Field>, IcicleError>;
}

/// Trait for Sumcheck proof operations.
///
/// This trait defines operations that can be performed on a Sumcheck proof,
/// including retrieving round polynomials and printing the proof.
pub trait SumcheckProofOps<F>: From<Vec<Vec<F>>> + Serialize + DeserializeOwned
where
    F: FieldImpl,
{
    /// Retrieves the round polynomials from the proof.
    ///
    /// # Returns
    /// * `Ok(Vec<Vec<F>>)` containing the round polynomials
    /// * `Err(e)` if an error occurs
    fn get_round_polys(&self) -> Result<Vec<Vec<F>>, IcicleError>;

    /// Prints the proof for debugging purposes.
    ///
    /// # Returns
    /// * `Ok(())` indicating success
    /// * `Err(e)` if an error occurs
    fn print(&self) -> Result<(), IcicleError>;
}

/// Macro to implement Sumcheck functionality for a specific field.
#[macro_export]
macro_rules! impl_sumcheck {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_cfg:ident) => {
        mod $field_prefix_ident {
            use super::{$field, $field_cfg};
            use crate::symbol::$field_prefix_ident::FieldSymbol;
            use icicle_core::program::{PreDefinedProgram, ProgramHandle, ReturningValueProgram};
            use icicle_core::sumcheck::{
                FFISumcheckTranscriptConfig, Sumcheck, SumcheckConfig, SumcheckProofOps, SumcheckTranscriptConfig,
            };
            use icicle_core::traits::{FieldImpl, Handle};
            use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, IcicleError};
            use serde::de::{self, Visitor};
            use serde::{Deserialize, Deserializer, Serialize, Serializer};
            use std::ffi::c_void;
            use std::slice;

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

                #[link_name = concat!($field_prefix, "_sumcheck_get_challenge_vector")]
                fn icicle_sumcheck_get_challenge_vector(
                    handle: SumcheckHandle,
                    challenge_vector: *mut $field,
                    challenge_vector_size: &mut usize,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_sumcheck_get_challenge_size")]
                fn icicle_sumcheck_get_challenge_size(
                    handle: SumcheckHandle,
                    challenge_size: *mut usize,
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

                #[link_name = concat!($field_prefix, "_sumcheck_proof_get_serialized_size")]
                fn icicle_sumcheck_proof_get_serialized_size(
                    handle: SumcheckProofHandle,
                    size: *mut usize,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_sumcheck_proof_serialize")]
                fn icicle_sumcheck_proof_serialize(
                    handle: SumcheckProofHandle,
                    buffer: *mut u8,
                    size: usize,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_sumcheck_proof_deserialize")]
                fn icicle_sumcheck_proof_deserialize(
                    handle: *mut SumcheckProofHandle,
                    buffer: *const u8,
                    size: usize,
                ) -> eIcicleError;
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

                fn new() -> Result<Self, IcicleError> {
                    let handle = unsafe { icicle_sumcheck_create() };
                    if handle.is_null() {
                        return Err(IcicleError::new(
                            eIcicleError::UnknownError,
                            "Failed to create Sumcheck instance",
                        ));
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
                ) -> Result<bool, IcicleError> {
                    let ffi_transcript_config = FFISumcheckTranscriptConfig::from(transcript_config);
                    let mut is_verified = false;
                    unsafe {
                        icicle_sumcheck_verify(
                            self.handle,
                            proof.handle,
                            &claimed_sum,
                            &ffi_transcript_config,
                            &mut is_verified,
                        )
                        .wrap_value(is_verified)
                    }
                }

                fn get_challenge_vector(&self) -> Result<Vec<$field>, IcicleError> {
                    let mut challenge_size = 0usize;

                    unsafe { icicle_sumcheck_get_challenge_size(self.handle, &mut challenge_size).wrap()? };

                    // Initialize the challenge vector with zeros; will be resized after getting the actual size from FFI
                    let mut challenge_vector = vec![$field::zero(); challenge_size];
                    unsafe {
                        icicle_sumcheck_get_challenge_vector(
                            self.handle,
                            challenge_vector.as_mut_ptr(),
                            &mut challenge_size,
                        )
                        .wrap_value(challenge_vector)
                    }
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

            impl Handle for SumcheckWrapper {
                fn handle(&self) -> *const c_void {
                    self.handle
                }
            }

            /***************** SumcheckProof *************************/
            type SumcheckProofHandle = *const c_void;

            pub struct SumcheckProof {
                pub(crate) handle: SumcheckProofHandle,
            }

            impl SumcheckProofOps<$field> for SumcheckProof {
                fn get_round_polys(&self) -> Result<Vec<Vec<$field>>, IcicleError> {
                    let mut poly_size = 0u64;
                    let mut nof_polys = 0u64;
                    unsafe {
                        icicle_sumcheck_proof_get_proof_sizes(self.handle, &mut poly_size, &mut nof_polys).wrap()?
                    };

                    let mut rounds = Vec::with_capacity(nof_polys as usize);
                    for i in 0..nof_polys {
                        let poly_ptr = unsafe { icicle_sumcheck_proof_get_proof_at(self.handle, i) };
                        let poly = unsafe { slice::from_raw_parts(poly_ptr, poly_size as usize) }.to_vec();
                        rounds.push(poly);
                    }
                    Ok(rounds)
                }

                fn print(&self) -> Result<(), IcicleError> {
                    unsafe { icicle_sumcheck_proof_print(self.handle).wrap() }
                }
            }

            impl From<Vec<Vec<$field>>> for SumcheckProof {
                fn from(polys: Vec<Vec<$field>>) -> Self {
                    let nof_polys = polys.len() as u64;
                    let poly_size = if nof_polys > 0 { polys[0].len() as u64 } else { 0 };

                    let mut polys_ptrs: Vec<*const $field> = Vec::with_capacity(nof_polys as usize);
                    for poly in &polys {
                        polys_ptrs.push(poly.as_ptr());
                    }

                    let handle = unsafe {
                        icicle_sumcheck_proof_create(polys_ptrs.as_ptr() as *const *const $field, nof_polys, poly_size)
                    };

                    SumcheckProof { handle }
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

            impl Serialize for SumcheckProof {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: Serializer,
                {
                    let mut size = 0;
                    unsafe {
                        icicle_sumcheck_proof_get_serialized_size(self.handle, &mut size)
                            .wrap_value(size)
                            .map_err(serde::ser::Error::custom)?;
                        let mut buffer = vec![0u8; size];
                        icicle_sumcheck_proof_serialize(self.handle, buffer.as_mut_ptr(), buffer.len())
                            .wrap()
                            .map_err(serde::ser::Error::custom)?;
                        serializer.serialize_bytes(&buffer)
                    }
                }
            }

            impl<'de> Deserialize<'de> for SumcheckProof {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: Deserializer<'de>,
                {
                    struct SumcheckProofVisitor;

                    impl<'de> Visitor<'de> for SumcheckProofVisitor {
                        type Value = SumcheckProof;

                        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                            formatter.write_str("a byte array representing a SumcheckProof")
                        }

                        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                        where
                            E: de::Error,
                        {
                            let mut handle = std::ptr::null();
                            unsafe {
                                icicle_sumcheck_proof_deserialize(&mut handle, v.as_ptr(), v.len())
                                    .wrap_value(SumcheckProof { handle })
                                    .map_err(de::Error::custom)
                            }
                        }
                        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                        where
                            A: serde::de::SeqAccess<'de>,
                        {
                            let mut buffer = Vec::with_capacity(
                                seq.size_hint()
                                    .unwrap_or(0),
                            );
                            while let Some(byte) = seq.next_element::<u8>()? {
                                buffer.push(byte);
                            }
                            self.visit_bytes(&buffer)
                        }
                    }

                    deserializer.deserialize_bytes(SumcheckProofVisitor)
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
        use super::*;
        use crate::program::$field_prefix_ident::FieldReturningValueProgram as Program;
        use icicle_core::sumcheck::tests::*;
        use icicle_hash::keccak::Keccak256;
        use icicle_runtime::{device::Device, runtime, test_utilities};
        use serde_json;
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

        #[test]
        fn test_sumcheck_proof_serialization() {
            initialize();
            test_utilities::test_set_ref_device();
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_proof_serialization::<SumcheckWrapper, Program, _, _, String>(
                &hash,
                |proof| serde_json::to_string(proof).unwrap(),
                |s| serde_json::from_str(&s).unwrap(),
            );
        }

        #[test]
        fn test_sumcheck_challenge_vector() {
            test_utilities::test_set_ref_device();
            let hash = Keccak256::new(0).unwrap();
            check_sumcheck_challenge_vector::<SumcheckWrapper, Program>(&hash);
        }
    };
}
