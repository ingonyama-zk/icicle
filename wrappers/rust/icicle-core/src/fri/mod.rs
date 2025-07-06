pub mod fri_proof;
pub mod fri_transcript_config;
pub mod tests;
use crate::traits::{Arithmetic, GenerateRandom};
use crate::{field::Field, hash::Hasher};
use fri_proof::FriProofOps;
use fri_transcript_config::FriTranscriptConfig;
use icicle_runtime::{config::ConfigExtension, eIcicleError, memory::HostOrDeviceSlice, IcicleStreamHandle};

pub type FriProof<F> = <F as FriMerkleTree<F>>::FriProof;

/// Computes the FRI proof using the given configuration and input data.
/// # Returns
/// - `Ok(())` if the FRI proof was successfully computed.
/// - `Err(eIcicleError)` if an error occurred during proof generation.
pub fn fri_merkle_tree_prove<F: Field>(
    config: &FriConfig,
    fri_transcript_config: &FriTranscriptConfig<F>,
    input_data: &(impl HostOrDeviceSlice<F> + ?Sized),
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    merkle_tree_min_layer_to_store: u64,
) -> Result<FriProof<F>, eIcicleError>
where
    F: FriMerkleTree<F>,
{
    F::fri_merkle_tree_prove(
        config,
        fri_transcript_config,
        input_data,
        merkle_tree_leaves_hash,
        merkle_tree_compress_hash,
        merkle_tree_min_layer_to_store,
    )
}

/// Verifies a FRI proof using the given configuration and Merkle tree hashes.
/// # Returns
/// - `Ok(true)` if the proof is valid.
/// - `Ok(false)` if the proof is invalid.
/// - `Err(eIcicleError)` if verification failed due to an error.
pub fn fri_merkle_tree_verify<F: Field>(
    config: &FriConfig,
    fri_transcript_config: &FriTranscriptConfig<F>,
    fri_proof: &FriProof<F>,
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
) -> Result<bool, eIcicleError>
where
    F: FriMerkleTree<F>,
{
    F::fri_merkle_tree_verify(
        config,
        fri_transcript_config,
        fri_proof,
        merkle_tree_leaves_hash,
        merkle_tree_compress_hash,
    )
}

#[repr(C)]
#[derive(Clone)]
pub struct FriConfig {
    pub stream_handle: IcicleStreamHandle, // Stream for asynchronous execution. Default is nullptr.
    pub folding_factor: u64,               // The factor by which the codeword is folded in each round.
    pub stopping_degree: u64,              // The minimal polynomial degree at which folding stops.
    pub pow_bits: u64,                     // Number of leading zeros required for proof-of-work. Default is 0.
    pub nof_queries: u64,                  // Number of queries, computed for each folded layer of FRI. Default is 1.
    pub are_inputs_on_device: bool, // True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false.
    pub is_async: bool, // True to run operations asynchronously, false to run synchronously. Default is false.
    pub ext: ConfigExtension, // Pointer to backend-specific configuration extensions. Default is nullptr.
}

impl Default for FriConfig {
    /// Create a default configuration (same as the C++ struct's defaults)
    fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            folding_factor: 2,
            stopping_degree: 0,
            pow_bits: 16,
            nof_queries: 100,
            are_inputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

pub trait FriMerkleTree<F: Field> {
    type FieldConfig: Field + GenerateRandom + Arithmetic;
    type FriProof: FriProofOps<F>;
    fn fri_merkle_tree_prove(
        config: &FriConfig,
        fri_transcript_config: &FriTranscriptConfig<F>,
        input_data: &(impl HostOrDeviceSlice<F> + ?Sized),
        merkle_tree_leaves_hash: &Hasher,
        merkle_tree_compress_hash: &Hasher,
        merkle_tree_min_layer_to_store: u64,
    ) -> Result<Self::FriProof, eIcicleError>;

    fn fri_merkle_tree_verify(
        config: &FriConfig,
        fri_transcript_config: &FriTranscriptConfig<F>,
        fri_proof: &Self::FriProof,
        merkle_tree_leaves_hash: &Hasher,
        merkle_tree_compress_hash: &Hasher,
    ) -> Result<bool, eIcicleError>;
}

#[macro_export]
macro_rules! impl_fri {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        mod $field_prefix_ident {
            use super::$field;
            use icicle_core::fri::fri_transcript_config::FriTranscriptConfig;
            use icicle_core::{
                field::Field,
                fri::{fri_transcript_config::FFIFriTranscriptConfig, FriConfig, FriMerkleTree},
                hash::{Hasher, HasherHandle},
                impl_fri_proof,
            };
            use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

            impl_fri_proof!($field_prefix, $field);

            extern "C" {
                #[link_name = concat!($field_prefix, "_fri_merkle_tree_prove")]
                fn icicle_fri_merkle_tree_prove(
                    fri_config: *const FriConfig,
                    fri_transcript_config: *const FFIFriTranscriptConfig<$field>,
                    input_data: *const $field,
                    input_size: u64,
                    merkle_tree_leaves_hash: HasherHandle,
                    merkle_tree_compress_hash: HasherHandle,
                    merkle_tree_min_layer_to_store: u64,
                    fri_proof: FriProofHandle,
                ) -> eIcicleError;
                #[link_name = concat!($field_prefix, "_fri_merkle_tree_verify")]
                fn fri_merkle_tree_verify(
                    fri_config: *const FriConfig,
                    fri_transcript_config: *const FFIFriTranscriptConfig<$field>,
                    fri_proof: FriProofHandle,
                    merkle_tree_leaves_hash: HasherHandle,
                    merkle_tree_compress_hash: HasherHandle,
                    valid: *mut bool,
                ) -> eIcicleError;
            }
            impl FriMerkleTree<$field> for $field {
                type FieldConfig = $field;
                type FriProof = FriProof;

                fn fri_merkle_tree_prove(
                    config: &FriConfig,
                    transcript_config: &FriTranscriptConfig<$field>,
                    input_data: &(impl HostOrDeviceSlice<$field> + ?Sized),
                    merkle_tree_leaves_hash: &Hasher,
                    merkle_tree_compress_hash: &Hasher,
                    merkle_tree_min_layer_to_store: u64,
                ) -> Result<FriProof, eIcicleError> {
                    if input_data.is_on_device() && !input_data.is_on_active_device() {
                        return Err(eIcicleError::InvalidDevice);
                    }
                    let mut local_cfg = config.clone();
                    if input_data.is_on_device() {
                        local_cfg.are_inputs_on_device = true;
                    }
                    let ffi_transcript_config = FFIFriTranscriptConfig::<$field>::from(transcript_config);
                    let mut fri_proof = FriProof::new()?;
                    unsafe {
                        icicle_fri_merkle_tree_prove(
                            config as *const FriConfig,
                            &ffi_transcript_config,
                            input_data.as_ptr() as *const $field,
                            input_data.len() as u64,
                            merkle_tree_leaves_hash.handle,
                            merkle_tree_compress_hash.handle,
                            merkle_tree_min_layer_to_store,
                            fri_proof.handle,
                        )
                        .wrap_value(fri_proof)
                    }
                }

                fn fri_merkle_tree_verify(
                    config: &FriConfig,
                    fri_transcript_config: &FriTranscriptConfig<$field>,
                    fri_proof: &FriProof,
                    merkle_tree_leaves_hash: &Hasher,
                    merkle_tree_compress_hash: &Hasher,
                ) -> Result<bool, eIcicleError> {
                    let ffi_transcript_config = FFIFriTranscriptConfig::<$field>::from(fri_transcript_config);
                    let mut valid: bool = false;
                    unsafe {
                        let err = fri_merkle_tree_verify(
                            config as *const FriConfig,
                            &ffi_transcript_config,
                            fri_proof.handle,
                            merkle_tree_leaves_hash.handle,
                            merkle_tree_compress_hash.handle,
                            &mut valid as *mut bool,
                        );
                        err.wrap_value(valid)
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_fri_tests {
    (
      $ntt_field:ident,
      $field:ident
    ) => {
        use super::*;
        use icicle_core::fri::fri_transcript_config::FriTranscriptConfig;
        use icicle_core::fri::tests::{check_fri, check_fri_on_device, check_fri_proof_serialization};
        use icicle_core::hash::Hasher;
        use icicle_core::ntt::tests::init_domain;
        use icicle_hash::keccak::Keccak256;
        use icicle_runtime::test_utilities;
        use icicle_runtime::{device::Device, runtime};
        use serial_test::parallel;
        use std::sync::Once;

        const MAX_SIZE: u64 = 1 << 10;
        static INIT: Once = Once::new();
        const FAST_TWIDDLES_MODE: bool = false;

        pub fn initialize() {
            INIT.call_once(move || {
                test_utilities::test_load_and_init_devices();
                // init domain for both devices
                test_utilities::test_set_ref_device();
                init_domain::<$ntt_field>(MAX_SIZE, FAST_TWIDDLES_MODE);

                test_utilities::test_set_main_device();
                init_domain::<$ntt_field>(MAX_SIZE, FAST_TWIDDLES_MODE);
            });
            test_utilities::test_set_main_device();
        }

        // Note: tests are prefixed with 'phase4' since they conflict with NTT tests domain.
        //       The fri tests are executed via 'cargo test phase4' as an additional step

        #[test]
        pub fn phase4_test_fri() {
            initialize();
            let merkle_tree_leaves_hash = Keccak256::new(std::mem::size_of::<$field>() as u64).unwrap();
            let merkle_tree_compress_hash = Keccak256::new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
            let transcript_hash = Keccak256::new(0).unwrap();
            check_fri::<$field>(
                &merkle_tree_leaves_hash,
                &merkle_tree_compress_hash,
                &transcript_hash,
            );
        }

        #[test]
        pub fn phase4_test_fri_on_device() {
            initialize();

            let merkle_tree_leaves_hash = Keccak256::new(std::mem::size_of::<$field>() as u64).unwrap();
            let merkle_tree_compress_hash = Keccak256::new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
            let transcript_hash = Keccak256::new(0).unwrap();
            check_fri_on_device::<$field>(
                &merkle_tree_leaves_hash,
                &merkle_tree_compress_hash,
                &transcript_hash,
            );
        }

        #[test]
        pub fn phase4_test_fri_proof_serialization() {
            initialize();
            let merkle_tree_leaves_hash = Keccak256::new(std::mem::size_of::<$field>() as u64).unwrap();
            let merkle_tree_compress_hash = Keccak256::new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
            let transcript_hash = Keccak256::new(0).unwrap();
            check_fri_proof_serialization::<$field, _, _, String>(
                &merkle_tree_leaves_hash,
                &merkle_tree_compress_hash,
                &transcript_hash,
                |fri_proof| serde_json::to_string(fri_proof).unwrap(),
                |s| serde_json::from_str(&s).unwrap(),
            );
        }
    };
}

#[macro_export]
macro_rules! impl_fri_test_with_poseidon {
    (
      $ntt_field:ident,
      $field:ident
    ) => {
        #[test]
        pub fn phase4_test_fri_poseidon2() {
            use icicle_core::poseidon2::Poseidon2;
            initialize();
            let merkle_tree_leaves_hash = Poseidon2::new_with_input_size::<$field>(3, None, 1).unwrap();
            let merkle_tree_compress_hash = Poseidon2::new::<$field>(2, None).unwrap();
            let transcript_hash = Keccak256::new(0).unwrap();
            check_fri::<$field>(
                &merkle_tree_leaves_hash,
                &merkle_tree_compress_hash,
                &transcript_hash,
            );
        }

        #[test]
        pub fn phase4_test_fri_poseidon() {
            use icicle_core::poseidon::Poseidon;
            initialize();
            let merkle_tree_leaves_hash = Poseidon::new_with_input_size::<$field>(3, None, 1).unwrap();
            let merkle_tree_compress_hash = Poseidon::new_with_input_size::<$field>(5, None, 2).unwrap();
            let transcript_hash = Keccak256::new(0).unwrap();
            check_fri::<$field>(
                &merkle_tree_leaves_hash,
                &merkle_tree_compress_hash,
                &transcript_hash,
            );
        }
    };
}
