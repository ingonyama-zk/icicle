pub mod fri_proof;
pub mod fri_transcript_config;
pub mod tests;
use crate::traits::{FieldConfig, FieldImpl, GenerateRandom};
use crate::{field::FieldArithmetic, hash::Hasher};
use fri_proof::FriProofTrait;
use fri_transcript_config::FriTranscriptConfig;
use icicle_runtime::{config::ConfigExtension, eIcicleError, memory::HostOrDeviceSlice, IcicleStreamHandle};

pub type FriProof<F> = <<F as FieldImpl>::Config as FriMerkleTreeImpl<F>>::FriProof;

/// Computes the FRI proof using the given configuration and input data.
/// # Returns
/// - `Ok(())` if the FRI proof was successfully computed.
/// - `Err(eIcicleError)` if an error occurred during proof generation.
pub fn fri_merkle_tree_prove<F: FieldImpl>(
    config: &FriConfig,
    fri_transcript_config: &FriTranscriptConfig<F>,
    input_data: &(impl HostOrDeviceSlice<F> + ?Sized),
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    output_store_min_layer: u64,
) -> Result<FriProof<F>, eIcicleError>
where
    <F as FieldImpl>::Config: FriMerkleTreeImpl<F>,
{
    if input_data.is_on_device() && !input_data.is_on_active_device() {
        panic!("input_data not allocated on the active device");
    }
    let mut local_cfg = config.clone();
    local_cfg.are_inputs_on_device = true;
    let mut fri_proof = FriProof::<F>::new();
    <F::Config as FriMerkleTreeImpl<F>>::fri_merkle_tree_prove(
        config,
        fri_transcript_config,
        input_data,
        merkle_tree_leaves_hash,
        merkle_tree_compress_hash,
        output_store_min_layer,
        &mut fri_proof,
    )?;
    Ok(fri_proof)
}

/// Verifies a FRI proof using the given configuration and Merkle tree hashes.
/// # Returns
/// - `Ok(true)` if the proof is valid.
/// - `Ok(false)` if the proof is invalid.
/// - `Err(eIcicleError)` if verification failed due to an error.
pub fn fri_merkle_tree_verify<F: FieldImpl>(
    config: &FriConfig,
    fri_transcript_config: &FriTranscriptConfig<F>,
    fri_proof: &FriProof<F>,
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    output_store_min_layer: u64,
) -> Result<bool, eIcicleError>
where
    <F as FieldImpl>::Config: FriMerkleTreeImpl<F>,
{
    <F::Config as FriMerkleTreeImpl<F>>::fri_merkle_tree_verify(
        config,
        fri_transcript_config,
        fri_proof,
        merkle_tree_leaves_hash,
        merkle_tree_compress_hash,
        output_store_min_layer,
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
    are_inputs_on_device: bool, // True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false.
    pub is_async: bool,         // True to run operations asynchronously, false to run synchronously. Default is false.
    pub ext: ConfigExtension,   // Pointer to backend-specific configuration extensions. Default is nullptr.
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

pub trait FriMerkleTreeImpl<F: FieldImpl> {
    type FieldConfig: FieldConfig + GenerateRandom<F> + FieldArithmetic<F>;
    type FriProof: FriProofTrait<F>;
    fn fri_merkle_tree_prove(
        config: &FriConfig,
        fri_transcript_config: &FriTranscriptConfig<F>,
        input_data: &(impl HostOrDeviceSlice<F> + ?Sized),
        merkle_tree_leaves_hash: &Hasher,
        merkle_tree_compress_hash: &Hasher,
        output_store_min_layer: u64,
        fri_proof: &mut Self::FriProof,
    ) -> Result<(), eIcicleError>;

    fn fri_merkle_tree_verify(
        config: &FriConfig,
        fri_transcript_config: &FriTranscriptConfig<F>,
        fri_proof: &Self::FriProof,
        merkle_tree_leaves_hash: &Hasher,
        merkle_tree_compress_hash: &Hasher,
        output_store_min_layer: u64,
    ) -> Result<bool, eIcicleError>;
}

#[macro_export]
macro_rules! impl_fri {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use super::{$field, $field_config};
            use icicle_core::fri::fri_transcript_config::FriTranscriptConfig;
            use icicle_core::{
                fri::{fri_transcript_config::FFIFriTranscriptConfig, FriConfig, FriMerkleTreeImpl},
                hash::{Hasher, HasherHandle},
                impl_fri_proof,
                traits::FieldImpl,
            };
            use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

            impl_fri_proof!($field_prefix, $field, $field_config);

            extern "C" {
                #[link_name = concat!($field_prefix, "_fri_merkle_tree_prove")]
                fn icicle_fri_merkle_tree_prove(
                    fri_config: *const FriConfig,
                    fri_transcript_config: *const FFIFriTranscriptConfig<$field>,
                    input_data: *const $field,
                    input_size: u64,
                    merkle_tree_leaves_hash: HasherHandle,
                    merkle_tree_compress_hash: HasherHandle,
                    output_store_min_layer: u64,
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
            impl FriMerkleTreeImpl<$field> for $field_config {
                type FieldConfig = $field_config;
                type FriProof = FriProof;

                fn fri_merkle_tree_prove(
                    config: &FriConfig,
                    transcript_config: &FriTranscriptConfig<$field>,
                    input_data: &(impl HostOrDeviceSlice<$field> + ?Sized),
                    merkle_tree_leaves_hash: &Hasher,
                    merkle_tree_compress_hash: &Hasher,
                    output_store_min_layer: u64,
                    fri_proof: &mut FriProof,
                ) -> Result<(), eIcicleError> {
                    let ffi_transcript_config = FFIFriTranscriptConfig::<$field>::from(transcript_config);
                    unsafe {
                        icicle_fri_merkle_tree_prove(
                            config as *const FriConfig,
                            &ffi_transcript_config,
                            input_data.as_ptr() as *const $field,
                            input_data.len() as u64,
                            merkle_tree_leaves_hash.handle,
                            merkle_tree_compress_hash.handle,
                            output_store_min_layer,
                            fri_proof.handle,
                        )
                        .wrap()
                    }
                }

                fn fri_merkle_tree_verify(
                    config: &FriConfig,
                    fri_transcript_config: &FriTranscriptConfig<$field>,
                    fri_proof: &FriProof,
                    merkle_tree_leaves_hash: &Hasher,
                    merkle_tree_compress_hash: &Hasher,
                    output_store_min_layer: u64,
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
      $field_prefix_ident:ident,
      $ntt_field:ident,
      $field:ident,
      $hasher_new:expr
    ) => {
        mod $field_prefix_ident {
            use super::*;
            use icicle_core::fri::fri_transcript_config::FriTranscriptConfig;
            use icicle_core::fri::tests::check_fri;
            use icicle_core::hash::Hasher;
            use icicle_core::ntt::tests::init_domain;
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use serial_test::parallel;
            use std::sync::Once;

            const MAX_SIZE: u64 = 1 << 10;
            static INIT: Once = Once::new();
            const FAST_TWIDDLES_MODE: bool = false;

            pub fn initialize() {
                INIT.call_once(move || {
                    // test_utilities::test_load_and_init_devices();
                    // // init domain for both devices
                    // test_utilities::test_set_ref_device();
                    init_domain::<$ntt_field>(MAX_SIZE, FAST_TWIDDLES_MODE);

                    // test_utilities::test_set_main_device();
                    // init_domain::<$ntt_field>(MAX_SIZE, FAST_TWIDDLES_MODE);
                });
                // test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_fri() {
                initialize();

                check_fri::<$field>(&$hasher_new);
            }
        }
    };
}
