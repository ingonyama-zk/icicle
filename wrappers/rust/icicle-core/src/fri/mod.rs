use std::{ffi::c_void, marker::PhantomData};

use crate::{hash::Hasher, traits::FieldImpl};
use icicle_runtime::{config::ConfigExtension, eIcicleError, IcicleStreamHandle};

pub type FriHandle = *const c_void;

pub struct Fri {
    pub handle: FriHandle,
}

pub type FriTranscriptConfigHandle = *const c_void;
pub struct FriTranscriptConfig<F: FieldImpl> {
    pub handle: FriTranscriptConfigHandle,
    p: PhantomData<F>,
}

pub type FriProofHandle = *mut c_void;

pub struct FriProof {
    pub handle: FriProofHandle,
}

pub trait FriTranscriptConfigTrait<F: FieldImpl>: Sized {
    fn from_handle(_handle: FriTranscriptConfigHandle) -> Self;

    fn new(hasher: Hasher, seed_rng: F) -> Result<Self, eIcicleError>; // TODO: add labels here

    fn default_wrapped() -> Result<Self, eIcicleError>;
}


#[repr(C)]
#[derive(Clone)]
pub struct FriConfig {
    pub stream_handle: IcicleStreamHandle, // Stream for asynchronous execution. Default is nullptr.
    pub pow_bits: u64,                 // Number of leading zeros required for proof-of-work. Default is 0.
    pub nof_queries: u64,              // Number of queries, computed for each folded layer of FRI. Default is 1.
    are_inputs_on_device: bool, // True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false.
    pub is_async: bool, // True to run operations asynchronously, false to run synchronously. Default is false.
    pub ext: ConfigExtension, // Pointer to backend-specific configuration extensions. Default is nullptr.
}

impl Default for FriConfig {
    /// Create a default configuration (same as the C++ struct's defaults)
    fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            pow_bits: 0,
            nof_queries: 1,
            are_inputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[macro_export]
macro_rules! impl_fri {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        use icicle_core::hash::HasherHandle;
        use icicle_core::hash::Hasher;
        use icicle_runtime::eIcicleError;
        use icicle_runtime::memory::HostOrDeviceSlice;
        use icicle_core::fri::FriTranscriptConfigHandle;
        use icicle_core::fri::FriTranscriptConfig;
        use icicle_core::fri::FriTranscriptConfigTrait;
        use icicle_core::traits::FieldImpl;
        use std::marker::PhantomData;
        use icicle_core::fri::FriProof;
        use icicle_core::fri::Fri;
        use icicle_core::fri::FriConfig;
        use icicle_core::fri::FriHandle;
        use icicle_core::fri::FriProofHandle;

        extern "C" {
            #[link_name = concat!($field_prefix, "_create_default_fri_transcript_config")]
            fn icicle_create_default_fri_transcript_config() -> FriTranscriptConfigHandle;
            #[link_name = concat!($field_prefix, "_create_fri_transcript_config")]
            fn icicle_create_fri_transcript_config(
                hasher_handle: HasherHandle,
                seed_rng: *const $field,
            ) -> FriTranscriptConfigHandle;
            fn icicle_delete_fri_transcript_config(config_handle: FriTranscriptConfigHandle) -> eIcicleError;
        }

        impl<$field> FriTranscriptConfigTrait<$field> for FriTranscriptConfig<$field> {
            fn from_handle(_handle: FriTranscriptConfigHandle) -> Self {
                Self { handle: _handle, p: PhantomData }
            }
        
            fn new(hasher: Hasher, seed_rng: $field) -> Result<Self, eIcicleError> { // TODO: add labels here
                let handle: FriTranscriptConfigHandle = unsafe { icicle_create_fri_transcript_config(hasher.handle, &seed_rng as *const $field) };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(Self::from_handle(handle))
            }
        
            fn default_wrapped() -> Result<Self, eIcicleError> {
                let handle: FriTranscriptConfigHandle = unsafe { icicle_create_default_fri_transcript_config() };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(Self::from_handle(handle))
            }
        }

        impl<$field> Default for FriTranscriptConfig<$field> {
            fn default() -> Self {
                Self::default_wrapped().unwrap()
            }
        }
        
        impl<$field> Drop for FriTranscriptConfig<$field> {
            fn drop(&mut self) {
                unsafe {
                    icicle_delete_fri_transcript_config(self.handle);
                }
            }
        }

        extern "C" {
            #[link_name = concat!($field_prefix, "_create_fri_proof")]
            fn icicle_create_fri_proof() -> FriProofHandle;
        }

        impl FriProof {
            pub fn from_handle(_handle: FriProofHandle) -> Self {
                FriProof { handle: _handle }
            }
            pub fn default_wrapped() -> Result<Self, eIcicleError> {
                let handle: FriProofHandle = unsafe { icicle_create_fri_proof() };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(Self::from_handle(handle))
            }
        }

        impl Default for FriProof {
            fn default() -> Self {
                Self::default_wrapped().unwrap()
            }
        }

        extern "C" {
            #[link_name = concat!($field_prefix, "_create_fri")]
            fn icicle_create_fri(
                input_size: u64,
                folding_factor: u64,
                stopping_degree: u64,
                merkle_tree_leaves_hash: HasherHandle,
                merkle_tree_compress_hash: HasherHandle,
                output_store_min_layer: u64
            ) -> FriHandle;
        
            fn icicle_delete_fri(fri_handle: FriHandle) -> eIcicleError;
        
            fn icicle_fri_get_proof(
                fri_ptr: FriHandle,
                fri_config: *const FriConfig,
                fri_transcript_config: FriTranscriptConfigHandle,
                input_data: *const $field,
                fri_proof: FriProofHandle) -> eIcicleError;
        }

        impl Fri {
            pub fn from_handle(_handle: FriHandle) -> Self {
                Self { handle: _handle }
            }

            pub fn new(
                input_size: u64,
                folding_factor: u64,
                stopping_degree: u64,
                merkle_tree_leaves_hash: &Hasher,
                merkle_tree_compress_hash: &Hasher,
                output_store_min_layer: u64
            ) -> Result<Self, eIcicleError> {
                let handle: FriHandle = unsafe { icicle_create_fri(
                    input_size,
                    folding_factor,
                    stopping_degree,
                    merkle_tree_leaves_hash.handle,
                    merkle_tree_compress_hash.handle,
                    output_store_min_layer
                ) };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(Self::from_handle(handle))
            }

            pub fn get_proof(
                &self,
                fri_config: &FriConfig,
                fri_transcript_config: &FriTranscriptConfig<$field>,
                input_data: &(impl HostOrDeviceSlice<$field> + ?Sized),
            ) -> Result<FriProof, eIcicleError> {
                let fri_proof = FriProof::default_wrapped()?;
                unsafe {
                    let err = icicle_fri_get_proof(self.handle, fri_config as *const FriConfig, fri_transcript_config.handle, input_data.as_ptr() as *const $field, fri_proof.handle);
                    err.wrap_value(fri_proof)
                }

            }
        }

        impl Drop for Fri {
            fn drop(&mut self) {
                unsafe {
                    icicle_delete_fri(self.handle);
                }
            }
        }
    }
}