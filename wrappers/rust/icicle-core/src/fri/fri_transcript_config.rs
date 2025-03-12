use crate::hash::Hasher;
use crate::traits::FieldConfig;
use crate::traits::FieldImpl;
use crate::traits::Handle;
use icicle_runtime::eIcicleError;

pub trait FriTranscriptConfigTrait<F: FieldImpl>: Sized + Default + Handle
where
    Self::Field: FieldImpl,
    Self::FieldConfig: FieldConfig,
{
    type Field;
    type FieldConfig;

    fn new(hasher: Hasher, seed_rng: &Self::Field) -> Result<Self, eIcicleError>; // TODO: add labels here

    fn default_wrapped() -> Result<Self, eIcicleError>;
}

#[macro_export]
macro_rules! impl_fri_transcript_config {
    (
        $field_prefix:literal,
        $field:ident,
        $field_config:ident
    ) => {
        use icicle_core::traits::Handle;
        use std::ffi::c_void;

        pub type FriTranscriptConfigHandle = *const c_void;

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

        pub struct FriTranscriptConfig {
            handle: FriTranscriptConfigHandle,
        }

        impl FriTranscriptConfigTrait<$field> for FriTranscriptConfig {
            type Field = $field;
            type FieldConfig = $field_config;

            fn new(hasher: Hasher, seed_rng: &$field) -> Result<Self, eIcicleError> {
                // TODO: add labels here
                let handle: FriTranscriptConfigHandle =
                    unsafe { icicle_create_fri_transcript_config(hasher.handle, seed_rng as *const $field) };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(Self { handle })
            }

            fn default_wrapped() -> Result<Self, eIcicleError> {
                let handle: FriTranscriptConfigHandle = unsafe { icicle_create_default_fri_transcript_config() };
                if handle.is_null() {
                    return Err(eIcicleError::UnknownError);
                }
                Ok(Self { handle })
            }
        }

        impl Handle for FriTranscriptConfig {
            fn handle(&self) -> *const c_void {
                self.handle
            }
        }

        impl Default for FriTranscriptConfig {
            fn default() -> Self {
                Self::default_wrapped().unwrap()
            }
        }

        impl Drop for FriTranscriptConfig {
            fn drop(&mut self) {
                unsafe {
                    if !self
                        .handle
                        .is_null()
                    {
                        icicle_delete_fri_transcript_config(self.handle);
                    }
                }
            }
        }
    };
}
