use crate::traits::{FieldConfig, FieldImpl, Handle};

pub trait FriProofTrait<F: FieldImpl>: Sized + Handle
// TODO TIMUR: MerkleProof deref
where
    Self::Field: FieldImpl,
    Self::FieldConfig: FieldConfig,
{
    type Field;
    type FieldConfig;

    fn new() -> Self;
}

#[macro_export]
macro_rules! impl_fri_proof {
    (
        $field_prefix:literal,
        $field:ident,
        $field_config:ident
    ) => {
        use icicle_core::fri::fri_proof::FriProofTrait;

        pub type FriProofHandle = *mut c_void;

        extern "C" {
            #[link_name = concat!($field_prefix, "_icicle_initialize_fri_proof")]
            fn icicle_initialize_fri_proof() -> FriProofHandle;

            fn icicle_delete_fri_proof(handle: FriProofHandle) -> eIcicleError;
        }

        pub struct FriProof {
            handle: FriProofHandle,
        }

        impl FriProofTrait<$field> for FriProof {
            type Field = $field;
            type FieldConfig = $field_config;

            fn new() -> Self {
                let handle: FriProofHandle = unsafe { icicle_initialize_fri_proof() };
                Self { handle }
            }
        }

        impl Drop for FriProof {
            fn drop(&mut self) {
                unsafe {
                    if !self
                        .handle
                        .is_null()
                    {
                        icicle_delete_fri_proof(self.handle);
                    }
                }
            }
        }

        impl Handle for FriProof {
            fn handle(&self) -> *const c_void {
                self.handle
            }
        }
    };
}
