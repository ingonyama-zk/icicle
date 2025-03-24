use std::mem::ManuallyDrop;

use icicle_runtime::eIcicleError;

use crate::{
    merkle::MerkleProof,
    traits::{FieldImpl, Handle},
};

pub trait FriProofTrait<F: FieldImpl>: Sized + Handle
where
    F: FieldImpl,
{
    /// Constructs a new instance of the `FriProof`.
    fn new() -> Self;

    /// Returns the query proofs.
    ///
    /// Note: `ManuallyDrop` is used because the values are borrowed from the `FriProof`
    /// and will be dropped along with it.
    fn get_query_proofs(&self) -> Result<Vec<Vec<ManuallyDrop<MerkleProof>>>, eIcicleError>;

    /// Returns the final polynomial values.
    fn get_final_poly(&self) -> Result<Vec<F>, eIcicleError>;

    /// Returns the proof-of-work nonce.
    fn get_pow_nonce(&self) -> Result<u64, eIcicleError>;

    /// Creates a new instance of `FriProof` with the given proof data and wraps it in `ManuallyDrop`.
    fn create_with_arguments(
        query_proofs: Vec<Vec<ManuallyDrop<MerkleProof>>>,
        final_poly: Vec<F>,
        pow_nonce: u64,
    ) -> ManuallyDrop<Self>;
}

#[macro_export]
macro_rules! impl_fri_proof {
    (
        $field_prefix:literal,
        $field:ident,
        $field_config:ident
    ) => {
        use icicle_core::fri::fri_proof::FriProofTrait;
        use icicle_core::merkle::{MerkleProof, MerkleProofHandle};
        use icicle_core::traits::Handle;
        use std::ffi::c_void;
        use std::mem::ManuallyDrop;
        use std::slice;

        pub type FriProofHandle = *mut c_void;

        extern "C" {
            #[link_name = concat!($field_prefix, "_icicle_initialize_fri_proof")]
            fn icicle_initialize_fri_proof() -> FriProofHandle;

            #[link_name = concat!($field_prefix, "_icicle_create_with_arguments_fri_proof")]
            fn create_with_arguments_fri_proof(
                proofs: *const *const MerkleProofHandle,
                nof_queries: usize,
                nof_rounds: usize,
                final_poly: *const $field,
                final_poly_size: usize,
                pow_nonce: u64,
            ) -> FriProofHandle;

            #[link_name = concat!($field_prefix, "_icicle_delete_fri_proof")]
            fn icicle_delete_fri_proof(handle: FriProofHandle) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_pow_nonce")]
            fn fri_proof_get_pow_nonce(handle: FriProofHandle, result: *mut u64) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_final_poly_size")]
            fn fri_proof_get_final_poly_size(handle: FriProofHandle, result: *mut usize) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_final_poly")]
            fn fri_proof_get_final_poly(handle: FriProofHandle, final_poly: *mut *const $field) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_nof_queries")]
            fn fri_proof_get_nof_queries(handle: FriProofHandle, nof_queries: *mut usize) -> eIcicleError;
            #[link_name = concat!($field_prefix, "_fri_proof_get_nof_rounds")]
            fn fri_proof_get_nof_rounds(handle: FriProofHandle, nof_rounds: *mut usize) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_round_proofs_for_query")]
            fn fri_proof_get_round_proofs_for_query(
                handle: FriProofHandle,
                query_idx: usize,
                proofs: *mut MerkleProofHandle,
            ) -> eIcicleError;
        }

        pub struct FriProof {
            handle: FriProofHandle,
        }

        impl FriProofTrait<$field> for FriProof {
            fn new() -> Self {
                let handle: FriProofHandle = unsafe { icicle_initialize_fri_proof() };
                if handle.is_null() {
                    panic!("Couldn't create FriProof");
                }
                Self { handle }
            }

            fn create_with_arguments(
                query_proofs: Vec<Vec<ManuallyDrop<MerkleProof>>>,
                final_poly: Vec<$field>,
                pow_nonce: u64,
            ) -> ManuallyDrop<FriProof> {
                let handle_vectors: Vec<Vec<MerkleProofHandle>> = query_proofs
                    .iter()
                    .map(|x| {
                        x.iter()
                            .map(|x| x.handle())
                            .collect()
                    })
                    .collect();
                let handle_vectors: Vec<*const MerkleProofHandle> = handle_vectors
                    .iter()
                    .map(|x| x.as_ptr())
                    .collect();

                let handle: FriProofHandle = unsafe {
                    create_with_arguments_fri_proof(
                        handle_vectors.as_ptr(),
                        query_proofs.len(),
                        query_proofs[0].len(),
                        final_poly.as_ptr(),
                        final_poly.len(),
                        pow_nonce,
                    )
                };
                if handle.is_null() {
                    panic!("Couldn't convert into FriProof");
                }
                ManuallyDrop::new(FriProof { handle })
            }

            fn get_query_proofs(&self) -> Result<Vec<Vec<ManuallyDrop<MerkleProof>>>, eIcicleError> {
                let mut nof_queries: usize = 0;
                let mut nof_rounds: usize = 0;
                unsafe {
                    fri_proof_get_nof_queries(self.handle, &mut nof_queries).wrap()?;
                    fri_proof_get_nof_rounds(self.handle, &mut nof_rounds).wrap()?;
                    let mut proofs: Vec<Vec<ManuallyDrop<MerkleProof>>> = Vec::with_capacity(nof_queries as usize);
                    for i in 0..nof_queries {
                        let mut proofs_per_query: Vec<MerkleProofHandle> = vec![std::ptr::null(); nof_rounds];
                        fri_proof_get_round_proofs_for_query(self.handle, i, proofs_per_query.as_mut_ptr()).wrap()?;
                        proofs.push(
                            proofs_per_query
                                .iter()
                                .map(|x| ManuallyDrop::new(MerkleProof::from_handle(*x)))
                                .collect(),
                        );
                    }
                    Ok(proofs)
                }
            }

            fn get_final_poly(&self) -> Result<Vec<$field>, eIcicleError> {
                let mut nof: usize = 0;
                unsafe {
                    fri_proof_get_final_poly_size(self.handle, &mut nof).wrap()?;
                    let mut final_poly_ptr: *const $field = std::ptr::null();
                    fri_proof_get_final_poly(self.handle, &mut final_poly_ptr).wrap()?;
                    Ok(slice::from_raw_parts(final_poly_ptr, nof as usize).to_vec())
                }
            }

            fn get_pow_nonce(&self) -> Result<u64, eIcicleError> {
                let mut nonce: u64 = 0;
                unsafe { fri_proof_get_pow_nonce(self.handle, &mut nonce).wrap_value(nonce) }
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
