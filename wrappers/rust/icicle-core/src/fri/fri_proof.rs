use icicle_runtime::eIcicleError;

use crate::{
    merkle::MerkleProofData,
    traits::{FieldImpl, Handle, Serialization},
};

pub trait FriProofTrait<F: FieldImpl>: Sized + Handle + Serialization
where
    F: FieldImpl,
{
    /// Constructs a new instance of the `FriProof`.
    fn new() -> Self;

    /// Returns the query proofs.
    fn get_query_proofs<T: Clone>(&self) -> Result<Vec<Vec<MerkleProofData<T>>>, eIcicleError>;

    /// Returns the final polynomial values.
    fn get_final_poly(&self) -> Result<Vec<F>, eIcicleError>;

    /// Returns the proof-of-work nonce.
    fn get_pow_nonce(&self) -> Result<u64, eIcicleError>;

    /// Creates a new instance of `FriProof` with the given proof data and wraps it in `ManuallyDrop`.
    fn create_with_arguments<T: Clone>(
        query_proofs_data: Vec<Vec<MerkleProofData<T>>>,
        final_poly: Vec<F>,
        pow_nonce: u64,
    ) -> Result<Self, eIcicleError>;
}

#[macro_export]
macro_rules! impl_fri_proof {
    (
        $field_prefix:literal,
        $field:ident,
        $field_config:ident
    ) => {
        use icicle_core::{
            fri::fri_proof::FriProofTrait,
            merkle::{MerkleProof, MerkleProofData, MerkleProofHandle},
            traits::{Handle, Serialization},
        };
        use std::{ffi::c_void, mem::ManuallyDrop, slice};

        pub type FriProofHandle = *const c_void;

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

            #[link_name = concat!($field_prefix, "_fri_proof_get_serialized_size")]
            fn fri_proof_get_serialized_size(handle: FriProofHandle, size: *mut usize) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_serialize")]
            fn fri_proof_serialize(handle: FriProofHandle, buffer: *mut u8, size: usize) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_deserialize")]
            fn fri_proof_deserialize(handle: *mut FriProofHandle, buffer: *const u8, size: usize) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_serialize_to_file")]
            fn fri_proof_serialize_to_file(
                handle: FriProofHandle,
                filename: *const u8,
                filename_len: usize,
            ) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_deserialize_from_file")]
            fn fri_proof_deserialize_from_file(
                handle: *mut FriProofHandle,
                filename: *const u8,
                filename_len: usize,
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

            fn create_with_arguments<T: Clone>(
                query_proofs_data: Vec<Vec<MerkleProofData<T>>>,
                final_poly: Vec<$field>,
                pow_nonce: u64,
            ) -> Result<Self, eIcicleError> {
                let query_proofs = query_proofs_data
                    .into_iter()
                    .map(|query_vec| {
                        query_vec
                            .into_iter()
                            .map(|proof_data| Ok(ManuallyDrop::new(MerkleProof::try_from(proof_data)?)))
                            .collect::<Result<Vec<ManuallyDrop<MerkleProof>>, eIcicleError>>()
                    })
                    .collect::<Result<Vec<Vec<ManuallyDrop<MerkleProof>>>, eIcicleError>>()?;
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
                    return Err(eIcicleError::InvalidArgument);
                }
                Ok(Self { handle })
            }

            fn get_query_proofs<T: Clone>(&self) -> Result<Vec<Vec<MerkleProofData<T>>>, eIcicleError> {
                let mut nof_queries: usize = 0;
                let mut nof_rounds: usize = 0;
                unsafe {
                    fri_proof_get_nof_queries(self.handle, &mut nof_queries).wrap()?;
                    fri_proof_get_nof_rounds(self.handle, &mut nof_rounds).wrap()?;
                    let mut proofs: Vec<Vec<MerkleProofData<T>>> = Vec::with_capacity(nof_queries as usize);
                    for i in 0..nof_queries {
                        let mut proofs_per_query: Vec<MerkleProofHandle> = vec![std::ptr::null(); nof_rounds];
                        fri_proof_get_round_proofs_for_query(self.handle, i, proofs_per_query.as_mut_ptr()).wrap()?;
                        proofs.push(
                            proofs_per_query
                                .iter()
                                .map(|x| {
                                    let proof_manually_dropped = ManuallyDrop::new(MerkleProof::from_handle(*x));
                                    let proof = &*(&*proof_manually_dropped as *const MerkleProof);
                                    let proof_data = MerkleProofData::<T>::from(proof);
                                    proof_data
                                })
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

        impl Serialization for FriProof {
            fn get_serialized_size(&self) -> Result<usize, eIcicleError> {
                let mut size: usize = 0;
                unsafe { fri_proof_get_serialized_size(self.handle, &mut size).wrap_value(size) }
            }

            fn serialize(&self) -> Result<Vec<u8>, eIcicleError> {
                let mut buffer: Vec<u8> = vec![0; self.get_serialized_size()?];
                unsafe { fri_proof_serialize(self.handle, buffer.as_mut_ptr(), buffer.len()).wrap_value(buffer) }
            }

            fn deserialize(buffer: &[u8]) -> Result<Self, eIcicleError> {
                let mut handle: FriProofHandle = std::ptr::null();
                unsafe { fri_proof_deserialize(&mut handle, buffer.as_ptr(), buffer.len()).wrap_value(Self { handle }) }
            }

            fn serialize_to_file(&self, filename: &str) -> Result<(), eIcicleError> {
                unsafe { fri_proof_serialize_to_file(self.handle, filename.as_ptr(), filename.len()).wrap() }
            }

            fn deserialize_from_file(filename: &str) -> Result<Self, eIcicleError> {
                let mut handle: FriProofHandle = std::ptr::null();
                unsafe {
                    fri_proof_deserialize_from_file(&mut handle, filename.as_ptr(), filename.len())
                        .wrap_value(Self { handle })
                }
            }
        }
    };
}
