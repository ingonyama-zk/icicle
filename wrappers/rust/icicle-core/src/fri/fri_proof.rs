use std::mem::ManuallyDrop;

use icicle_runtime::eIcicleError;

use crate::{
    merkle::MerkleProof,
    traits::{FieldImpl, Handle},
};

pub struct FriProofExposed<F: FieldImpl> {
    pub query_proofs: Vec<Vec<ManuallyDrop<MerkleProof>>>,
    pub final_poly: Vec<F>,
    pub pow_nonce: u64
}

pub trait FriProofTrait<F: FieldImpl>: Sized + Handle + From<FriProofExposed<F>>
where
    F: FieldImpl,
{
    /// Constructs a new instance of the FRI proof.
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

    fn expose(&self) -> Result<FriProofExposed<F>, eIcicleError>;
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
        use icicle_core::fri::fri_proof::FriProofExposed;

        pub type FriProofHandle = *mut c_void;

        extern "C" {
            #[link_name = concat!($field_prefix, "_icicle_initialize_fri_proof")]
            fn icicle_initialize_fri_proof() -> FriProofHandle;

            #[link_name = concat!($field_prefix, "_icicle_initialize_with_arguments_fri_proof")]
            fn initialize_with_arguments_fri_proof(proofs: *const *const MerkleProofHandle, final_poly: *const $field, pow_nonce: u64) -> FriProofHandle;

            #[link_name = concat!($field_prefix, "_icicle_delete_fri_proof")]
            fn icicle_delete_fri_proof(handle: FriProofHandle) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_pow_nonce")]
            fn fri_proof_get_pow_nonce(handle: FriProofHandle, result: *mut u64) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_final_poly_size")]
            fn fri_proof_get_final_poly_size(handle: FriProofHandle, result: *mut usize) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_final_poly")]
            fn fri_proof_get_final_poly(handle: FriProofHandle, final_poly: *mut *const $field) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_nof_queries")]
            fn fri_proof_get_nof_queries(
                handle: FriProofHandle,
                nof_queries: *mut usize,
            ) -> eIcicleError;
            #[link_name = concat!($field_prefix, "_fri_proof_get_nof_rounds")]
            fn fri_proof_get_nof_rounds(
                handle: FriProofHandle,
                nof_rounds: *mut usize,
            ) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_fri_proof_get_round_proofs_for_query")]
            fn fri_proof_get_round_proofs_for_query(
                handle: FriProofHandle,
                query_idx: usize,
                proofs: *mut *const MerkleProofHandle,
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

            fn get_query_proofs(&self) -> Result<Vec<Vec<ManuallyDrop<MerkleProof>>>, eIcicleError> {
                let mut nof_queries: usize = 0;
                let mut nof_rounds: usize = 0;
                unsafe {
                    fri_proof_get_nof_queries(self.handle, &mut nof_queries).wrap()?;
                    fri_proof_get_nof_rounds(self.handle, &mut nof_rounds).wrap()?;
                    let mut proofs: Vec<Vec<ManuallyDrop<MerkleProof>>> = Vec::with_capacity(nof_queries as usize);
                    for i in 0..nof_queries {
                        let mut proof: *const MerkleProofHandle = std::ptr::null();
                        fri_proof_get_round_proofs_for_query(self.handle, i, &mut proof).wrap()?;
                        let proofs_per_query = slice::from_raw_parts(proof, nof_rounds as usize);
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

            fn expose(&self) -> Result<FriProofExposed<$field>, eIcicleError> {
                let query_proofs = self.get_query_proofs()?;
                let final_poly = self.get_final_poly()?;
                let pow_nonce = self.get_pow_nonce()?;
                Ok(FriProofExposed{query_proofs, final_poly, pow_nonce})
            }
        }

        impl From<FriProofExposed<$field>> for FriProof {
            fn from(proof_exposed: FriProofExposed<$field>) -> FriProof {
                let handle_vectors: Vec<Vec<MerkleProofHandle>> = proof_exposed.query_proofs
                    .iter()
                    .map(|inner| inner.iter().map(|proof| proof.handle()).collect())
                    .collect();
                let vec_of_pointers: Vec<*const MerkleProofHandle> = handle_vectors
                    .iter()
                    .map(|vec| vec.as_ptr())
                    .collect();
                let handle: FriProofHandle = unsafe {initialize_with_arguments_fri_proof(vec_of_pointers.as_ptr(), proof_exposed.final_poly.as_ptr(), proof_exposed.pow_nonce)};
                if handle.is_null() {
                    panic!("Couldn't convert into FriProof");
                }
                FriProof{handle}
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
