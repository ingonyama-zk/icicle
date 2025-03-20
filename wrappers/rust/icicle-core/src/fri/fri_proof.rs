use std::ffi::c_void;

use icicle_runtime::eIcicleError;

use crate::traits::{FieldImpl, Handle};

pub struct PointerArray {
    pub ptr: *const c_void,
    pub element_size: usize,
    pub len: usize
}

impl PointerArray {
    pub unsafe fn get_untyped<T>(&self, index: usize) -> &T {
        assert!(index < self.len, "Index out of bounds");

        // Calculate offset: (ptr + index * element_size) and cast to T
        let byte_ptr = (self.ptr as *const u8).add(index * self.element_size);
        &*(byte_ptr as *const T)
    }
}

impl PointerArray {
    pub unsafe fn get(&self, index: usize) -> *const c_void {
        assert!(index < self.len, "Index out of bounds");
        let byte_ptr = (self.ptr as *const u8).add(index * self.element_size);
        byte_ptr as *const c_void
    }
}

pub trait FriProofTrait<F: FieldImpl>: Sized + Handle
where
    F: FieldImpl,
{
    /// Constructs a new instance of the FRI proof.
    fn new() -> Self;

    /// Returns the query proofs.
    ///
    /// Note: `ManuallyDrop` is used because the values are borrowed from the `FriProof`
    /// and will be dropped along with it.
    fn get_query_proofs(&self) -> Result<Vec<PointerArray>, eIcicleError>;

    /// Returns the final polynomial values.
    fn get_final_poly(&self) -> Result<Vec<F>, eIcicleError>;

    /// Returns the proof-of-work nonce.
    fn get_pow_nonce(&self) -> Result<u64, eIcicleError>;

    fn create_with_arguments(query_proofs: Vec<PointerArray>, final_poly: Vec<F>, pow_nonce: u64) -> Self;
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
        use icicle_core::fri::fri_proof::PointerArray;

        pub type FriProofHandle = *mut c_void;

        extern "C" {
            #[link_name = concat!($field_prefix, "_icicle_initialize_fri_proof")]
            fn icicle_initialize_fri_proof() -> FriProofHandle;

            #[link_name = concat!($field_prefix, "_icicle_create_with_arguments_fri_proof")]
            fn create_with_arguments_fri_proof(proofs: *const *const c_void, nof_queries: usize, nof_rounds: usize, final_poly: *const $field, final_poly_size: usize, pow_nonce: u64) -> FriProofHandle;

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
                proofs: *mut MerkleProofHandle,
                element_size: *mut usize
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

            fn create_with_arguments(query_proofs: Vec<PointerArray>, final_poly: Vec<$field>, pow_nonce: u64) -> FriProof {
                let handle_vectors: Vec<*const c_void> = query_proofs.iter().map(|x| x.ptr).collect();
                let handle: FriProofHandle = unsafe {create_with_arguments_fri_proof(handle_vectors.as_ptr(), query_proofs.len(), query_proofs[0].len, final_poly.as_ptr(), final_poly.len(), pow_nonce)};
                if handle.is_null() {
                    panic!("Couldn't convert into FriProof");
                }
                FriProof{handle}
            }

            fn get_query_proofs(&self) -> Result<Vec<PointerArray>, eIcicleError> {
                let mut nof_queries: usize = 0;
                let mut nof_rounds: usize = 0;
                unsafe {
                    fri_proof_get_nof_queries(self.handle, &mut nof_queries).wrap()?;
                    fri_proof_get_nof_rounds(self.handle, &mut nof_rounds).wrap()?;
                    let mut proofs: Vec<PointerArray> = Vec::with_capacity(nof_queries as usize);
                    for i in 0..nof_queries {
                        let mut proofs_per_query: *const c_void = std::ptr::null();
                        let mut size: usize = 0;
                        fri_proof_get_round_proofs_for_query(self.handle, i, &mut proofs_per_query, &mut size).wrap()?;                        proofs.push(
                            PointerArray{ ptr: proofs_per_query, element_size: size, len: nof_rounds}
                        );
                        // let second = proofs[i].get(2);
                        // let p = MerkleProof::from_handle(second.clone());
                        // let leaf = p.get_leaf::<u8>();
                        // println!("{:?}", leaf);
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
