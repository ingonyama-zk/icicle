#[doc(hidden)]
pub mod tests;

use std::ffi::c_void;

pub type PoseidonConstantsHandle = *const c_void;

#[macro_export]
macro_rules! impl_poseidon {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_cfg:ident
    ) => {
        mod $field_prefix_ident {
            use crate::poseidon::{$field, $field_cfg};
            use icicle_core::{
                hash::{Hasher, HasherHandle},
                poseidon::PoseidonConstantsHandle,
                traits::FieldImpl,
            };
            use icicle_runtime::errors::eIcicleError;
            use std::marker::PhantomData;

            pub struct Poseidon {}
            pub struct PoseidonConstants {
                handle: PoseidonConstantsHandle,
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "_poseidon_init_constants")]
                fn poseidon_init_constants(
                    arity: u32,
                    alpha: u32,
                    full_rounds_half: u32,
                    partial_rounds: u32,
                    round_constants: *const $field,
                    mds_matrix: *const $field,
                    non_sparse_matrix: *const $field,
                    sparse_matrices: *const $field,
                    domain_tag: *const $field,
                    constants_out: *mut PoseidonConstantsHandle,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_poseidon_init_default_constants")]
                fn poseidon_init_default_constants(constants_out: *mut PoseidonConstantsHandle) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_poseidon_delete_constants")]
                fn poseidon_delete_constants(handle: PoseidonConstantsHandle) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_create_poseidon_hasher")]
                fn create_poseidon_hasher(handle: PoseidonConstantsHandle) -> HasherHandle;
            }

            impl PoseidonConstants {
                // init for any valid arity
                pub fn default() -> Result<Self, eIcicleError> {
                    let mut poseidon_constants_handle: PoseidonConstantsHandle = std::ptr::null_mut();
                    let err: eIcicleError = unsafe { poseidon_init_default_constants(&mut poseidon_constants_handle) };
                    if err == eIcicleError::Success {
                        Ok(Self {
                            handle: poseidon_constants_handle,
                        })
                    } else {
                        Err(err)
                    }
                }

                // TODO: can we do it for any supported arity?
                fn new(
                    arity: u32,
                    alpha: u32,
                    full_rounds_half: u32,
                    partial_rounds: u32,
                    round_constants: &[$field],
                    mds_matrix: &[$field],
                    non_sparse_matrix: &[$field],
                    sparse_matrices: &[$field],
                    domain_tag: &[$field],
                ) -> Result<Self, eIcicleError> {
                    let mut poseidon_constants_handle: PoseidonConstantsHandle = std::ptr::null_mut();
                    let err: eIcicleError = unsafe {
                        poseidon_init_constants(
                            arity,
                            alpha,
                            full_rounds_half,
                            partial_rounds,
                            round_constants as *const _ as *const $field,
                            mds_matrix as *const _ as *const $field,
                            non_sparse_matrix as *const _ as *const $field,
                            sparse_matrices as *const _ as *const $field,
                            domain_tag as *const _ as *const $field,
                            &mut poseidon_constants_handle,
                        )
                    };
                    if err == eIcicleError::Success {
                        Ok(Self {
                            handle: poseidon_constants_handle,
                        })
                    } else {
                        Err(err)
                    }
                }
            }

            impl Drop for PoseidonConstants {
                fn drop(&mut self) {
                    if (!self
                        .handle
                        .is_null())
                    {
                        unsafe {
                            poseidon_delete_constants(self.handle);
                        }
                    }
                }
            }

            impl Poseidon {
                pub fn new(constants: &PoseidonConstants) -> Result<Hasher, eIcicleError> {
                    let handle: HasherHandle = unsafe { create_poseidon_hasher(constants.handle) };
                    if handle.is_null() {
                        return Err(eIcicleError::UnknownError);
                    }
                    Ok(Hasher::from_handle(handle))
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_poseidon_tests {
    (
      $field_prefix_ident:ident,
      $field:ident
    ) => {
        use super::*;
        use crate::poseidon::$field_prefix_ident::{Poseidon, PoseidonConstants};
        use icicle_core::hash::Hasher;

        #[test]
        fn test_poseidon_hash() {
            let hasher: Hasher;
            {
                let poseidon_constants = PoseidonConstants::default().unwrap();
                // ownership of the poseidon_constants is shared with the hasher here so the poseidon_constants can be dropped
                hasher = Poseidon::new(&poseidon_constants).unwrap();
            }
            check_poseidon_hash::<$field>(&hasher);
        }
    };
}
