#[doc(hidden)]
pub mod tests;

use crate::{hash::Hasher, traits::FieldImpl};
use icicle_runtime::errors::eIcicleError;
use std::marker::PhantomData;

pub struct PoseidonConstantsInitOptions<F: FieldImpl> {
    // TODO: Define the struct with fields such as arity, alpha, nof_rounds, mds_matrix, etc.
    // It must be compatible with FFI, so make sure to use only types like integers, arrays, and pointers.
    phantom: PhantomData<F>,
}

pub trait PoseidonHasher<F: FieldImpl> {
    fn initialize_constants(options: &PoseidonConstantsInitOptions<F>) -> Result<(), eIcicleError>;
    fn initialize_default_constants() -> Result<(), eIcicleError>;
    fn new(arity: u32) -> Result<Hasher, eIcicleError>;
}

pub fn initialize_default_poseidon_constants<F>() -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    <<F as FieldImpl>::Config as PoseidonHasher<F>>::initialize_default_constants()
}

pub fn initialize_poseidon_constants<F>(_options: &PoseidonConstantsInitOptions<F>) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    return Err(eIcicleError::ApiNotImplemented);
    // TODO define PoseidonConstantsInitOptions and implement
    // <<F as FieldImpl>::Config as PoseidonHasher<F>>::initialize_constants(_options)
}

pub fn create_poseidon_hasher<F>(arity: u32) -> Result<Hasher, eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    <<F as FieldImpl>::Config as PoseidonHasher<F>>::new(arity)
}

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
                poseidon::{PoseidonConstantsInitOptions, PoseidonHasher},
                traits::FieldImpl,
            };
            use icicle_runtime::errors::eIcicleError;
            use std::marker::PhantomData;

            extern "C" {
                #[link_name = concat!($field_prefix, "_poseidon_init_default_constants")]
                fn poseidon_init_default_constants() -> eIcicleError;

                #[link_name = concat!($field_prefix, "_poseidon_init_constants")]
                fn poseidon_init_constants(options: *const PoseidonConstantsInitOptions<$field>) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_create_poseidon_hasher")]
                fn create_poseidon_hasher(arity: u32) -> HasherHandle;
            }

            impl PoseidonHasher<$field> for $field_cfg {
                fn initialize_default_constants() -> Result<(), eIcicleError> {
                    unsafe { poseidon_init_default_constants().wrap() }
                }

                fn initialize_constants(options: &PoseidonConstantsInitOptions<$field>) -> Result<(), eIcicleError> {
                    unsafe { poseidon_init_constants(options as *const PoseidonConstantsInitOptions<$field>).wrap() }
                }

                fn new(arity: u32) -> Result<Hasher, eIcicleError> {
                    let handle: HasherHandle = unsafe { create_poseidon_hasher(arity) };
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

        use std::sync::Once;

        static INIT: Once = Once::new();

        pub fn initialize() {
            INIT.call_once(move || {
                // TODO load CUDA backend
                // test_utilities::test_load_and_init_devices();
            });
        }

        #[test]
        fn test_poseidon_hash() {
            check_poseidon_hash::<$field>();
        }

        #[test]
        fn test_poseidon_tree() {
            check_poseidon_tree::<$field>();
        }
    };
}
