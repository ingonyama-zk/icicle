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

/// Trait to define the behavior of a Poseidon hasher for different field types.
/// This allows the implementation of Poseidon hashing for various field types that implement `FieldImpl`.
pub trait PoseidonHasher<F: FieldImpl> {
    /// Method to initialize Poseidon constants with user-defined options.
    fn initialize_constants(options: &PoseidonConstantsInitOptions<F>) -> Result<(), eIcicleError>;

    /// Method to initialize Poseidon constants with default values.
    fn initialize_default_constants() -> Result<(), eIcicleError>;

    /// Method to create a new Poseidon hasher for a given arity (branching factor).
    fn new(arity: u32) -> Result<Hasher, eIcicleError>;
}

/// Function to initialize Poseidon constants based on user-defined options for a specific field type.
/// Currently, this function returns an error as the feature is not yet implemented.
/// TODO: Define PoseidonConstantsInitOptions and implement the function logic.
pub fn initialize_poseidon_constants<F>(_options: &PoseidonConstantsInitOptions<F>) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonHasher<F>, // Requires that the `Config` associated with `F` implements `PoseidonHasher`.
{
    return Err(eIcicleError::ApiNotImplemented); // Placeholder error until the function is implemented.
}

/// Function to create a Poseidon hasher for a specific field type and arity (branching factor).
/// Delegates the creation to the `new` method of the `PoseidonHasher` trait.
pub fn create_poseidon_hasher<F>(arity: u32) -> Result<Hasher, eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: PoseidonHasher<F>, // Requires that the `Config` associated with `F` implements `PoseidonHasher`.
{
    <<F as FieldImpl>::Config as PoseidonHasher<F>>::new(arity)
}

pub struct Poseidon;

impl Poseidon {
    pub fn new<F>(arity: u32) -> Result<Hasher, eIcicleError>
    where
        F: FieldImpl,                 // F must implement the FieldImpl trait
        F::Config: PoseidonHasher<F>, // The Config associated with F must implement PoseidonHasher<F>
    {
        create_poseidon_hasher::<F>(arity)
    }
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

            // Implement the `PoseidonHasher` trait for the given field configuration.
            impl PoseidonHasher<$field> for $field_cfg {
                fn initialize_default_constants() -> Result<(), eIcicleError> {
                    unsafe { poseidon_init_default_constants().wrap() } // Calls the external FFI function and wraps the result.
                }

                fn initialize_constants(options: &PoseidonConstantsInitOptions<$field>) -> Result<(), eIcicleError> {
                    unsafe { poseidon_init_constants(options as *const PoseidonConstantsInitOptions<$field>).wrap() }
                    // Calls the external FFI function with user-defined options.
                }

                fn new(arity: u32) -> Result<Hasher, eIcicleError> {
                    let handle: HasherHandle = unsafe { create_poseidon_hasher(arity) }; // Calls the external FFI function to create the hasher.
                    if handle.is_null() {
                        return Err(eIcicleError::UnknownError); // Checks if the handle is null and returns an error if so.
                    }
                    Ok(Hasher::from_handle(handle)) // Wraps the handle in a `Hasher` object and returns it.
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_poseidon_tests {
    (
      $field:ident
    ) => {
        use super::*;

        use icicle_core::test_utilities;
        use std::sync::Once;

        static INIT: Once = Once::new();

        pub fn initialize() {
            INIT.call_once(move || {
                test_utilities::test_load_and_init_devices();
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
