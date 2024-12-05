#[doc(hidden)]
pub mod tests;

use crate::{hash::Hasher, traits::FieldImpl};
use icicle_runtime::errors::eIcicleError;

/// Trait to define the behavior of a Poseidon2 hasher for different field types.
/// This allows the implementation of Poseidon2 hashing for various field types that implement `FieldImpl`.
pub trait Poseidon2Hasher<F: FieldImpl> {
    /// Method to create a new Poseidon2 hasher for a given t (branching factor).
    fn new(t: u32, domain_tag: Option<&F>) -> Result<Hasher, eIcicleError>;
}

/// Function to create a Poseidon2 hasher for a specific field type and t (branching factor).
/// Delegates the creation to the `new` method of the `Poseidon2Hasher` trait.
pub fn create_poseidon2_hasher<F>(t: u32, domain_tag: Option<&F>) -> Result<Hasher, eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Poseidon2Hasher<F>, // Requires that the `Config` associated with `F` implements `Poseidon2Hasher`.
{
    <<F as FieldImpl>::Config as Poseidon2Hasher<F>>::new(t, domain_tag)
}

pub struct Poseidon2;

impl Poseidon2 {
    pub fn new<F>(t: u32, domain_tag: Option<&F>) -> Result<Hasher, eIcicleError>
    where
        F: FieldImpl,                  // F must implement the FieldImpl trait
        F::Config: Poseidon2Hasher<F>, // The Config associated with F must implement Poseidon2Hasher<F>
    {
        create_poseidon2_hasher::<F>(t, domain_tag)
    }
}

#[macro_export]
macro_rules! impl_poseidon2 {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_cfg:ident
    ) => {
        mod $field_prefix_ident {
            use crate::poseidon2::{$field, $field_cfg};
            use icicle_core::{
                hash::{Hasher, HasherHandle},
                poseidon2::Poseidon2Hasher,
                traits::FieldImpl,
            };
            use icicle_runtime::errors::eIcicleError;
            use std::marker::PhantomData;

            extern "C" {
                #[link_name = concat!($field_prefix, "_create_poseidon2_hasher")]
                fn create_poseidon2_hasher(t: u32, domain_tag: *const $field) -> HasherHandle;
            }

            // Implement the `Poseidon2Hasher` trait for the given field configuration.
            impl Poseidon2Hasher<$field> for $field_cfg {
                fn new(t: u32, domain_tag: Option<&$field>) -> Result<Hasher, eIcicleError> {
                    let handle: HasherHandle = unsafe {
                        create_poseidon2_hasher(t, domain_tag.map_or(std::ptr::null(), |tag| tag as *const $field))
                    }; // Calls the external FFI function to create the hasher.
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
macro_rules! impl_poseidon2_tests {
    (
      $field:ident
    ) => {
        use super::*;

        use icicle_runtime::test_utilities;
        use std::sync::Once;

        static INIT: Once = Once::new();

        pub fn initialize() {
            INIT.call_once(move || {
                test_utilities::test_load_and_init_devices();
            });
        }

        #[test]
        fn test_poseidon2_hash() {
            initialize();
            check_poseidon2_hash::<$field>();
        }

        // TODO uncomment when feature is ready
        // #[test]
        // fn test_poseidon2_hash_sponge() {
        //     initialize();
        //     check_poseidon2_hash_sponge::<$field>();
        // }

        #[test]
        fn test_poseidon2_hash_multi_device() {
            initialize();
            test_utilities::test_set_main_device();
            let nof_devices = icicle_runtime::get_device_count().unwrap();
            if nof_devices > 1 {
                check_poseidon2_hash_multi_device::<$field>();
            } else {
                println!("Skipping test_poseidon2_hash_multi_device due to single device in the machine");
            }
        }

        #[test]
        fn test_poseidon2_tree() {
            initialize();
            test_utilities::test_set_main_device();
            check_poseidon2_tree::<$field>();
            test_utilities::test_set_ref_device();
            check_poseidon2_tree::<$field>();
        }
    };
}
