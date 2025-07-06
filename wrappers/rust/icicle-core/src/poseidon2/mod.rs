#[doc(hidden)]
pub mod tests;

use crate::{field::Field, hash::Hasher};
use icicle_runtime::errors::eIcicleError;

/// Trait to define the behavior of a Poseidon2 hasher for different field types.
/// This allows the implementation of Poseidon2 hashing for various field types that implement `PrimeField`.
pub trait Poseidon2Hasher: Field {
    /// Creates a Poseidon2 hasher with an explicit `input_size` (rate).
    fn new_with_input_size(t: u32, domain_tag: Option<&Self>, input_size: u32) -> Result<Hasher, eIcicleError>;

    /// Convenience constructor that forwards to `new_with_input_size` with
    /// `input_size = 0` (backend default).
    fn new(t: u32, domain_tag: Option<&Self>) -> Result<Hasher, eIcicleError> {
        Self::new_with_input_size(t, domain_tag, 0)
    }
}

/// Function to create a Poseidon2 hasher for a specific field type and t (branching factor).
/// Delegates the creation to the `new` method of the `Poseidon2Hasher` trait.
pub fn create_poseidon2_hasher<F>(t: u32, domain_tag: Option<&F>) -> Result<Hasher, eIcicleError>
where
    F: Field,
    F: Poseidon2Hasher, // Requires that the `Config` associated with `F` implements `Poseidon2Hasher`.
{
    <F as Poseidon2Hasher>::new(t, domain_tag)
}

pub struct Poseidon2;

impl Poseidon2 {
    pub fn new<F>(t: u32, domain_tag: Option<&F>) -> Result<Hasher, eIcicleError>
    where
        F: Field + Poseidon2Hasher, // F must implement the Field trait
    {
        create_poseidon2_hasher::<F>(t, domain_tag)
    }

    pub fn new_with_input_size<F>(t: u32, domain_tag: Option<&F>, input_size: u32) -> Result<Hasher, eIcicleError>
    where
        F: Field,
        F: Poseidon2Hasher, // The Config associated with F must implement Poseidon2Hasher<F>
    {
        <F as Poseidon2Hasher>::new_with_input_size(t, domain_tag, input_size)
    }
}

#[macro_export]
macro_rules! impl_poseidon2 {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        mod $field_prefix_ident {
            use crate::poseidon2::$field;
            use icicle_core::{
                field::Field,
                hash::{Hasher, HasherHandle},
                poseidon2::Poseidon2Hasher,
            };
            use icicle_runtime::errors::eIcicleError;
            use std::marker::PhantomData;

            extern "C" {
                #[link_name = concat!($field_prefix, "_create_poseidon2_hasher")]
                fn create_poseidon2_hasher(t: u32, domain_tag: *const $field, input_size: u32) -> HasherHandle;
            }

            // Implement the `Poseidon2Hasher` trait for the given field configuration.
            impl Poseidon2Hasher for $field {
                fn new_with_input_size(
                    t: u32,
                    domain_tag: Option<&$field>,
                    input_size: u32,
                ) -> Result<Hasher, eIcicleError> {
                    let handle: HasherHandle = unsafe {
                        create_poseidon2_hasher(
                            t,
                            domain_tag.map_or(std::ptr::null(), |tag| tag as *const $field),
                            input_size,
                        )
                    };
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

        #[test]
        fn test_poseidon2_hash_sponge() {
            initialize();
            check_poseidon2_hash_sponge::<$field>();
        }

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
