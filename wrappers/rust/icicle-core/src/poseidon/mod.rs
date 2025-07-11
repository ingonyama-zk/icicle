#[doc(hidden)]
pub mod tests;

use crate::{field::Field, hash::Hasher};
use icicle_runtime::IcicleError;

/// Trait to define the behavior of a Poseidon hasher for different field types.
/// This allows the implementation of Poseidon hashing for various field types that implement `Field`.
pub trait PoseidonHasher<F: Field> {
    /// Creates a Poseidon hasher with an explicit `input_size` (rate). This is the
    /// low-level constructor that all implementations must provide.
    fn new_with_input_size(t: u32, domain_tag: Option<&F>, input_size: u32) -> Result<Hasher, IcicleError>;

    /// Convenience constructor that forwards to `new_with_input_size` with
    /// `input_size = 0`, signalling the backend to use a default value.
    fn new(t: u32, domain_tag: Option<&F>) -> Result<Hasher, IcicleError> {
        Self::new_with_input_size(t, domain_tag, 0)
    }
}

/// Function to create a Poseidon hasher for a specific field type and t (branching factor).
/// Delegates the creation to the `new` method of the `PoseidonHasher` trait.
pub fn create_poseidon_hasher<F>(t: u32, domain_tag: Option<&F>) -> Result<Hasher, IcicleError>
where
    F: Field + PoseidonHasher<F>,
{
    <F as PoseidonHasher<F>>::new(t, domain_tag)
}

pub struct Poseidon;

impl Poseidon {
    pub fn new<F>(t: u32, domain_tag: Option<&F>) -> Result<Hasher, IcicleError>
    where
        F: Field + PoseidonHasher<F>, // F must implement the Field trait
    {
        create_poseidon_hasher::<F>(t, domain_tag)
    }

    pub fn new_with_input_size<F>(t: u32, domain_tag: Option<&F>, input_size: u32) -> Result<Hasher, IcicleError>
    where
        F: Field,
        F: PoseidonHasher<F>,
    {
        <F as PoseidonHasher<F>>::new_with_input_size(t, domain_tag, input_size)
    }
}

#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! impl_poseidon {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        mod $field_prefix_ident {
            use crate::poseidon::$field;
            use icicle_core::{
                field::Field,
                hash::{Hasher, HasherHandle},
                poseidon::PoseidonHasher,
            };
            use icicle_runtime::errors::{eIcicleError, IcicleError};
            use std::marker::PhantomData;

            extern "C" {
                #[link_name = concat!($field_prefix, "_create_poseidon_hasher")]
                fn create_poseidon_hasher(t: u32, domain_tag: *const $field, input_size: u32) -> HasherHandle;
            }

            // Implement the `PoseidonHasher` trait for the given field configuration.
            impl PoseidonHasher<$field> for $field {
                fn new_with_input_size(
                    t: u32,
                    domain_tag: Option<&$field>,
                    input_size: u32,
                ) -> Result<Hasher, IcicleError> {
                    let handle: HasherHandle = unsafe {
                        create_poseidon_hasher(
                            t,
                            domain_tag.map_or(std::ptr::null(), |tag| tag as *const $field),
                            input_size,
                        )
                    };
                    if handle.is_null() {
                        return Err(IcicleError::new(
                            eIcicleError::UnknownError,
                            "poseidon hasher handle is null",
                        ));
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
        fn test_poseidon_hash() {
            initialize();
            check_poseidon_hash::<$field>();
        }

        #[test]
        fn test_poseidon_hash_sponge() {
            initialize();
            check_poseidon_hash_sponge::<$field>();
        }

        #[test]
        fn test_poseidon_hash_multi_device() {
            initialize();
            test_utilities::test_set_main_device();
            let nof_devices = icicle_runtime::get_device_count().unwrap();
            if nof_devices > 1 {
                check_poseidon_hash_multi_device::<$field>();
            } else {
                println!("Skipping test_poseidon_hash_multi_device due to single device in the machine");
            }
        }

        #[test]
        fn test_poseidon_tree() {
            initialize();
            test_utilities::test_set_main_device();
            check_poseidon_tree::<$field>();
            test_utilities::test_set_ref_device();
            check_poseidon_tree::<$field>();
        }
    };
}
