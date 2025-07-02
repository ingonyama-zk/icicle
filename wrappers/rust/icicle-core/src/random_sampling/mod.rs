use crate::traits::FieldImpl;
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{memory::HostOrDeviceSlice, IcicleError};

pub mod tests;

/// Trait for random sampling operations on group elements.
pub trait RandomSampling<T: FieldImpl> {
    fn random_sampling(
        size: usize,
        fast_mode: bool,
        seed: &[u8],
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), IcicleError>;
}

pub fn random_sampling<T>(
    size: usize,
    fast_mode: bool,
    seed: &[u8],
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError>
where
    T: FieldImpl,
    T::Config: RandomSampling<T>,
{
    T::Config::random_sampling(size, fast_mode, seed, cfg, output)
}

/// Implements RandomSampling for a scalar ring type using FFI.
#[macro_export]
macro_rules! impl_random_sampling {
    ($prefix:literal, $scalar_type:ty, $implement_for:ty) => {
        use icicle_core::random_sampling::RandomSampling;
        use icicle_core::vec_ops::VecOpsConfig;
        use icicle_runtime::errors::{eIcicleError, IcicleError};
        use icicle_runtime::memory::HostOrDeviceSlice;

        extern "C" {
            #[link_name = concat!($prefix, "_random_sampling")]
            fn random_sampling_ffi(
                size: usize,
                fast_mode: bool,
                seed: *const u8,
                seed_len: usize,
                cfg: *const VecOpsConfig,
                output: *mut $scalar_type,
            ) -> eIcicleError;
        }

        impl RandomSampling<$scalar_type> for $implement_for {
            fn random_sampling(
                size: usize,
                fast_mode: bool,
                seed: &[u8],
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$scalar_type> + ?Sized),
            ) -> Result<(), IcicleError> {
                if output.is_on_device() && !output.is_on_active_device() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        "Output is on an inactive device",
                    ));
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_result_on_device = output.is_on_device();

                unsafe {
                    random_sampling_ffi(
                        size,
                        fast_mode,
                        seed.as_ptr(),
                        seed.len(),
                        &cfg_clone,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Implements unit tests for RandomSampling on scalar ring types.
#[macro_export]
macro_rules! impl_random_sampling_tests {
    ($scalar_type: ident) => {
        mod test_scalar {
            use super::*;
            use icicle_core::random_sampling::tests::*;
            use icicle_runtime::test_utilities;

            /// Initializes devices before running tests.
            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
            }

            #[test]
            fn test_random_sampling() {
                initialize();
                check_random_sampling::<$scalar_type>();
            }
        }
    };
}
