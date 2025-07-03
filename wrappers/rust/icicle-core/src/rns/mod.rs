use crate::{field::PrimeField, vec_ops::VecOpsConfig};
use icicle_runtime::{memory::HostOrDeviceSlice, IcicleError};

pub mod tests;

/// Trait for RNS conversions (`Zq <--> ZqRns`)
pub trait RnsConversion<Zq: PrimeField, ZqRns: PrimeField> {
    fn to_rns(
        input: &(impl HostOrDeviceSlice<Zq> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<ZqRns> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn from_rns(
        input: &(impl HostOrDeviceSlice<ZqRns> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<Zq> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;
}

// Note: An in-place RNS conversion could be implemented, but it's unnecessary for now.
// On GPUs, out-of-place conversion is generally faster due to better memory access patterns.
// Additionally, in-place conversion would require move semantics to transfer ownership of the underlying memory,
// meaning it would need separate implementations for both `Vec` and `DeviceVec`, rather than relying on the `HostOrDeviceSlice` trait.

/// Performs `Zq -> ZqRns` conversion.
pub fn to_rns<Zq: PrimeField, ZqRns: PrimeField>(
    input: &(impl HostOrDeviceSlice<Zq> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<ZqRns> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    Zq: RnsConversion<Zq, ZqRns>,
{
    Zq::to_rns(input, output, cfg)
}

/// Performs `ZqRns -> Zq` conversion.
pub fn from_rns<Zq: PrimeField, ZqRns: PrimeField>(
    input: &(impl HostOrDeviceSlice<ZqRns> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<Zq> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    Zq: RnsConversion<Zq, ZqRns>,
{
    Zq::from_rns(input, output, cfg)
}

/// Implements RNS conversion for a given ring via C bindings.
/// Note: This implements `RnsConversion` for `ZqConfigType`, not `ZqRnsConfigType`.
#[macro_export]
macro_rules! impl_rns_conversions {
    (
        $ring_prefix: literal,
        $ZqType: ident,
        $ZqRnsType: ident
    ) => {
        extern "C" {
            #[link_name = concat!($ring_prefix, "_convert_to_rns")]
            fn convert_to_rns(
                input: *const $ZqType,
                size: u64,
                cfg: *const VecOpsConfig,
                output: *mut $ZqRnsType,
            ) -> eIcicleError;

            #[link_name = concat!($ring_prefix, "_convert_from_rns")]
            fn convert_from_rns(
                input: *const $ZqRnsType,
                size: u64,
                cfg: *const VecOpsConfig,
                output: *mut $ZqType,
            ) -> eIcicleError;
        }

        use icicle_core::rns::RnsConversion;
        use icicle_runtime::errors::IcicleError;

        impl RnsConversion<$ZqType, $ZqRnsType> for $ZqType {
            fn to_rns(
                input: &(impl HostOrDeviceSlice<$ZqType> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$ZqRnsType> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                // Ensure sizes match and batch-size divides the size
                if input.len() != output.len() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        format!(
                            "Mismatched slice sizes: input = {}, output = {}",
                            input.len(),
                            output.len()
                        ),
                    ));
                }
                if input.len() % (cfg.batch_size as usize) != 0 {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        format!(
                            "Batch-size={} does not divide total-size={}",
                            cfg.batch_size,
                            input.len()
                        ),
                    ));
                }

                // Ensure input/output are on the active device
                if input.is_on_device() && !input.is_on_active_device() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        "Input is allocated on an inactive device.",
                    ));
                }
                if output.is_on_device() && !output.is_on_active_device() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        "Output is allocated on an inactive device.",
                    ));
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_a_on_device = input.is_on_device();
                cfg_clone.is_result_on_device = output.is_on_device();

                unsafe {
                    convert_to_rns(
                        input.as_ptr(),
                        (input.len() / (cfg.batch_size as usize)) as u64,
                        cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn from_rns(
                input: &(impl HostOrDeviceSlice<$ZqRnsType> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$ZqType> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                // Ensure sizes match and batch-size divides the size
                if input.len() != output.len() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        format!(
                            "Mismatched slice sizes: input = {}, output = {}",
                            input.len(),
                            output.len()
                        ),
                    ));
                }
                if input.len() % (cfg.batch_size as usize) != 0 {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        format!(
                            "Batch-size={} does not divide total-size={}",
                            cfg.batch_size,
                            input.len()
                        ),
                    ));
                }

                // Ensure input/output are on the active device
                if input.is_on_device() && !input.is_on_active_device() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        "Input is allocated on an inactive device.",
                    ));
                }
                if output.is_on_device() && !output.is_on_active_device() {
                    return Err(IcicleError::new(
                        eIcicleError::InvalidArgument,
                        "Output is allocated on an inactive device.",
                    ));
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_a_on_device = input.is_on_device();
                cfg_clone.is_result_on_device = output.is_on_device();

                unsafe {
                    convert_from_rns(
                        input.as_ptr(),
                        (input.len() / (cfg.batch_size as usize)) as u64,
                        cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Implements tests for RNS conversions for a given ring.
#[macro_export]
macro_rules! impl_rns_conversions_tests {
    (
        $ZqType: ident,
        $ZqRnsType: ident
    ) => {
        use icicle_runtime::{device::Device, runtime, test_utilities};

        /// Initializes devices before running tests.
        pub fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn test_rns_conversions() {
                initialize();
                check_rns_conversion::<$ZqType, $ZqRnsType>();
            }

            #[test]
            fn test_rns_arithmetic_consistency() {
                initialize();
                check_rns_arithmetic_consistency::<$ZqType, $ZqRnsType>();
            }
        }
    };
}
