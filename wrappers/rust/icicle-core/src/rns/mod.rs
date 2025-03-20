use crate::{traits::FieldImpl, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

// Trait for RNS-conversions (Zq <--> ZqRns)
pub trait RnsConversion<Zq: FieldImpl, ZqRns: FieldImpl> {
    fn to_rns(
        input: &(impl HostOrDeviceSlice<Zq> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<ZqRns> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn from_rns(
        input: &(impl HostOrDeviceSlice<ZqRns> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<Zq> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

// Floating functions for RNS-conversions (Zq <--> ZqRns)
pub fn to_rns<Zq: FieldImpl, ZqRns: FieldImpl>(
    input: &(impl HostOrDeviceSlice<Zq> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<ZqRns> + ?Sized),
) -> Result<(), eIcicleError>
where
    <Zq as FieldImpl>::Config: RnsConversion<Zq, ZqRns>,
{
    if input.len() != output.len() {
        eprintln!(
            "input (size={}) and output (size={}) slices must have the same length",
            input.len(),
            output.len()
        );
        return Err(eIcicleError::InvalidArgument);
    }

    // check device slices are on active device
    if input.is_on_device() && !input.is_on_active_device() {
        eprintln!("input not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }
    if output.is_on_device() && !output.is_on_active_device() {
        eprintln!("output not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }

    let mut cfg_clone = cfg.clone();
    cfg_clone.is_a_on_device = input.is_on_device();
    cfg_clone.is_result_on_device = output.is_on_device();

    <<Zq as FieldImpl>::Config as RnsConversion<Zq, ZqRns>>::to_rns(input, cfg, output)
}

pub fn from_rns<Zq: FieldImpl, ZqRns: FieldImpl>(
    input: &(impl HostOrDeviceSlice<ZqRns> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<Zq> + ?Sized),
) -> Result<(), eIcicleError>
where
    <Zq as FieldImpl>::Config: RnsConversion<Zq, ZqRns>,
{
    if input.len() != output.len() {
        eprintln!(
            "input (size={}) and output (size={}) slices must have the same length",
            input.len(),
            output.len()
        );
        return Err(eIcicleError::InvalidArgument);
    }

    // check device slices are on active device
    if input.is_on_device() && !input.is_on_active_device() {
        eprintln!("input not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }
    if output.is_on_device() && !output.is_on_active_device() {
        eprintln!("output not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }

    let mut cfg_clone = cfg.clone();
    cfg_clone.is_a_on_device = input.is_on_device();
    cfg_clone.is_result_on_device = output.is_on_device();

    <<Zq as FieldImpl>::Config as RnsConversion<Zq, ZqRns>>::from_rns(input, cfg, output)
}

// Macro that implements RNS-conversions for a given ring, via C-bindings
// Note: this macro implements the RnsConversion trait for the ZqConfigType only, not need to implement it for ZqRnsConfigType as well
#[macro_export]
macro_rules! impl_rns_conversions {
    (
        $ring_prefix: literal, 
        $ZqType: ident, 
        $ZqRnsType: ident,
        $ZqConfigType: ident
    ) => {
        extern "C" {
            #[link_name = concat!($ring_prefix, "_convert_to_rns")]
            fn convert_to_rns(input: *const $ZqType, size: u64, cfg: *const VecOpsConfig, output: *mut $ZqRnsType) -> eIcicleError;

            #[link_name = concat!($ring_prefix, "_convert_from_rns")]
            fn convert_from_rns(input: *const $ZqRnsType, size: u64, cfg: *const VecOpsConfig, output: *mut $ZqType) -> eIcicleError;
        }

        use icicle_core::rns::RnsConversion;

        impl RnsConversion<$ZqType, $ZqRnsType> for $ZqConfigType {
            fn to_rns(
                input: &(impl HostOrDeviceSlice<$ZqType> + ?Sized),
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$ZqRnsType> + ?Sized),
            ) -> Result<(), eIcicleError> {
                unsafe {
                    convert_to_rns(
                        input.as_ptr(),
                        input.len() as u64,
                        cfg,
                        output.as_mut_ptr(),
                    ).wrap()
                }
            }

            fn from_rns(
                input: &(impl HostOrDeviceSlice<$ZqRnsType> + ?Sized),
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$ZqType> + ?Sized),
            ) -> Result<(), eIcicleError> {
                unsafe {
                    convert_from_rns(
                        input.as_ptr(),
                        input.len() as u64,
                        cfg,
                        output.as_mut_ptr(),
                    ).wrap()
                }
            }
        }
    };
}

// Macro that instantiates tests for RNS-conversions for a given ring
#[macro_export]
macro_rules! impl_rns_conversions_tests {
    (
        $ZqType: ident,
        $ZqRnsType: ident
    ) => {
        use icicle_runtime::test_utilities;
        use icicle_runtime::{device::Device, runtime};
        

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
        }
    };
}
