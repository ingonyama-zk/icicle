use crate::traits::FieldImpl;
use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};

pub mod tests;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GateOpsConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_constants_on_device: bool,
    pub is_fixed_on_device: bool,
    pub is_advice_on_device: bool,
    pub is_instance_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl GateOpsConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            is_constants_on_device: false,
            is_fixed_on_device: false,
            is_advice_on_device: false,
            is_instance_on_device: false,
            is_result_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[doc(hidden)]
pub trait GateOps<F> {
    fn gate_evaluation(
        constants: &(impl HostOrDeviceSlice<F> + ?Sized),
        fixed: &(impl HostOrDeviceSlice<F> + ?Sized),
        advice: &(impl HostOrDeviceSlice<F> + ?Sized),
        instance: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &GateOpsConfig,
    ) -> Result<(), eIcicleError>;
}

fn check_gate_ops_args<F>(
    constants: &(impl HostOrDeviceSlice<F> + ?Sized),
    fixed: &(impl HostOrDeviceSlice<F> + ?Sized),
    advice: &(impl HostOrDeviceSlice<F> + ?Sized),
    instance: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &GateOpsConfig,
) -> GateOpsConfig {
    setup_config(constants, fixed, advice, instance, result, cfg)
}

/// Modify GateOpsConfig according to the given vectors
fn setup_config<F>(
    constants: &(impl HostOrDeviceSlice<F> + ?Sized),
    fixed: &(impl HostOrDeviceSlice<F> + ?Sized),
    advice: &(impl HostOrDeviceSlice<F> + ?Sized),
    instance: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &GateOpsConfig,
) -> GateOpsConfig {
    // check device slices are on active device
    if constants.is_on_device() && !constants.is_on_active_device() {
        panic!("input constants is allocated on an inactive device");
    }
    if fixed.is_on_device() && !fixed.is_on_active_device() {
        panic!("input fixed is allocated on an inactive device");
    }
    if advice.is_on_device() && !advice.is_on_active_device() {
        panic!("input advice is allocated on an inactive device");
    }
    if instance.is_on_device() && !instance.is_on_active_device() {
        panic!("input instance is allocated on an inactive device");
    }
    if result.is_on_device() && !result.is_on_active_device() {
        panic!("output is allocated on an inactive device");
    }

    let mut res_cfg = cfg.clone();
    res_cfg.is_constants_on_device = constants.is_on_device();
    res_cfg.is_fixed_on_device = fixed.is_on_device();
    res_cfg.is_advice_on_device = advice.is_on_device();
    res_cfg.is_instance_on_device = instance.is_on_device();
    res_cfg.is_result_on_device = result.is_on_device();
    res_cfg
}

pub fn gate_evaluation<F>(
    constants: &(impl HostOrDeviceSlice<F> + ?Sized),
    fixed: &(impl HostOrDeviceSlice<F> + ?Sized),
    advice: &(impl HostOrDeviceSlice<F> + ?Sized),
    instance: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &GateOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: GateOps<F>,
{
    let cfg = check_gate_ops_args(constants, fixed, advice, instance, result, cfg);
    <<F as FieldImpl>::Config as GateOps<F>>::gate_evaluation(constants, fixed, advice, instance, result, &cfg)
}

#[macro_export]
macro_rules! impl_gate_ops_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        mod $field_prefix_ident {

            use crate::gate_ops::{$field, HostOrDeviceSlice};
            use icicle_core::gate_ops::GateOpsConfig;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($field_prefix, "_gate_evaluation")]
                pub(crate) fn gate_evaluation_ffi(
                    constants: *const $field,
                    fixed: *const $field,
                    advice: *const $field,
                    instance: *const $field,
                    cfg: *const GateOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;
            }
        }

        impl GateOps<$field> for $field_config {
            fn gate_evaluation(
                constants: &(impl HostOrDeviceSlice<$field> + ?Sized),
                fixed: &(impl HostOrDeviceSlice<$field> + ?Sized),
                advice: &(impl HostOrDeviceSlice<$field> + ?Sized),
                instance: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &GateOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::gate_evaluation_ffi(
                        constants.as_ptr(),
                        fixed.as_ptr(),
                        advice.as_ptr(),
                        instance.as_ptr(),
                        cfg as *const GateOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_gate_ops_tests {
    (
      $field:ident
    ) => {
        pub(crate) mod test_gateops {
            use super::*;
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_gate_ops_scalars() {
                initialize();
                check_gate_ops_scalars::<$field>()
            }
        }
    };
}