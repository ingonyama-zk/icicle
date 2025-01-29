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
    pub is_challenges_on_device: bool,
    pub is_rotations_on_device: bool,
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
            is_challenges_on_device: false,
            is_rotations_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct HornerData {
    pub value_types: *const u32,
    pub i_value_indices: *const u32,
    pub j_value_indices: *const u32,
    pub offsets: *const u32,
    pub sizes: *const u32,
}

impl HornerData {
    pub fn new(
        value_types: *const u32,
        i_value_indices: *const u32,
        j_value_indices: *const u32,
        offsets: *const u32,
        sizes: *const u32,
    ) -> Self {
        Self {
            value_types,
            i_value_indices,
            j_value_indices,
            offsets,
            sizes,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CalculationData {
    pub value_types: *const u32,
    pub i_value_types: *const u32,
    pub j_value_types: *const u32,
    pub i_value_indices: *const u32,
    pub j_value_indices: *const u32,
    pub num_calculations: u32,
    pub num_intermediates: u32,
}

impl CalculationData {
    pub fn new(
        value_types: *const u32,
        i_value_types: *const u32,
        j_value_types: *const u32,
        i_value_indices: *const u32,
        j_value_indices: *const u32,
        num_calculations: u32,
        num_intermediates: u32,
    ) -> Self {
        Self {
            value_types,
            i_value_types,
            j_value_types,
            i_value_indices,
            j_value_indices,
            num_calculations,
            num_intermediates,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GateData<T> {
    pub constants: *const T,
    pub num_constants: u32,
    pub fixed: *const *const T,
    pub num_fixed_columns: u32,
    pub num_fixed_rows: u32,
    pub advice: *const *const T,
    pub num_advice_columns: u32,
    pub num_advice_rows: u32,
    pub instance: *const *const T,
    pub num_instance_columns: u32,
    pub num_instance_rows: u32,
    pub rotations: *const u32,
    pub num_rotations: u32,
    pub challenges: *const T,
    pub num_challenges: u32,
    pub beta: *const T,
    pub gamma: *const T,
    pub theta: *const T,
    pub y: *const T,
    pub previous_value: *const T,
    pub num_elements: u32,
    pub rot_scale: u32,
    pub i_size: u32,
}

impl<T> GateData<T> {
    pub fn new(
        constants: *const T,
        num_constants: u32,
        fixed: *const *const T,
        num_fixed_columns: u32,
        num_fixed_rows: u32,
        advice: *const *const T,
        num_advice_columns: u32,
        num_advice_rows: u32,
        instance: *const *const T,
        num_instance_columns: u32,
        num_instance_rows: u32,
        rotations: *const u32,
        num_rotations: u32,
        challenges: *const T,
        num_challenges: u32,
        beta: *const T,
        gamma: *const T,
        theta: *const T,
        y: *const T,
        previous_value: *const T,
        num_elements: u32,
        rot_scale: u32,
        i_size: u32,
    ) -> Self {
        Self {
            constants,
            num_constants,
            fixed,
            num_fixed_columns,
            num_fixed_rows,
            advice,
            num_advice_columns,
            num_advice_rows,
            instance,
            num_instance_columns,
            num_instance_rows,
            rotations,
            num_rotations,
            challenges,
            num_challenges,
            beta,
            gamma,
            theta,
            y,
            previous_value,
            num_elements,
            rot_scale,
            i_size,
        }
    }
}

#[doc(hidden)]
pub trait GateOps<F> {
    fn gate_evaluation(
        gate_data: &GateData<F>,
        calc_data: &CalculationData,
        horner_data: &HornerData,
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &GateOpsConfig,
    ) -> Result<(), eIcicleError>;
}

fn check_gate_ops_args<F>(
    gate_data: &GateData<F>,
    calc_data: &CalculationData,
    horner_data: &HornerData,
    cfg: &GateOpsConfig,
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> GateOpsConfig {
    setup_config(
        gate_data,
        calc_data,
        horner_data,
        result,
        cfg)
}


fn setup_config<F>(
    gate_data: &GateData<F>,
    calc_data: &CalculationData,
    horner_data: &HornerData,
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &GateOpsConfig,
) -> GateOpsConfig {
    // if gate_data.constants.is_on_device() && !gate_data.constants.is_on_active_device() {
    //     panic!("input constants is allocated on an inactive device");
    // }
    // if gate_data.fixed.is_on_device() && !gate_data.fixed.is_on_active_device() {
    //     panic!("input fixed is allocated on an inactive device");
    // }
    // if gate_data.advice.is_on_device() && !gate_data.advice.is_on_active_device() {
    //     panic!("input advice is allocated on an inactive device");
    // }
    // if gate_data.instance.is_on_device() && !gate_data.instance.is_on_active_device() {
    //     panic!("input instance is allocated on an inactive device");
    // }
    // if result.is_on_device() && !result.is_on_active_device() {
    //     panic!("output is allocated on an inactive device");
    // }

    let mut res_cfg = cfg.clone();
    // res_cfg.is_constants_on_device = gate_data.constants.is_on_device();
    // res_cfg.is_fixed_on_device = gate_data.fixed.is_on_device();
    // res_cfg.is_advice_on_device = gate_data.advice.is_on_device();
    // res_cfg.is_instance_on_device = gate_data.instance.is_on_device();
    // res_cfg.is_result_on_device = result.is_on_device();
    res_cfg
}

pub fn gate_evaluation<F>(
    gate_data: &GateData<F>,
    calc_data: &CalculationData,
    horner_data: &HornerData,
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &GateOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: GateOps<F>,
{
    let cfg = check_gate_ops_args(
        gate_data,
        calc_data,
        horner_data,
        cfg,
        result
    );
    <<F as FieldImpl>::Config as GateOps<F>>::gate_evaluation(
        gate_data,
        calc_data,
        horner_data,
        result, 
        &cfg,
    )
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
            use icicle_core::gate_ops::{GateData, CalculationData, HornerData, GateOpsConfig};
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($field_prefix, "_gate_evaluation")]
                pub(crate) fn gate_evaluation_ffi(
                    gate_data: *const GateData<$field>,
                    calc_data: *const CalculationData,
                    horner_data: *const HornerData,
                    cfg: *const GateOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;
            }
        }

        impl GateOps<$field> for $field_config {
            fn gate_evaluation(
                gate_data: &GateData<$field>,
                calc_data: &CalculationData,
                horner_data: &HornerData,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &GateOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::gate_evaluation_ffi(
                        gate_data as *const GateData<$field>,
                        calc_data as *const CalculationData,
                        horner_data as *const HornerData,
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
