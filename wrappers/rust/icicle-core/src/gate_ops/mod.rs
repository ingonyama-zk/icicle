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
    pub value_types: *const i32,
    pub i_value_indices: *const i32,
    pub j_value_indices: *const i32,
    pub offsets: *const i32,
    pub sizes: *const i32,
}

impl HornerData {
    pub fn new(
        value_types: *const i32,
        i_value_indices: *const i32,
        j_value_indices: *const i32,
        offsets: *const i32,
        sizes: *const i32,
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
    pub value_types: *const i32,
    pub i_value_types: *const i32,
    pub j_value_types: *const i32,
    pub i_value_indices: *const i32,
    pub j_value_indices: *const i32,
    pub num_calculations: usize,
    pub num_intermediates: usize,
}

impl CalculationData {
    pub fn new(
        value_types: *const i32,
        i_value_types: *const i32,
        j_value_types: *const i32,
        i_value_indices: *const i32,
        j_value_indices: *const i32,
        num_calculations: usize,
        num_intermediates: usize,
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
    pub num_constants: usize,
    pub fixed: *const T,
    pub num_fixed_columns: usize,
    pub advice: *const T,
    pub num_advice_columns: usize,
    pub instance: *const T,
    pub num_instance_columns: usize,
    pub rotations: *const i32,
    pub num_rotations: usize,
    pub challenges: *const T,
    pub num_challenges: usize,
    pub beta: *const T,
    pub gamma: *const T,
    pub theta: *const T,
    pub y: *const T,
    pub previous_value: *const T,
    pub num_elements: i32,
    pub rot_scale: i32,
    pub i_size: i32,
}

impl<T> GateData<T> {
    pub fn new(
        constants: *const T,
        num_constants: usize,
        fixed: *const T,
        num_fixed_columns: usize,
        advice: *const T,
        num_advice_columns: usize,
        instance: *const T,
        num_instance_columns: usize,
        rotations: *const i32,
        num_rotations: usize,
        challenges: *const T,
        num_challenges: usize,
        beta: *const T,
        gamma: *const T,
        theta: *const T,
        y: *const T,
        previous_value: *const T,
        num_elements: i32,
        rot_scale: i32,
        i_size: i32,
    ) -> Self {
        Self {
            constants,
            num_constants,
            fixed,
            num_fixed_columns,
            advice,
            num_advice_columns,
            instance,
            num_instance_columns,
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
