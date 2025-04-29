use crate::traits::FieldImpl;
use icicle_runtime::{
    errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
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
    pub is_rotations_on_device: bool,
    pub is_challenges_on_device: bool,
    pub is_calculations_on_device: bool,
    pub is_previous_value_on_device: bool,
    pub is_horners_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
}

impl GateOpsConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            is_constants_on_device: false,
            is_fixed_on_device: false,
            is_advice_on_device: false,
            is_instance_on_device: false,
            is_rotations_on_device: false,
            is_challenges_on_device: false,
            is_calculations_on_device: false,
            is_previous_value_on_device: false,
            is_horners_on_device: false,
            is_result_on_device: false,
            is_async: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LookupConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_table_values_on_device: bool,
    pub is_permuted_input_coset_on_device: bool,
    pub is_permuted_table_coset_on_device: bool,
    pub is_product_coset_on_device: bool,
    pub is_l_on_device: bool,
    pub is_previous_values_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
}

impl LookupConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            is_table_values_on_device: false,
            is_permuted_input_coset_on_device: false,
            is_permuted_table_coset_on_device: false,
            is_product_coset_on_device: false,
            is_l_on_device: false,
            is_previous_values_on_device: false,
            is_result_on_device: false,
            is_async: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LookupData<T> {
    pub table_values: *const T,
    pub num_table_values: u32,
    pub permuted_input_coset: *const T,
    pub num_permuted_input_coset: u32,
    pub permuted_table_coset: *const T,
    pub num_permuted_table_coset: u32,
    pub product_coset: *const T,
    pub num_product_coset: u32,
    pub l0: *const T,
    pub num_l0: u32,
    pub l_last: *const T,
    pub num_l_last: u32,
    pub l_active_row: *const T,
    pub num_l_active_row: u32,
    pub y: *const T,
    pub beta: *const T,
    pub gamma: *const T,
    pub previous_value: *const T,
    pub num_elements: u32,
    pub rot_scale: u32,
    pub i_size: u32,
}

impl<T> LookupData<T> {
    pub fn new(
        table_values: *const T,
        num_table_values: u32,
        permuted_input_coset: *const T,
        num_permuted_input_coset: u32,
        permuted_table_coset: *const T,
        num_permuted_table_coset: u32,
        product_coset: *const T,
        num_product_coset: u32,
        l0: *const T,
        num_l0: u32,
        l_last: *const T,
        num_l_last: u32,
        l_active_row: *const T,
        num_l_active_row: u32,
        y: *const T,
        beta: *const T,
        gamma: *const T,
        previous_value: *const T,
        num_elements: u32,
        rot_scale: u32,
        i_size: u32,
    ) -> Self {
        Self {
            table_values,
            num_table_values,
            permuted_input_coset,
            num_permuted_input_coset,
            permuted_table_coset,
            num_permuted_table_coset,
            product_coset,
            num_product_coset,
            l0,
            num_l0,
            l_last,
            num_l_last,
            l_active_row,
            num_l_active_row,
            y,
            beta,
            gamma,
            previous_value,
            num_elements,
            rot_scale,
            i_size,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct HornerData {
    pub value_types: *const u32,
    pub value_indices: *const u32,
    pub offsets: *const u32,
    pub sizes: *const u32,
    pub num_horner: u32,
}

impl HornerData {
    pub fn new(
        value_types: *const u32,
        value_indices: *const u32,
        offsets: *const u32,
        sizes: *const u32,
        num_horner: u32,
    ) -> Self {
        Self {
            value_types,
            value_indices,
            offsets,
            sizes,
            num_horner
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CalculationData<T> {
    pub calc_types: *const u32,
    pub targets: *const u32,
    pub value_types: *const u32,
    pub value_indices: *const u32,
    pub constants: *const T,
    pub num_constants: u32,
    pub rotations: *const i32,
    pub num_rotations: u32,
    pub previous_value: *const T,
    pub num_calculations: u32,
    pub num_intermediates: u32,
    pub num_elements: u32,
    pub rot_scale: u32,
    pub i_size: u32,
}

impl<T> CalculationData<T> {
    pub fn new(
        calc_types: *const u32,
        targets: *const u32,
        value_types: *const u32,
        value_indices: *const u32,
        constants: *const T,
        num_constants: u32,
        rotations: *const i32,
        num_rotations: u32,
        previous_value: *const T,
        num_calculations: u32,
        num_intermediates: u32,
        num_elements: u32,
        rot_scale: u32,
        i_size: u32,
    ) -> Self {
        Self {
            calc_types,
            targets,
            value_types,
            value_indices,
            constants,
            num_constants,
            rotations,
            num_rotations,
            previous_value,
            num_calculations,
            num_intermediates,
            num_elements,
            rot_scale,
            i_size,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GateData<T> {
    pub fixed: *const T,
    pub num_fixed_columns: u32,
    pub num_fixed_rows: u32,
    pub advice: *const T,
    pub num_advice_columns: u32,
    pub num_advice_rows: u32,
    pub instance: *const T,
    pub num_instance_columns: u32,
    pub num_instance_rows: u32,
    pub challenges: *const T,
    pub num_challenges: u32,
    pub beta: *const T,
    pub gamma: *const T,
    pub theta: *const T,
    pub y: *const T,
}

impl<T> GateData<T> {
    pub fn new(
        fixed: *const T,
        num_fixed_columns: u32,
        num_fixed_rows: u32,
        advice: *const T,
        num_advice_columns: u32,
        num_advice_rows: u32,
        instance: *const T,
        num_instance_columns: u32,
        num_instance_rows: u32,
        challenges: *const T,
        num_challenges: u32,
        beta: *const T,
        gamma: *const T,
        theta: *const T,
        y: *const T,
    ) -> Self {
        Self {
            fixed,
            num_fixed_columns,
            num_fixed_rows,
            advice,
            num_advice_columns,
            num_advice_rows,
            instance,
            num_instance_columns,
            num_instance_rows,
            challenges,
            num_challenges,
            beta,
            gamma,
            theta,
            y,
        }
    }
}

#[doc(hidden)]
pub trait GateOps<F> {
    fn gate_evaluation(
        gate_data: &GateData<F>,
        calc_data: &CalculationData<F>,
        horner_data: &HornerData,
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &GateOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn lookups_constraint(
        lookup_data: &LookupData<F>,
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &LookupConfig,
    ) -> Result<(), eIcicleError>;
}

pub fn gate_evaluation<F>(
    gate_data: &GateData<F>,
    calc_data: &CalculationData<F>,
    horner_data: &HornerData,
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &GateOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: GateOps<F>,
{
    <<F as FieldImpl>::Config as GateOps<F>>::gate_evaluation(
        gate_data,
        calc_data,
        horner_data,
        result, 
        &cfg,
    )
}

pub fn lookups_constraint<F>(
    lookup_data: &LookupData<F>,
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &LookupConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: GateOps<F>,
{
    <<F as FieldImpl>::Config as GateOps<F>>::lookups_constraint(
        lookup_data,
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
            use icicle_core::gate_ops::{GateData, CalculationData, HornerData, LookupData, GateOpsConfig, LookupConfig};
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($field_prefix, "_gate_evaluation")]
                pub(crate) fn gate_evaluation_ffi(
                    gate_data: *const GateData<$field>,
                    calc_data: *const CalculationData<$field>,
                    horner_data: *const HornerData,
                    cfg: *const GateOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_lookups_constraint")]
                pub(crate) fn lookups_constraint_ffi(
                    lookup_data: *const LookupData<$field>,
                    cfg: *const LookupConfig,
                    result: *mut $field,
                ) -> eIcicleError;
            }
        }

        impl GateOps<$field> for $field_config {
            fn gate_evaluation(
                gate_data: &GateData<$field>,
                calc_data: &CalculationData<$field>,
                horner_data: &HornerData,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &GateOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::gate_evaluation_ffi(
                        gate_data as *const GateData<$field>,
                        calc_data as *const CalculationData<$field>,
                        horner_data as *const HornerData,
                        cfg as *const GateOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn lookups_constraint(
                lookup_data: &LookupData<$field>,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &LookupConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::lookups_constraint_ffi(
                        lookup_data as *const LookupData<$field>,
                        cfg as *const LookupConfig,
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

            #[test]
            pub fn test_lookup_constraints_scalars() {
                initialize();
                check_lookup_constraints_scalars::<$field>()
            }
        }
    };
}
