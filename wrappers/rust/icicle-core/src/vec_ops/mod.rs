use icicle_cuda_runtime::{
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::HostOrDeviceSlice,
};

use crate::{error::IcicleResult, traits::FieldImpl};

pub mod tests;

/// Struct that encodes VecOps parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VecOpsConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    is_a_on_device: bool,
    is_b_on_device: bool,
    is_result_on_device: bool,
    is_result_montgomery_form: bool,
    /// Whether to run the vector operations asynchronously. If set to `true`, the functions will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the functions will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for VecOpsConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> VecOpsConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        VecOpsConfig {
            ctx: DeviceContext::default_for_device(device_id),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_result_montgomery_form: false,
            is_async: false,
        }
    }
}

#[doc(hidden)]
pub trait VecOps<F> {
    fn add(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn sub(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn mul(
        a: &HostOrDeviceSlice<F>,
        b: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;
}

fn check_vec_ops_args<F>(a: &HostOrDeviceSlice<F>, b: &HostOrDeviceSlice<F>, result: &mut HostOrDeviceSlice<F>) {
    if a.len() != b.len() || a.len() != result.len() {
        panic!(
            "left, right and output lengths {}; {}; {} do not match",
            a.len(),
            b.len(),
            result.len()
        );
    }
}

pub fn add_scalars<F>(
    a: &HostOrDeviceSlice<F>,
    b: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    check_vec_ops_args(a, b, result);
    <<F as FieldImpl>::Config as VecOps<F>>::add(a, b, result, cfg)
}

pub fn sub_scalars<F>(
    a: &HostOrDeviceSlice<F>,
    b: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    check_vec_ops_args(a, b, result);
    <<F as FieldImpl>::Config as VecOps<F>>::sub(a, b, result, cfg)
}

pub fn mul_scalars<F>(
    a: &HostOrDeviceSlice<F>,
    b: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    check_vec_ops_args(a, b, result);
    <<F as FieldImpl>::Config as VecOps<F>>::mul(a, b, result, cfg)
}

#[macro_export]
macro_rules! impl_vec_ops_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, CudaError, DeviceContext, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;

            extern "C" {
                #[link_name = concat!($field_prefix, "AddCuda")]
                pub(crate) fn add_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "SubCuda")]
                pub(crate) fn sub_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "MulCuda")]
                pub(crate) fn mul_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;
            }
        }

        impl VecOps<$field> for $field_config {
            fn add(
                a: &HostOrDeviceSlice<$field>,
                b: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::add_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const _ as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn sub(
                a: &HostOrDeviceSlice<$field>,
                b: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::sub_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const _ as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn mul(
                a: &HostOrDeviceSlice<$field>,
                b: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::mul_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const _ as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_add_tests {
    (
      $field:ident
    ) => {
        #[test]
        pub fn test_vec_add_scalars() {
            check_vec_ops_scalars::<$field>()
        }
    };
}
