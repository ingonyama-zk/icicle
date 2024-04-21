use icicle_cuda_runtime::device::check_device;
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
            is_async: false,
        }
    }
}

#[doc(hidden)]
pub trait VecOps<F> {
    fn add(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn sub(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn transpose(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_size: u32,
        column_size: u32,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        ctx: &DeviceContext,
        on_device: bool,
        is_async: bool,
    ) -> IcicleResult<()>;
}

fn check_vec_ops_args<'a, F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig<'a>,
) -> VecOpsConfig<'a> {
    if a.len() != b.len() || a.len() != result.len() {
        panic!(
            "left, right and output lengths {}; {}; {} do not match",
            a.len(),
            b.len(),
            result.len()
        );
    }
    let ctx_device_id = cfg
        .ctx
        .device_id;
    if let Some(device_id) = a.device_id() {
        assert_eq!(device_id, ctx_device_id, "Device ids in a and context are different");
    }
    if let Some(device_id) = b.device_id() {
        assert_eq!(device_id, ctx_device_id, "Device ids in b and context are different");
    }
    if let Some(device_id) = result.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in result and context are different"
        );
    }
    check_device(ctx_device_id);

    let mut res_cfg = cfg.clone();
    res_cfg.is_a_on_device = a.is_on_device();
    res_cfg.is_b_on_device = b.is_on_device();
    res_cfg.is_result_on_device = result.is_on_device();
    res_cfg
}

pub fn add_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::add(a, b, result, &cfg)
}

pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::sub(a, b, result, &cfg)
}

pub fn mul_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::mul(a, b, result, &cfg)
}

pub fn transpose_matrix<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_size: u32,
    column_size: u32,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ctx: &DeviceContext,
    on_device: bool,
    is_async: bool,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    <<F as FieldImpl>::Config as VecOps<F>>::transpose(input, row_size, column_size, output, ctx, on_device, is_async)
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
                #[link_name = concat!($field_prefix, "_add_cuda")]
                pub(crate) fn add_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_sub_cuda")]
                pub(crate) fn sub_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_mul_cuda")]
                pub(crate) fn mul_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_transpose_matrix_cuda")]
                pub(crate) fn transpose_cuda(
                    input: *const $field,
                    row_size: u32,
                    column_size: u32,
                    output: *mut $field,
                    ctx: *const DeviceContext,
                    on_device: bool,
                    is_async: bool,
                ) -> CudaError;
            }
        }

        impl VecOps<$field> for $field_config {
            fn add(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::add_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn sub(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::sub_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn mul(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::mul_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn transpose(
                input: &(impl HostOrDeviceSlice<$field> + ?Sized),
                row_size: u32,
                column_size: u32,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                ctx: &DeviceContext,
                on_device: bool,
                is_async: bool,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::transpose_cuda(
                        input.as_ptr(),
                        row_size,
                        column_size,
                        output.as_mut_ptr(),
                        ctx as *const _ as *const DeviceContext,
                        on_device,
                        is_async,
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
