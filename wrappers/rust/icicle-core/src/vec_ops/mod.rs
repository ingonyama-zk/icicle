use crate::traits::FieldImpl;
use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};

pub mod tests;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct VecOpsConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_a_on_device: bool,
    pub is_b_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub batch_size: i32,
    pub columns_batch: bool,
    pub ext: ConfigExtension,
}

impl VecOpsConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_async: false,
            batch_size: 1,
            columns_batch: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[doc(hidden)]
pub trait VecOps {
    fn add(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn accumulate(
        a: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn sub(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn mul(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn div(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn sum(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn product(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn scalar_add(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn scalar_sub(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn scalar_mul(
        a: &(impl HostOrDeviceSlice<Self> + ?Sized),
        b: &(impl HostOrDeviceSlice<Self> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn transpose(
        input: &(impl HostOrDeviceSlice<Self> + ?Sized),
        nof_rows: u32,
        nof_cols: u32,
        output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn bit_reverse(
        input: &(impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn bit_reverse_inplace(
        input: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError> where Self: Sized;

    fn slice(
        input: &(impl HostOrDeviceSlice<Self> + ?Sized),
        offset: u64,
        stride: u64,
        size_in: u64,
        size_out: u64,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
    ) -> Result<(), eIcicleError> where Self: Sized;
}

fn check_vec_ops_args<'a, F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> VecOpsConfig {
    if a.len() != b.len() || a.len() != result.len() {
        panic!(
            "left, right and output lengths {}; {}; {} do not match",
            a.len(),
            b.len(),
            result.len()
        );
    }

    // check device slices are on active device
    if a.is_on_device() && !a.is_on_active_device() {
        panic!("input a is allocated on an inactive device");
    }
    if b.is_on_device() && !b.is_on_active_device() {
        panic!("input b is allocated on an inactive device");
    }
    if result.is_on_device() && !result.is_on_active_device() {
        panic!("output is allocated on an inactive device");
    }

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
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <F as VecOps>::add(a, b, result, &cfg)
}

pub fn accumulate_scalars<F>(
    a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, b, a, cfg);
    <F as VecOps>::accumulate(a, b, &cfg)
}

pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <F as VecOps>::sub(a, b, result, &cfg)
}

pub fn mul_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <F as VecOps>::mul(a, b, result, &cfg)
}

pub fn div_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <F as VecOps>::div(a, b, result, &cfg)
}

pub fn sum_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, a, result, cfg); //TODO: emirsoyturk
    <F as VecOps>::sum(a, result, &cfg)
}

pub fn product_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(a, a, result, cfg); //TODO: emirsoyturk
    <F as VecOps>::product(a, result, &cfg)
}

pub fn scalar_add<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(b, b, result, cfg); //TODO: emirsoyturk
    <F as VecOps>::scalar_add(a, b, result, &cfg)
}

pub fn scalar_sub<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(b, b, result, cfg); //TODO: emirsoyturk
    <F as VecOps>::scalar_sub(a, b, result, &cfg)
}

pub fn scalar_mul<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(b, b, result, cfg); //TODO: emirsoyturk
    <F as VecOps>::scalar_mul(a, b, result, &cfg)
}

pub fn transpose_matrix<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    <F as VecOps>::transpose(input, nof_rows, nof_cols, output, &cfg)
}

pub fn bit_reverse<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(input, input /*dummy*/, output, cfg);
    <F as VecOps>::bit_reverse(input, &cfg, output)
}

pub fn bit_reverse_inplace<F>(
    input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    let cfg = check_vec_ops_args(input, input /*dummy*/, input, cfg);
    <F as VecOps>::bit_reverse_inplace(input, &cfg)
}

pub fn slice<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    offset: u64,
    stride: u64,
    size_in: u64,
    size_out: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), eIcicleError>
where
    F: FieldImpl + VecOps,
{
    <F as VecOps>::slice(input, offset, stride, size_in, size_out, &cfg, output)
}

#[macro_export]
macro_rules! impl_vec_ops_field {
    (
        $field_name:literal,
        $field_prefix_ident:ident,
        $field_type:ident
    ) => {
        mod $field_prefix_ident {

            use crate::vec_ops::{$field_type, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($field_name, "_vector_add")]
                pub(crate) fn vector_add_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_vector_accumulate")]
                pub(crate) fn vector_accumulate_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_vector_sub")]
                pub(crate) fn vector_sub_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_vector_mul")]
                pub(crate) fn vector_mul_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_vector_div")]
                pub(crate) fn vector_div_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_vector_sum")]
                pub(crate) fn vector_sum_ffi(
                    a: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_vector_product")]
                pub(crate) fn vector_product_ffi(
                    a: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_scalar_add_vec")]
                pub(crate) fn scalar_add_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_scalar_sub_vec")]
                pub(crate) fn scalar_sub_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_scalar_mul_vec")]
                pub(crate) fn scalar_mul_ffi(
                    a: *const $field_type,
                    b: *const $field_type,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_matrix_transpose")]
                pub(crate) fn matrix_transpose_ffi(
                    input: *const $field_type,
                    nof_rows: u32,
                    nof_cols: u32,
                    cfg: *const VecOpsConfig,
                    output: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_bit_reverse")]
                pub(crate) fn bit_reverse_ffi(
                    input: *const $field_type,
                    size: u64,
                    config: *const VecOpsConfig,
                    output: *mut $field_type,
                ) -> eIcicleError;

                #[link_name = concat!($field_name, "_slice")]
                pub(crate) fn slice_ffi(
                    input: *const $field_type,
                    offset: u64,
                    stride: u64,
                    size_in: u64,
                    size_out: u64,
                    cfg: *const VecOpsConfig,
                    output: *mut $field_type,
                ) -> eIcicleError;
            }
        }

        impl VecOps for $field_type {
            fn add(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_add_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn accumulate(
                a: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_accumulate_ffi(
                        a.as_mut_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                    )
                    .wrap()
                }
            }

            fn sub(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_sub_ffi(
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
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_mul_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn div(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_div_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn sum(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_sum_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn product(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_sum_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn scalar_add(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_add_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        b.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn scalar_sub(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_sub_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        b.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn scalar_mul(
                a: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_mul_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        b.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn transpose(
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                nof_rows: u32,
                nof_cols: u32,
                output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::matrix_transpose_ffi(
                        input.as_ptr(),
                        nof_rows,
                        nof_cols,
                        cfg as *const VecOpsConfig,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn bit_reverse(
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::bit_reverse_ffi(
                        input.as_ptr(),
                        input.len() as u64,
                        cfg as *const VecOpsConfig,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn bit_reverse_inplace(
                input: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::bit_reverse_ffi(
                        input.as_ptr(),
                        input.len() as u64,
                        cfg as *const VecOpsConfig,
                        input.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn slice(
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                offset: u64,
                stride: u64,
                size_in: u64,
                size_out: u64,
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::slice_ffi(
                        input.as_ptr(),
                        offset,
                        stride,
                        size_in,
                        size_out,
                        cfg as *const VecOpsConfig,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_ops_tests {
    (
      $field_type:ident
    ) => {
        pub(crate) mod test_vecops {
            use super::*;
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_vec_ops_scalars() {
                initialize();
                check_vec_ops_scalars::<$field_type>()
            }

            #[test]
            pub fn test_matrix_transpose() {
                initialize();
                check_matrix_transpose::<$field_type>()
            }

            #[test]
            pub fn test_bit_reverse() {
                initialize();
                check_bit_reverse::<$field_type>()
            }

            #[test]
            pub fn test_bit_reverse_inplace() {
                initialize();
                check_bit_reverse_inplace::<$field_type>()
            }

            #[test]
            pub fn test_slice() {
                initialize();
                check_slice::<$field_type>()
            }
        }
    };
}
