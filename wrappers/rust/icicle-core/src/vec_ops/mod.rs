use crate::ring::IntegerRing;
use icicle_runtime::{
    config::ConfigExtension, eIcicleError, errors::IcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};

pub mod poly_vecops;
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
pub trait VecOps<T> {
    fn add(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn accumulate(
        a: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn sub(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn mul(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn div(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn inv(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn sum(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn product(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn scalar_add(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn scalar_sub(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn scalar_mul(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn bit_reverse(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), IcicleError>;

    fn bit_reverse_inplace(
        input: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;

    fn slice(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        offset: u64,
        stride: u64,
        size_in: u64,
        size_out: u64,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), IcicleError>;
}

#[doc(hidden)]
pub trait MixedVecOps<T, S: IntegerRing> {
    fn mul(
        a: &(impl HostOrDeviceSlice<S> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<S> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), IcicleError>;
}

fn check_vec_ops_args<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<VecOpsConfig, IcicleError> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            format!(
                "left, right and output lengths {}; {}; {} do not match",
                a.len(),
                b.len(),
                result.len()
            ),
        ));
    }
    setup_config(
        a, b, result, cfg, 1, /* Placeholder no need for batch_size in this operation */
    )
}

fn check_vec_ops_args_scalar_ops<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<VecOpsConfig, IcicleError> {
    if b.len() != result.len() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            format!("b.len() and result.len() do not match {} != {}", b.len(), result.len()),
        ));
    }
    if b.len() % a.len() != 0 {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            format!("b.len(), a.len() do not match {} % {} != 0", b.len(), a.len()),
        ));
    }
    let batch_size = a.len();
    setup_config(a, b, result, cfg, batch_size)
}

fn check_vec_ops_args_reduction_ops<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<VecOpsConfig, IcicleError> {
    if input.len() % result.len() != 0 {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            format!(
                "input length and result length do not match {} % {} != 0",
                input.len(),
                cfg.batch_size
            ),
        ));
    }
    let batch_size = result.len();
    setup_config(input, input, result, cfg, batch_size)
}

fn check_vec_ops_args_slice<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    offset: u64,
    stride: u64,
    size_in: u64,
    size_out: u64,
    output: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<VecOpsConfig, IcicleError> {
    if input.len() as u64 % size_in != 0 {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            format!("size_in does not divide input size {} % {} != 0", input.len(), size_in),
        ));
    }
    if output.len() as u64 % size_out != 0 {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            format!(
                "size_out does not divide output size {} % {} != 0",
                output.len(),
                size_out
            ),
        ));
    }
    if offset + (size_out - 1) * stride >= size_in {
        return Err(IcicleError::new(eIcicleError::InvalidArgument, format!("Slice exceed input size: offset + (size_out - 1) * stride >= size_in where offset={}, size_out={}, stride={}, size_in={}", offset, size_out, stride, size_in)));
    }
    let batch_size = output.len() / size_out as usize;
    setup_config(input, input, output, cfg, batch_size)
}

/// Modify VecopsConfig according to the given vectors
fn setup_config<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    batch_size: usize,
) -> Result<VecOpsConfig, IcicleError> {
    // check device slices are on active device
    if a.is_on_device() && !a.is_on_active_device() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "input a is allocated on an inactive device",
        ));
    }
    if b.is_on_device() && !b.is_on_active_device() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "input b is allocated on an inactive device",
        ));
    }
    if result.is_on_device() && !result.is_on_active_device() {
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "output is allocated on an inactive device",
        ));
    }

    let mut res_cfg = cfg.clone();
    res_cfg.batch_size = batch_size as i32;
    res_cfg.is_a_on_device = a.is_on_device();
    res_cfg.is_b_on_device = b.is_on_device();
    res_cfg.is_result_on_device = result.is_on_device();
    Ok(res_cfg)
}

pub fn add_scalars<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args(a, b, result, cfg)?;
    <T as VecOps<T>>::add(a, b, result, &cfg)
}

pub fn accumulate_scalars<T: IntegerRing + VecOps<T>>(
    a: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args(a, b, a, cfg)?;
    <T as VecOps<T>>::accumulate(a, b, &cfg)
}

pub fn sub_scalars<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    T: IntegerRing + VecOps<T>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg)?;
    <T as VecOps<T>>::sub(a, b, result, &cfg)
}

pub fn mul_scalars<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    T: IntegerRing + VecOps<T>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg)?;
    <T as VecOps<T>>::mul(a, b, result, &cfg)
}

pub fn mixed_mul_scalars<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    F: IntegerRing + MixedVecOps<T, F>,
    T: IntegerRing,
{
    let cfg = check_vec_ops_args(a, b, result, cfg)?;
    <F as MixedVecOps<T, F>>::mul(a, b, result, &cfg)
}

pub fn div_scalars<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError>
where
    T: IntegerRing + VecOps<T>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg)?;
    <T as VecOps<T>>::div(a, b, result, &cfg)
}

pub fn inv_scalars<T: IntegerRing + VecOps<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args(input, input, output, cfg)?;
    <T as VecOps<T>>::inv(input, output, &cfg)
}

pub fn sum_scalars<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args_reduction_ops(a, result, cfg)?;
    <T as VecOps<T>>::sum(a, result, &cfg)
}

pub fn product_scalars<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args_reduction_ops(a, result, cfg)?;
    <T as VecOps<T>>::product(a, result, &cfg)
}

pub fn scalar_add<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args_scalar_ops(a, b, result, cfg)?;
    <T as VecOps<T>>::scalar_add(a, b, result, &cfg)
}

pub fn scalar_sub<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args_scalar_ops(a, b, result, cfg)?;
    <T as VecOps<T>>::scalar_sub(a, b, result, &cfg)
}

pub fn scalar_mul<T: IntegerRing + VecOps<T>>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args_scalar_ops(a, b, result, cfg)?;
    <T as VecOps<T>>::scalar_mul(a, b, result, &cfg)
}

pub fn bit_reverse<T: IntegerRing + VecOps<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args(input, input /*dummy*/, output, cfg)?;
    <T as VecOps<T>>::bit_reverse(input, &cfg, output)
}

pub fn bit_reverse_inplace<T: IntegerRing + VecOps<T>>(
    input: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args(input, input /*dummy*/, input, cfg)?;
    <T as VecOps<T>>::bit_reverse_inplace(input, &cfg)
}

pub fn slice<T: IntegerRing + VecOps<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    offset: u64,
    stride: u64,
    size_in: u64,
    size_out: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), IcicleError> {
    let cfg = check_vec_ops_args_slice(input, offset, stride, size_in, size_out, output, cfg)?;
    <T as VecOps<T>>::slice(input, offset, stride, size_in, size_out, &cfg, output)
}

#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! impl_vec_ops_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident
    ) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::{eIcicleError, IcicleError};

            extern "C" {
                #[link_name = concat!($field_prefix, "_vector_add")]
                pub(crate) fn vector_add_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_accumulate")]
                pub(crate) fn vector_accumulate_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_sub")]
                pub(crate) fn vector_sub_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_mul")]
                pub(crate) fn vector_mul_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_div")]
                pub(crate) fn vector_div_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_inv")]
                pub(crate) fn vector_inv_ffi(
                    input: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    output: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_sum")]
                pub(crate) fn vector_sum_ffi(
                    a: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_vector_product")]
                pub(crate) fn vector_product_ffi(
                    a: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_scalar_add_vec")]
                pub(crate) fn scalar_add_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_scalar_sub_vec")]
                pub(crate) fn scalar_sub_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_scalar_mul_vec")]
                pub(crate) fn scalar_mul_ffi(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_bit_reverse")]
                pub(crate) fn bit_reverse_ffi(
                    input: *const $field,
                    size: u64,
                    config: *const VecOpsConfig,
                    output: *mut $field,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_slice")]
                pub(crate) fn slice_ffi(
                    input: *const $field,
                    offset: u64,
                    stride: u64,
                    size_in: u64,
                    size_out: u64,
                    cfg: *const VecOpsConfig,
                    output: *mut $field,
                ) -> eIcicleError;

            }
        }

        impl VecOps<$field> for $field {
            fn add(
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
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
                a: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
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
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
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
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
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
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
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

            fn inv(
                input: &(impl HostOrDeviceSlice<Self> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::vector_inv_ffi(
                        input.as_ptr(),
                        input.len() as u32,
                        cfg as *const VecOpsConfig,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn sum(
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::vector_sum_ffi(
                        a.as_ptr(),
                        a.len() as u32 / cfg.batch_size as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn product(
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::vector_product_ffi(
                        a.as_ptr(),
                        a.len() as u32 / cfg.batch_size as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn scalar_add(
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_add_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        b.len() as u32 / cfg.batch_size as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn scalar_sub(
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_sub_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        b.len() as u32 / cfg.batch_size as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn scalar_mul(
                a: &(impl HostOrDeviceSlice<Self> + ?Sized),
                b: &(impl HostOrDeviceSlice<Self> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_mul_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        b.len() as u32 / cfg.batch_size as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn bit_reverse(
                input: &(impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::bit_reverse_ffi(
                        input.as_ptr(),
                        input.len() as u64 / cfg.batch_size as u64,
                        cfg as *const VecOpsConfig,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn bit_reverse_inplace(
                input: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
                unsafe {
                    $field_prefix_ident::bit_reverse_ffi(
                        input.as_ptr(),
                        input.len() as u64 / cfg.batch_size as u64,
                        cfg as *const VecOpsConfig,
                        input.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn slice(
                input: &(impl HostOrDeviceSlice<Self> + ?Sized),
                offset: u64,
                stride: u64,
                size_in: u64,
                size_out: u64,
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
            ) -> Result<(), IcicleError> {
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
#[allow(clippy::crate_in_macro_def)]
macro_rules! impl_vec_ops_mixed_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $ext_field:ident,
        $field:ident
    ) => {
        mod $field_prefix_ident {

            use crate::vec_ops::{$ext_field, $field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;

            extern "C" {
                #[link_name = concat!($field_prefix, "_vector_mixed_mul")]
                pub(crate) fn vector_mul_ffi(
                    a: *const $ext_field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $ext_field,
                ) -> eIcicleError;
            }
        }

        impl MixedVecOps<$field, $ext_field> for $ext_field {
            fn mul(
                a: &(impl HostOrDeviceSlice<$ext_field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$ext_field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), IcicleError> {
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
        }
    };
}

#[macro_export]
macro_rules! impl_vec_ops_tests {
    (
      $field_prefix_ident: ident,
      $field:ident
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
            pub fn test_vec_ops_scalars_add() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_add::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_sub() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_sub::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_mul() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_mul::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_div() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_div::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_sum() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_sum::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_product() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_product::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_add_scalar() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_add_scalar::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_sub_scalar() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_sub_scalar::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_mul_scalar() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_mul_scalar::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_accumulate() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_accumulate::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_inv() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_inv::<$field>(test_size);
            }

            #[test]
            pub fn test_bit_reverse() {
                initialize();
                check_bit_reverse::<$field>()
            }

            #[test]
            pub fn test_bit_reverse_inplace() {
                initialize();
                check_bit_reverse_inplace::<$field>()
            }

            #[test]
            pub fn test_slice() {
                initialize();
                check_slice::<$field>()
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mixed_vec_ops_tests {
    (
      $ext_field:ident,
      $field:ident
    ) => {
        pub(crate) mod test_mixed_vecops {
            use super::*;
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_mixed_vec_ops_scalars() {
                initialize();
                check_mixed_vec_ops_scalars::<$ext_field, $field>()
            }
        }
    };
}
