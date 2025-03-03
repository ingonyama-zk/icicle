use crate::traits::FieldImpl;
use crate::program::Program;
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
pub trait VecOps<F> {
    fn add(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn accumulate(
        a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn sub(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn div(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn sum(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn product(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn scalar_add(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn scalar_sub(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn scalar_mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn transpose(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        nof_rows: u32,
        nof_cols: u32,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn bit_reverse(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn bit_reverse_inplace(
        input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn slice(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        offset: u64,
        stride: u64,
        size_in: u64,
        size_out: u64,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn execute_program<Prog, Data>(
        data: &mut Vec<&Data>,
        program: &Prog,
        cfg: &VecOpsConfig
    ) -> Result<(), eIcicleError>
    where
        F: FieldImpl,
        <F as FieldImpl>::Config: VecOps<F>,
        Data: HostOrDeviceSlice<F> + ?Sized,
        Prog: Program<F>;
}

#[doc(hidden)]
pub trait MixedVecOps<F, T> {
    fn mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;
}

fn check_vec_ops_args<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
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
    setup_config(
        a, b, result, cfg, 1, /* Placeholder no need for batch_size in this operation */
    )
}

fn check_vec_ops_args_scalar_ops<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> VecOpsConfig {
    if b.len() != result.len() {
        panic!("b.len() and result.len() do not match {} != {}", b.len(), result.len());
    }
    if b.len() % a.len() != 0 {
        panic!("b.len(), a.len() do not match {} % {} != 0", b.len(), a.len(),);
    }
    let batch_size = a.len();
    setup_config(a, b, result, cfg, batch_size)
}

fn check_vec_ops_args_reduction_ops<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> VecOpsConfig {
    if input.len() % result.len() != 0 {
        panic!(
            "input length and result length do not match {} % {} != 0",
            input.len(),
            cfg.batch_size,
        );
    }
    let batch_size = result.len();
    setup_config(input, input, result, cfg, batch_size)
}

fn check_vec_ops_args_transpose<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    output: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> VecOpsConfig {
    if input.len() != output.len() {
        panic!(
            "Input size, and output size do not match {} != {}",
            input.len(),
            output.len()
        );
    }
    if input.len() as u32 % (nof_rows * nof_cols) != 0 {
        panic!(
            "Input size is not a whole multiple of matrix size (#rows * #cols), {} % ({} * {}) != 0",
            input.len(),
            nof_rows,
            nof_cols,
        );
    }
    let batch_size = input.len() / (nof_rows * nof_cols) as usize;
    setup_config(input, input, output, cfg, batch_size)
}

fn check_vec_ops_args_slice<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    offset: u64,
    stride: u64,
    size_in: u64,
    size_out: u64,
    output: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> VecOpsConfig {
    if input.len() as u64 % size_in != 0 {
        panic!("size_in does not divide input size {} % {} != 0", input.len(), size_in,);
    }
    if output.len() as u64 % size_out != 0 {
        panic!(
            "size_out does not divide output size {} % {} != 0",
            output.len(),
            size_out,
        );
    }
    if offset + (size_out - 1) * stride >= size_in {
        panic!(
            "Slice exceed input size: offset + (size_out - 1) * stride >= size_in where offset={}, size_out={}, stride={}, size_in={}",
            offset,
            size_out,
            stride,
            size_in,
        );
    }
    let batch_size = output.len() / size_out as usize;
    setup_config(input, input, output, cfg, batch_size)
}

fn check_execute_program<F, Data>(
    data: &Vec<&Data>,
    cfg: &VecOpsConfig
) -> VecOpsConfig
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
    Data: HostOrDeviceSlice<F> + ?Sized,
{
    // All parameters' config should match so each one is compared to the first one
    let nof_iterations = data[0].len();
    let is_on_device = data[0].is_on_device();

    for i in 1..data.len() {
        if data[i].len() != nof_iterations {
            panic!("First parameter length ({}) and parameter[{}] length do not match", nof_iterations, data[i].len());
        }
        if data[i].is_on_device() != is_on_device {
            panic!("First parameter length ({}) and parameter[{}] length ({}) do not match",
                nof_iterations, i, data[i].len());
        }
    }
    setup_config(data[0], data[0], data[0], cfg, 1)
}

/// Modify VecopsConfig according to the given vectors
fn setup_config<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    batch_size: usize,
) -> VecOpsConfig {
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
    res_cfg.batch_size = batch_size as i32;
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
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::add(a, b, result, &cfg)
}

pub fn accumulate_scalars<F>(
    a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, a, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::accumulate(a, b, &cfg)
}

pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
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
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::mul(a, b, result, &cfg)
}

pub fn mixed_mul_scalars<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: MixedVecOps<F, T>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as MixedVecOps<F, T>>::mul(a, b, result, &cfg)
}

pub fn div_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::div(a, b, result, &cfg)
}

pub fn sum_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_reduction_ops(a, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::sum(a, result, &cfg)
}

pub fn product_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_reduction_ops(a, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::product(a, result, &cfg)
}

pub fn scalar_add<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_scalar_ops(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::scalar_add(a, b, result, &cfg)
}

pub fn scalar_sub<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_scalar_ops(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::scalar_sub(a, b, result, &cfg)
}

pub fn scalar_mul<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_scalar_ops(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::scalar_mul(a, b, result, &cfg)
}

pub fn transpose_matrix<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_transpose(input, nof_rows, nof_cols, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::transpose(input, nof_rows, nof_cols, output, &cfg)
}

pub fn bit_reverse<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(input, input /*dummy*/, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::bit_reverse(input, &cfg, output)
}

pub fn bit_reverse_inplace<F>(
    input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(input, input /*dummy*/, input, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::bit_reverse_inplace(input, &cfg)
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
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args_slice(input, offset, stride, size_in, size_out, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::slice(input, offset, stride, size_in, size_out, &cfg, output)
}

pub fn execute_program<F, Prog, Data>(
    data: &mut Vec<&Data>,
    program: &Prog,
    cfg: &VecOpsConfig
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
    Data: HostOrDeviceSlice<F> + ?Sized,
    Prog: Program<F>,
{
    let cfg = check_execute_program(&data, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::execute_program(data, program, &cfg)
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
            use icicle_core::program::{Program, ProgramHandle};
            use icicle_core::symbol::Symbol;
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;

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

                #[link_name = concat!($field_prefix, "_matrix_transpose")]
                pub(crate) fn matrix_transpose_ffi(
                    input: *const $field,
                    nof_rows: u32,
                    nof_cols: u32,
                    cfg: *const VecOpsConfig,
                    output: *mut $field,
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

                #[link_name = concat!($field_prefix, "_execute_program")]
                pub(crate) fn execute_program_ffi(
                    data_ptr: *const *const $field,
                    nof_params: u64,
                    program: ProgramHandle,
                    nof_iterations: u64,
                    cfg: *const VecOpsConfig
                ) -> eIcicleError;
            }
        }

        impl VecOps<$field> for $field_config {
            fn add(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
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
                a: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
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

            fn scalar_add(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
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
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
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

            fn transpose(
                input: &(impl HostOrDeviceSlice<$field> + ?Sized),
                nof_rows: u32,
                nof_cols: u32,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
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
                input: &(impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
            ) -> Result<(), eIcicleError> {
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
                input: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
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
                input: &(impl HostOrDeviceSlice<$field> + ?Sized),
                offset: u64,
                stride: u64,
                size_in: u64,
                size_out: u64,
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
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

            fn execute_program<Prog, Data>(
                data: &mut Vec<&Data>,
                program: &Prog,
                cfg: &VecOpsConfig
            ) -> Result<(), eIcicleError>
            where
                <$field as FieldImpl>::Config: VecOps<$field>,
                Data: HostOrDeviceSlice<$field> + ?Sized,
                Prog: Program<$field>,
            {
                unsafe {
                    let data_vec: Vec<*const $field> = data.iter().map(|s| s.as_ptr()).collect();
                    $field_prefix_ident::execute_program_ffi(
                        data_vec.as_ptr(),
                        data.len() as u64,
                        program.handle(),
                        data[0].len() as u64,
                        cfg as *const VecOpsConfig
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_ops_mixed_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $ext_field:ident,
        $field:ident,
        $ext_field_config:ident
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

        impl MixedVecOps<$ext_field, $field> for $ext_field_config {
            fn mul(
                a: &(impl HostOrDeviceSlice<$ext_field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$ext_field> + ?Sized),
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
            use crate::program::$field_prefix_ident::{FieldProgram, FieldReturningValueProgram};

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_vec_ops_scalars() {
                initialize();
                check_vec_ops_scalars::<$field>()
            }

            #[test]
            pub fn test_matrix_transpose() {
                initialize();
                check_matrix_transpose::<$field>()
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

            #[test]
            pub fn test_program() {
                initialize();
                test_utilities::test_set_main_device();
                check_program::<$field, FieldProgram>();
                test_utilities::test_set_ref_device();
                check_program::<$field, FieldProgram>()
            }

            #[test]
            pub fn test_predefined_program() {
                initialize();
                test_utilities::test_set_main_device();
                check_predefined_program::<$field, FieldProgram>();
                test_utilities::test_set_ref_device();
                check_predefined_program::<$field, FieldProgram>()
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
