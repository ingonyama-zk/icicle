use crate::traits::FieldImpl;
use icicle_runtime::{
    config::ConfigExtension,
    errors::eIcicleError,
    memory::HostOrDeviceSlice,
    stream::IcicleStreamHandle,
};

use super::VecOpsConfig;

/// Checks vector operation arguments for basic operations
pub(crate) fn check_vec_ops_args<F, T>(
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

/// Checks vector operation arguments for scalar operations
pub(crate) fn check_vec_ops_args_scalar_ops<F, T>(
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

/// Checks vector operation arguments for reduction operations
pub(crate) fn check_vec_ops_args_reduction_ops<F>(
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

/// Checks vector operation arguments for transpose operations
pub(crate) fn check_vec_ops_args_transpose<F>(
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

/// Checks vector operation arguments for slice operations
pub(crate) fn check_vec_ops_args_slice<F>(
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

/// Sets up the configuration for vector operations
pub(crate) fn setup_config<F, T>(
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