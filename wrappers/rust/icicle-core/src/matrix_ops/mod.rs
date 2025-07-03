use crate::vec_ops::{setup_config, VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

fn check_matrix_ops_args_transpose<F>(
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

/// Trait defining matrix operations for types stored in host or device memory.
///
/// ### Supported Operations
/// - **Matrix multiplication (`matmul`)**  
///   Multiplies two matrices `a` and `b`, both stored in **row-major** (rows-first) order,
///   and writes the result to `result`, also in row-major order.
///
///   Dimensions must satisfy:
///   - `a` is of shape `(a_rows × a_cols)`
///   - `b` is of shape `(b_rows × b_cols)`
///   - `a_cols == b_rows`
///   - `result` must be sized to hold `(a_rows × b_cols)` elements
///
///   Memory for all inputs and the result can reside on either the host or device.
pub trait MatrixOps<T> {
    fn matmul(
        a: &(impl HostOrDeviceSlice<T> + ?Sized),
        a_rows: u32,
        a_cols: u32,
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        b_rows: u32,
        b_cols: u32,
        cfg: &VecOpsConfig,
        result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn transpose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        nof_rows: u32,
        nof_cols: u32,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

pub fn matmul<T>(
    a: &(impl HostOrDeviceSlice<T> + ?Sized),
    a_rows: u32,
    a_cols: u32,
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    b_rows: u32,
    b_cols: u32,
    cfg: &VecOpsConfig,
    result: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    T: MatrixOps<T>,
{
    T::matmul(a, a_rows, a_cols, b, b_rows, b_cols, cfg, result)
}

pub fn matrix_transpose<T>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    T: MatrixOps<T>,
{
    let cfg = check_matrix_ops_args_transpose(input, nof_rows, nof_cols, output, cfg);
    T::transpose(input, nof_rows, nof_cols, &cfg, output)
}

/// Implements matrix multiplication over polynomial rings via FFI
#[macro_export]
macro_rules! impl_matrix_ops {
    ($prefix: literal, $prefix_ident:ident, $element_type: ty) => {
        mod $prefix_ident {
            use crate::matrix_ops::$prefix_ident;
            use icicle_core::{matrix_ops::MatrixOps, vec_ops::VecOpsConfig};
            use icicle_runtime::errors::eIcicleError;
            use icicle_runtime::memory::HostOrDeviceSlice;

            extern "C" {
                #[link_name = concat!($prefix, "_matmul")]
                pub(crate) fn matmul_ffi(
                    a: *const $element_type,
                    a_rows: u32,
                    a_cols: u32,
                    b: *const $element_type,
                    b_rows: u32,
                    b_cols: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $element_type,
                ) -> eIcicleError;

                #[link_name = concat!($prefix, "_matrix_transpose")]
                pub(crate) fn matrix_transpose_ffi(
                    input: *const $element_type,
                    nof_rows: u32,
                    nof_cols: u32,
                    cfg: *const VecOpsConfig,
                    output: *mut $element_type,
                ) -> eIcicleError;
            }

            impl MatrixOps<$element_type> for $element_type {
                fn matmul(
                    a: &(impl HostOrDeviceSlice<$element_type> + ?Sized),
                    nof_rows_a: u32,
                    nof_cols_a: u32,
                    b: &(impl HostOrDeviceSlice<$element_type> + ?Sized),
                    nof_rows_b: u32,
                    nof_cols_b: u32,
                    cfg: &VecOpsConfig,
                    result: &mut (impl HostOrDeviceSlice<$element_type> + ?Sized),
                ) -> Result<(), eIcicleError> {
                    if a.len() as u32 != nof_rows_a * nof_cols_a {
                        eprintln!(
                            "Matrix A has invalid size: got {}, expected {} ({} × {})",
                            a.len(),
                            nof_rows_a * nof_cols_a,
                            nof_rows_a,
                            nof_cols_a
                        );
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if b.len() as u32 != nof_rows_b * nof_cols_b {
                        eprintln!(
                            "Matrix B has invalid size: got {}, expected {} ({} × {})",
                            b.len(),
                            nof_rows_b * nof_cols_b,
                            nof_rows_b,
                            nof_cols_b
                        );
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if result.len() as u32 != nof_rows_a * nof_cols_b {
                        eprintln!(
                            "Result matrix has invalid size: got {}, expected {} ({} × {})",
                            result.len(),
                            nof_rows_a * nof_cols_b,
                            nof_rows_a,
                            nof_cols_b
                        );
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if result.is_on_device() && !result.is_on_active_device() {
                        eprintln!("Result matrix is on an inactive device");
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if a.is_on_device() && !a.is_on_active_device() {
                        eprintln!("Input a is on an inactive device");
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if b.is_on_device() && !b.is_on_active_device() {
                        eprintln!("Input b  is on an inactive device");
                        return Err(eIcicleError::InvalidArgument);
                    }

                    let mut cfg_clone = cfg.clone();
                    cfg_clone.is_a_on_device = a.is_on_device();
                    cfg_clone.is_b_on_device = b.is_on_device();
                    cfg_clone.is_result_on_device = result.is_on_device();

                    unsafe {
                        matmul_ffi(
                            a.as_ptr(),
                            nof_rows_a,
                            nof_cols_a,
                            b.as_ptr(),
                            nof_rows_b,
                            nof_cols_b,
                            &cfg_clone,
                            result.as_mut_ptr(),
                        )
                        .wrap()
                    }
                }

                fn transpose(
                    input: &(impl HostOrDeviceSlice<$element_type> + ?Sized),
                    nof_rows: u32,
                    nof_cols: u32,
                    cfg: &VecOpsConfig,
                    output: &mut (impl HostOrDeviceSlice<$element_type> + ?Sized),
                ) -> Result<(), eIcicleError> {
                    unsafe {
                        matrix_transpose_ffi(
                            input.as_ptr(),
                            nof_rows,
                            nof_cols,
                            cfg as *const VecOpsConfig,
                            output.as_mut_ptr(),
                        )
                        .wrap()
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_matrix_ops_tests {
    ($element_type:ty) => {
        #[cfg(test)]
        mod test_matmul_device_memory {

            use icicle_core::matrix_ops::tests::*;
            use icicle_runtime::test_utilities;

            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_matmul_device_memory() {
                initialize();
                check_matmul_device_memory::<$element_type>();
            }

            #[test]
            pub fn test_matrix_transpose() {
                initialize();
                check_matrix_transpose::<$element_type>()
            }
        }
    };
}
