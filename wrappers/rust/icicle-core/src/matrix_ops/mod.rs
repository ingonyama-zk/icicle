//! Matrix operations over data stored in host or device memory.
//!
//! This module defines a trait [`MatrixOps`] and convenience functions for
//! performing matrix computations such as multiplication and transposition
//! across heterogeneous memory backends (CPU or GPU).
//!
//! ## Supported Operations
//!
//! - Matrix multiplication (`matmul`)
//! - Matrix transpose (`matrix_transpose`)
//!
//! All functions are backend-agnostic and dispatched using [`VecOpsConfig`].

use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

/// Trait defining matrix operations over row-major matrices stored in host or device memory.
pub trait MatrixOps<T> {
    /// Performs matrix multiplication: `result = a × b`
    ///
    /// - `a`: shape `(a_rows × a_cols)` (row-major)
    /// - `b`: shape `(b_rows × b_cols)` (row-major)
    /// - `result`: shape `(a_rows × b_cols)` (row-major, must be preallocated)
    ///
    /// Requirements:
    /// - `a_cols == b_rows`
    /// - All buffers may reside in host or device memory
    ///
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

    /// Computes the transpose of a matrix in row-major order.
    ///
    /// - `input`: shape `(nof_rows × nof_cols)`
    /// - `output`: shape `(nof_cols × nof_rows)` (must be preallocated)
    ///
    /// Both input and output can reside on host or device memory.
    fn matrix_transpose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        nof_rows: u32,
        nof_cols: u32,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

/// Dispatches [`MatrixOps::matmul`] using the type `T`.
///
/// All matrices are expected to be in row-major order.
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

/// Dispatches [`MatrixOps::matrix_transpose`] using the type `T`.
///
/// All matrices are assumed to be in row-major order.
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
    T::matrix_transpose(input, nof_rows, nof_cols, cfg, output)
}

/// Implements matrix Ops any type via FFI
#[macro_export]
macro_rules! impl_matrix_ops {
    ($prefix: literal, $element_type: ty) => {
        mod labrador {
            use crate::matrix_ops::labrador;
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
                    nof_ows: u32,
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
                        labrador::matmul_ffi(
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

                fn matrix_transpose(
                    input: &(impl HostOrDeviceSlice<$element_type> + ?Sized),
                    nof_rows: u32,
                    nof_cols: u32,
                    cfg: &VecOpsConfig,
                    output: &mut (impl HostOrDeviceSlice<$element_type> + ?Sized),
                ) -> Result<(), eIcicleError> {
                    if input.len() as u32 != nof_rows * nof_cols {
                        eprintln!(
                            "Matrix A has invalid size: got {}, expected {} ({} × {})",
                            input.len(),
                            nof_rows * nof_cols,
                            nof_rows,
                            nof_cols
                        );
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if output.len() != input.len() {
                        eprintln!("Output matrix has invalid size",);
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if input.is_on_device() && !input.is_on_active_device() {
                        eprintln!("Input a is on an inactive device");
                        return Err(eIcicleError::InvalidArgument);
                    }

                    if output.is_on_device() && !output.is_on_active_device() {
                        eprintln!("Result matrix is on an inactive device");
                        return Err(eIcicleError::InvalidArgument);
                    }

                    let mut cfg_clone = cfg.clone();
                    cfg_clone.is_a_on_device = input.is_on_device();
                    cfg_clone.is_b_on_device = false;
                    cfg_clone.is_result_on_device = output.is_on_device();

                    unsafe {
                        labrador::matrix_transpose_ffi(
                            input.as_ptr(),
                            nof_rows,
                            nof_cols,
                            &cfg_clone,
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
    ($test_mod_name:ident, $element_type:ty) => {
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
        fn test_matrix_transpose() {
            initialize();
            check_matrix_transpose_device_memory::<$element_type>();
        }
    };
}
