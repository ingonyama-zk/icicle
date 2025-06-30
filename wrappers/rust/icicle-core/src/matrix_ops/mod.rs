pub use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

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

/// Implements matrix multiplication over polynomial rings via FFI
#[macro_export]
macro_rules! impl_matmul {
    ($prefix: literal, $poly_type: ty) => {
        mod labrador {
            use crate::matrix_ops::labrador;
            use icicle_core::{matrix_ops::MatrixOps, vec_ops::VecOpsConfig};
            use icicle_runtime::errors::eIcicleError;
            use icicle_runtime::memory::HostOrDeviceSlice;

            extern "C" {
                #[link_name = concat!($prefix, "_matmul")]
                pub(crate) fn matmul_ffi(
                    a: *const $poly_type,
                    a_rows: u32,
                    a_cols: u32,
                    b: *const $poly_type,
                    b_rows: u32,
                    b_cols: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $poly_type,
                ) -> eIcicleError;
            }

            impl MatrixOps<$poly_type> for $poly_type {
                fn matmul(
                    a: &(impl HostOrDeviceSlice<$poly_type> + ?Sized),
                    nof_rows_a: u32,
                    nof_cols_a: u32,
                    b: &(impl HostOrDeviceSlice<$poly_type> + ?Sized),
                    nof_rows_b: u32,
                    nof_cols_b: u32,
                    cfg: &VecOpsConfig,
                    result: &mut (impl HostOrDeviceSlice<$poly_type> + ?Sized),
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
            }
        }
    };
}

#[macro_export]
macro_rules! impl_matmul_device_tests {
    ($poly:ty) => {
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
                check_matmul_device_memory::<$poly>();
            }
        }
    };
}
