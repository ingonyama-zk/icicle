mod tests;

use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

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

    //TODO Lisa: Transpose
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
            //use crate::polynomial_ring::$poly_type;
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
                    unsafe {
                        labrador::matmul_ffi(
                            a.as_ptr(),
                            nof_rows_a,
                            nof_cols_a,
                            b.as_ptr(),
                            nof_rows_b,
                            nof_cols_b,
                            cfg,
                            result.as_mut_ptr(),
                        )
                        .wrap()
                    }
                }
            }
        }
    };
}
