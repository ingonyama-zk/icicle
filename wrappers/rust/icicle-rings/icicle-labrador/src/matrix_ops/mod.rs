



mod labrador {
    use crate::matrix_ops::labrador;
    use crate::polynomial_ring::PolyRing;
    use icicle_core::{matrix_ops::MatrixOps,vec_ops::VecOpsConfig};
    use icicle_runtime::errors::eIcicleError;
    use icicle_runtime::memory::HostOrDeviceSlice;

    //const polyring_prefix: &str = "labrador";


    extern "C" {
        #[link_name = concat!("labrador", "_matmul")]
        pub(crate) fn matmul_ffi(
            a: *const PolyRing,
            a_rows: u32,
            a_cols: u32,
            b: *const PolyRing,
            b_rows: u32,
            b_cols: u32,
            cfg: *const VecOpsConfig,
            result: *mut PolyRing,
        ) -> eIcicleError;
    }

    impl MatrixOps<PolyRing> for PolyRing{
        fn matmul(
            a: &(impl HostOrDeviceSlice<PolyRing> + ?Sized),
            nof_rows_a: u32,
            nof_cols_a: u32,
            b: &(impl HostOrDeviceSlice<PolyRing> + ?Sized),
            nof_rows_b: u32,
            nof_cols_b: u32,
            cfg: &VecOpsConfig,
            result: &mut (impl HostOrDeviceSlice<PolyRing> + ?Sized),
        ) -> Result<(), eIcicleError> { 
            unsafe {
                labrador::matmul_ffi(a.as_ptr(), nof_rows_a, nof_cols_a, b.as_ptr(), nof_rows_b, nof_cols_b, cfg, result.as_mut_ptr()).wrap()
            }
        }
    }
}

// #[cfg(test)]
// pub(crate) mod tests {
//     use crate::matrix_ops::labrador;
//     use icicle_core::{matrix_ops::MatrixOps::matmul,vec_ops::VecOpsConfig, traits::*};
//     use crate::polynomial_ring::PolyRing;

//     #[test]
//     fn test_matmul() {
//         let a = HostSlice::from_slice(&PolyRing::generate_random(4));
//         let b = HostSlice::from_slice(&PolyRing::generate_random(4));
//         let result = HostSlice::from_slice(&PolyRing::generate_random(4));
//         let cfg = VecOpsConfig::default();
//         let result = matmul(&a, 2, 2, &b, 2, 2, &cfg, &mut result);
//         assert!(result.is_ok());
//     }
// }  