use crate::polynomial_ring::PolyRing;
use crate::ring::{ScalarCfg, ScalarCfgRns, ScalarRing, ScalarRingRns};
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("labrador_poly_ring", labrador_poly_ring, PolyRing, PolyRing);
impl_matrix_ops!("labrador", labrador, ScalarRing, ScalarCfg);
impl_matrix_ops!("labrador_rns", labrador_rns, ScalarRingRns, ScalarCfgRns);

#[cfg(test)]
mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::ScalarRing;
    use icicle_core::impl_matrix_ops_tests;
    use icicle_core::matrix_ops::MatrixOps;
    use icicle_core::polynomial_ring::PolynomialRing;
    use icicle_core::traits::GenerateRandom;
    use icicle_core::vec_ops::VecOpsConfig;
    use icicle_runtime::memory::HostSlice;
    // use icicle_runtime::test_utilities;
    
    impl_matrix_ops_tests!(ScalarRing);
    
    // mod rns { // TODO: add RNS tests ?
    //     use super::*;
    //     impl_matrix_ops_tests!(ScalarRingRns);
    // }
    
    mod poly_ring {
        use super::*;
        
        fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }
        
        #[test]
        fn test_polyring_matmul_device_memory() {
            initialize();
            check_polyring_matmul_device_memory();
        }
        
        #[test]
        fn test_polyring_matrix_transpose() {
            initialize();
            check_polyring_matrix_transpose_device_memory();
        }
        
        fn check_polyring_matmul_device_memory() {
            let cfg = VecOpsConfig::default();
            let n = 1 << 3; // Smaller size for polynomial rings
            let m = 1 << 4;
            let k = 1 << 2;
            let out_size = n * k;
            let input_a = PolyRing::generate_random(n * m);
            let input_b = PolyRing::generate_random(m * k);
            
            let mut output = vec![PolyRing::zero(); out_size];
            
            // Use the specific MatrixOps implementation for PolyRing
            PolyRing::matmul(
                HostSlice::from_slice(&input_a),
                n as u32,
                m as u32,
                HostSlice::from_slice(&input_b),
                m as u32,
                k as u32,
                &cfg,
                HostSlice::from_mut_slice(&mut output),
            )
            .expect("PolyRing matmul failed");
            
            // Basic check that output is not all zeros
            let has_non_zero = output.iter().any(|p| p != &PolyRing::zero());
            assert!(has_non_zero, "PolyRing matmul produced all zeros");
        }
        
        fn check_polyring_matrix_transpose_device_memory() {
            let cfg = VecOpsConfig::default();
            let nof_rows = 1 << 3;
            let nof_cols = 1 << 4;
            let matrix_size = nof_rows * nof_cols;
            
            let input_matrix = PolyRing::generate_random(matrix_size);
            let mut output_matrix = vec![PolyRing::zero(); matrix_size];
            
            // Use the specific MatrixOps implementation for PolyRing
            PolyRing::matrix_transpose(
                HostSlice::from_slice(&input_matrix),
                nof_rows as u32,
                nof_cols as u32,
                &cfg,
                HostSlice::from_mut_slice(&mut output_matrix),
            )
            .expect("PolyRing matrix transpose failed");
            
            // Basic check that transpose changed the data
            assert_ne!(input_matrix, output_matrix, "PolyRing transpose should modify data");
            
            // Transpose back and check we get original
            let mut restored_matrix = vec![PolyRing::zero(); matrix_size];
            PolyRing::matrix_transpose(
                HostSlice::from_slice(&output_matrix),
                nof_cols as u32,
                nof_rows as u32,
                &cfg,
                HostSlice::from_mut_slice(&mut restored_matrix),
            )
            .expect("PolyRing matrix transpose back failed");
            
            assert_eq!(input_matrix, restored_matrix, "Double transpose should restore original");
        }
    }
}
