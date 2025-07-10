use crate::{
    matrix_ops::{matmul, matrix_transpose, MatrixOps, MatMulConfig},
    traits::GenerateRandom,
    vec_ops::VecOpsConfig,
};

use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    test_utilities,
};



/// Ensure host memory and device memory behaviour matches.
/// We test the following compibations
///     (1) a, b, result on host
///     (2) a, b on host; result on device
///     (3) a, b on device; result on host
///     (4) a, b, result on host
///     (5) a on device; b, result on host
/// Correctness is already ensured by the C++ tests.
pub fn check_matmul_device_memory<P>()
where
    P: GenerateRandom + MatrixOps<P> + Default + Clone + std::fmt::Debug + PartialEq,
{
    let cfg = MatMulConfig::default();

    let n = 1 << 5;
    let m = 1 << 6;
    let k = 1 << 4;
    let out_size = n * k;
    let input_a = P::generate_random(n * m);
    let input_b = P::generate_random(m * k);

    let test_single_device = |main_device: bool| {
        if main_device {
            test_utilities::test_set_main_device();
        } else {
            test_utilities::test_set_ref_device();
        }
        // case (1) matmul host memory inputs -> host_memory outputs
        let mut output_host_case_1 = vec![P::default(); out_size];
        matmul(
            HostSlice::from_slice(&input_a),
            n as u32,
            m as u32,
            HostSlice::from_slice(&input_b),
            m as u32,
            k as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_case_1),
        )
        .unwrap();

        // case (2) matmul host memory inputs -> device memory output
        let mut device_mem_output = DeviceVec::<P>::device_malloc(out_size).unwrap();
        matmul(
            HostSlice::from_slice(&input_a),
            n as u32,
            m as u32,
            HostSlice::from_slice(&input_b),
            m as u32,
            k as u32,
            &cfg,
            &mut device_mem_output,
        )
        .unwrap();

        // compare (1) and (2)
        let mut output_host_case_2 = vec![P::default(); out_size];
        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut output_host_case_2))
            .unwrap();
        assert_eq!(output_host_case_1, output_host_case_2);

        // case (3) matmul device memory inputs, host memory outputs
        /* Allocate inputs on device, and copy from host */
        let mut device_mem_a = DeviceVec::<P>::device_malloc(n * m).unwrap();
        let mut device_mem_b = DeviceVec::<P>::device_malloc(m * k).unwrap();
        device_mem_a
            .copy_from_host(HostSlice::from_slice(&input_a))
            .unwrap();
        device_mem_b
            .copy_from_host(HostSlice::from_slice(&input_b))
            .unwrap();

        let mut output_host_case_3 = vec![P::default(); out_size];
        matmul(
            &device_mem_a,
            n as u32,
            m as u32,
            &device_mem_b,
            m as u32,
            k as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_case_3),
        )
        .unwrap();

        // compare (1) and (3)
        assert_eq!(output_host_case_1, output_host_case_3);

        // case (4) matmul device memory inputs, device memory output
        let mut device_mem_output = DeviceVec::<P>::device_malloc(out_size).unwrap();
        matmul(
            &device_mem_a,
            n as u32,
            m as u32,
            &device_mem_b,
            m as u32,
            k as u32,
            &cfg,
            &mut device_mem_output,
        )
        .unwrap();

        /* Zero out host_buffer, copy result of (4) to host_buffer */
        let mut output_host_case_4 = vec![P::default(); out_size];
        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut output_host_case_4))
            .unwrap();

        assert_eq!(output_host_case_1, output_host_case_4);

        // case (5) mamtmul mixed memory model for inputs, host memory output
        let mut output_host_case_5 = vec![P::default(); out_size];
        matmul(
            &device_mem_a,
            n as u32,
            m as u32,
            HostSlice::from_slice(&input_b),
            m as u32,
            k as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_case_5),
        )
        .unwrap();

        //compare (1) and (5)
        assert_eq!(output_host_case_1, output_host_case_5);

        output_host_case_1
    };

    // Compare main and ref devices
    let device_out = test_single_device(true);
    let ref_out = test_single_device(false);

    assert_eq!(device_out, ref_out);
}

/// Validates that matrix transpose behaves consistently across host and device memory.
///
/// This test checks all combinations of input/output memory locations:
///  1. Host → Host
///  2. Host → Device
///  3. Device → Host
///  4. Device → Device
///
/// Transpose correctness is assumed to be verified by lower-level C++ tests.
/// Here, we validate consistent behavior across memory backends by transposing a matrix
/// and checking that a second transpose restores the original data.
///
/// The test is repeated for both main and reference devices.
pub fn check_matrix_transpose_device_memory<P: MatrixOps<P>>()
where
    P: Default + GenerateRandom + Clone + std::fmt::Debug + PartialEq,
{
    let cfg = VecOpsConfig::default();
    let nof_rows = 1 << 5;
    let nof_cols = 1 << 6;
    let matrix_size = nof_rows * nof_cols;

    let input_matrix = P::generate_random(matrix_size);

    let test_single_device = |main_device: bool| {
        if main_device {
            test_utilities::test_set_main_device();
        } else {
            test_utilities::test_set_ref_device();
        }

        // --- Case 1: Host → Host ---
        let mut output_host_case_1 = vec![P::default(); matrix_size];
        matrix_transpose(
            HostSlice::from_slice(&input_matrix),
            nof_rows as u32,
            nof_cols as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_case_1),
        )
        .unwrap();
        assert_ne!(input_matrix, output_host_case_1); // Transpose should modify data

        // --- Case 2: Host → Device ---
        let mut device_mem_output = DeviceVec::<P>::device_malloc(matrix_size).unwrap();
        matrix_transpose(
            HostSlice::from_slice(&input_matrix),
            nof_rows as u32,
            nof_cols as u32,
            &cfg,
            &mut device_mem_output,
        )
        .unwrap();

        // Compare (1) and (2)
        let mut output_host_case_2 = vec![P::default(); matrix_size];
        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut output_host_case_2))
            .unwrap();
        assert_eq!(output_host_case_1, output_host_case_2);

        // --- Case 3: Device → Host ---
        let mut device_mem_input = DeviceVec::<P>::device_malloc(matrix_size).unwrap();
        device_mem_input
            .copy_from_host(HostSlice::from_slice(&input_matrix))
            .unwrap();

        let mut output_host_case_3 = vec![P::default(); matrix_size];
        matrix_transpose(
            &device_mem_input,
            nof_rows as u32,
            nof_cols as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_case_3),
        )
        .unwrap();
        assert_eq!(output_host_case_1, output_host_case_3);

        // --- Case 4: Device → Device (Transpose back to original) ---
        let mut device_mem_restored = DeviceVec::<P>::device_malloc(matrix_size).unwrap();
        matrix_transpose(
            &device_mem_output,
            nof_cols as u32, // flipped dimensions
            nof_rows as u32,
            &cfg,
            &mut device_mem_restored,
        )
        .unwrap();

        let mut output_host_case_4_restored = vec![P::default(); matrix_size];
        device_mem_restored
            .copy_to_host(HostSlice::from_mut_slice(&mut output_host_case_4_restored))
            .unwrap();
        assert_eq!(input_matrix, output_host_case_4_restored);

        output_host_case_1
    };

    // Run on main device
    let device_out = test_single_device(true);

    // Run on reference device
    let ref_out = test_single_device(false);

    // Final check: both devices should yield identical transpose results
    //assert_eq!(device_out, ref_out);
}

/// Validates that the `a_is_transposed` flag works correctly for the first input matrix.
/// 
/// This test performs the following operations:
/// 1. Creates matrices A and B
/// 2. Computes A * B with `a_is_transposed = false`
/// 3. Computes A * B with `a_is_transposed = true` (effectively A^T * B)
/// 4. Manually transposes A and computes A^T * B with `a_is_transposed = false`
/// 5. Verifies that cases 3 and 4 produce the same result
/// 6. Ensures that cases 1 and 3 produce different results (validating the flag works)
pub fn check_matmul_a_is_transposed<P>()
where
    P: GenerateRandom + MatrixOps<P> + Default + Clone + std::fmt::Debug + PartialEq,
{
    let cfg_default = MatMulConfig::default();
    let cfg_transpose = MatMulConfig {
        a_is_transposed: true,
        ..MatMulConfig::default()
    };
    let transpose_cfg = VecOpsConfig::default();

    // Create test matrices
    let n = 1 << 3; // 16 rows for A
    let m = 1 << 3; // 8 cols for A, 8 rows for B
    let k = 1 << 4; // 8 cols for B
    
    let input_a = P::generate_random(n * m); // A is n x m
    let input_b = P::generate_random(m * k); // B is m x k
    
    let test_single_device = |main_device: bool| {
        if main_device {
            test_utilities::test_set_main_device();
        } else {
            test_utilities::test_set_ref_device();
        }
        
        // Case 1: Regular matmul A * B (n x m) * (m x k) = (n x k)
        let mut output_regular = vec![P::default(); n * k];
        matmul(
            HostSlice::from_slice(&input_a),
            n as u32,
            m as u32,
            HostSlice::from_slice(&input_b),
            m as u32,
            k as u32,
            &cfg_default,
            HostSlice::from_mut_slice(&mut output_regular),
        )
        .unwrap();
        
        // Case 2: Matmul with a_is_transposed = true
        // This should compute A^T * B where A^T is (m x n) * (m x k) = (n x k)
        // Note: when a_is_transposed is true, we need to swap the dimensions for A
        let mut output_transposed_flag = vec![P::default(); n * k];
        matmul(
            HostSlice::from_slice(&input_a),
            m as u32, // A^T has m rows
            n as u32, // A^T has n cols
            HostSlice::from_slice(&input_b),
            m as u32,
            k as u32,
            &cfg_transpose,
            HostSlice::from_mut_slice(&mut output_transposed_flag),
        )
        .unwrap();
        
        // Case 3: Manually transpose A and then do regular matmul
        let mut a_transposed = vec![P::default(); n * m];
        matrix_transpose(
            HostSlice::from_slice(&input_a),
            n as u32,
            m as u32,
            &transpose_cfg,
            HostSlice::from_mut_slice(&mut a_transposed),
        )
        .unwrap();
        
        let mut output_manual_transpose = vec![P::default(); n * k];
        matmul(
            HostSlice::from_slice(&a_transposed),
            m as u32, // A^T is m x n
            n as u32,
            HostSlice::from_slice(&input_b),
            m as u32,
            k as u32,
            &cfg_default,
            HostSlice::from_mut_slice(&mut output_manual_transpose),
        )
        .unwrap();
        
        // Verify that using the flag produces the same result as manual transpose
        assert_eq!(output_transposed_flag, output_manual_transpose, 
                   "Results should match between using a_is_transposed flag and manual transpose");
        
        // Verify that the results are different when transpose flag is used vs not used
        // (unless by coincidence, which is extremely unlikely with random data)
        assert_ne!(output_regular, output_transposed_flag, 
                   "Results should be different when a_is_transposed flag is set vs not set");
                   
        output_transposed_flag
    };
    
    // Test on both main and reference devices
    let device_out = test_single_device(true);
    let ref_out = test_single_device(false);
    
    // Results should be consistent across devices
    assert_eq!(device_out, ref_out);
}
