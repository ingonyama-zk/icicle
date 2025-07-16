use crate::{
    matrix_ops::{matmul, matrix_transpose, MatMulConfig, MatrixOps},
    traits::GenerateRandom,
    vec_ops::VecOpsConfig,
};

use icicle_runtime::{
    memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
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
    P: GenerateRandom + MatrixOps<P> + Default + Clone + Copy + std::fmt::Debug + PartialEq,
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
            input_a.into_slice(),
            n as u32,
            m as u32,
            input_b.into_slice(),
            m as u32,
            k as u32,
            &cfg,
            output_host_case_1.into_slice_mut(),
        )
        .unwrap();

        // case (2) matmul host memory inputs -> device memory output
        let mut device_mem_output = DeviceVec::<P>::malloc(out_size);
        matmul(
            input_a.into_slice(),
            n as u32,
            m as u32,
            input_b.into_slice(),
            m as u32,
            k as u32,
            &cfg,
            &mut device_mem_output,
        )
        .unwrap();

        // compare (1) and (2)
        let output_host_case_2 = device_mem_output.to_host_vec();
        assert_eq!(output_host_case_1, output_host_case_2);

        // case (3) matmul device memory inputs, host memory outputs
        /* Allocate inputs on device, and copy from host */
        let device_mem_a = DeviceVec::from_host_slice(input_a.as_slice());
        let device_mem_b = DeviceVec::from_host_slice(input_b.as_slice());

        let mut output_host_case_3 = vec![P::default(); out_size];
        matmul(
            &device_mem_a,
            n as u32,
            m as u32,
            &device_mem_b,
            m as u32,
            k as u32,
            &cfg,
            output_host_case_3.into_slice_mut(),
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
        let output_host_case_4 = device_mem_output.to_host_vec();

        assert_eq!(output_host_case_1, output_host_case_4);

        // case (5) mamtmul mixed memory model for inputs, host memory output
        let mut output_host_case_5 = vec![P::default(); out_size];
        matmul(
            &device_mem_a,
            n as u32,
            m as u32,
            input_b.into_slice(),
            m as u32,
            k as u32,
            &cfg,
            output_host_case_5.into_slice_mut(),
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
    P: Default + GenerateRandom + Copy + Clone + std::fmt::Debug + PartialEq,
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
            input_matrix.into_slice(),
            nof_rows as u32,
            nof_cols as u32,
            &cfg,
            output_host_case_1.into_slice_mut(),
        )
        .unwrap();
        assert_ne!(input_matrix, output_host_case_1); // Transpose should modify data

        // --- Case 2: Host → Device ---
        let mut device_mem_output = DeviceVec::<P>::device_malloc(matrix_size).unwrap();
        matrix_transpose(
            input_matrix.into_slice(),
            nof_rows as u32,
            nof_cols as u32,
            &cfg,
            &mut device_mem_output,
        )
        .unwrap();

        // Compare (1) and (2)
        let output_host_case_2 = device_mem_output.to_host_vec();
        assert_eq!(output_host_case_1, output_host_case_2);

        // --- Case 3: Device → Host ---
        let mut device_mem_input = DeviceVec::<P>::device_malloc(matrix_size).unwrap();
        device_mem_input
            .copy_from_host(input_matrix.into_slice())
            .unwrap();

        let mut output_host_case_3 = vec![P::default(); matrix_size];
        matrix_transpose(
            &device_mem_input,
            nof_rows as u32,
            nof_cols as u32,
            &cfg,
            output_host_case_3.into_slice_mut(),
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

        let output_host_case_4_restored = device_mem_restored.to_host_vec();
        assert_eq!(input_matrix, output_host_case_4_restored);

        output_host_case_1
    };

    // Run on main device
    let device_out = test_single_device(true);

    // Run on reference device
    let ref_out = test_single_device(false);

    // Final check: both devices should yield identical transpose results
    assert_eq!(device_out, ref_out);
}

/// Verifies that `a_transposed` and `b_transposed` flags in matmul behave as expected.
///
/// Test procedure:
/// 1. Compute A * B with no transpose.
/// 2. Compute Aᵗ * Bᵗ using the matmul API with transpose flags.
/// 3. Manually transpose A and B, then compute Aᵗ * Bᵗ using standard matmul.
/// 4. Assert that (2) and (3) produce identical results.
/// 5. Assert that (1) and (2) differ, confirming the transposition takes effect.
pub fn check_matmul_transposed<P>()
where
    P: GenerateRandom + MatrixOps<P> + Default + Clone + std::fmt::Debug + PartialEq,
{
    let cfg_transposed = MatMulConfig {
        a_transposed: true,
        b_transposed: true,
        ..MatMulConfig::default()
    };

    // Create test matrices A (n x m) and B (m x k)
    let n = 1 << 3; // 8
    let m = n;
    let k = n;

    let input_a = P::generate_random(n * m); // A: n x m
    let input_b = P::generate_random(m * k); // B: m x k

    let test_single_device = |main_device: bool| {
        if main_device {
            test_utilities::test_set_main_device();
        } else {
            test_utilities::test_set_ref_device();
        }

        // Case 1: A * B (n×m) × (m×k) = n×k
        let mut output_case_1 = vec![P::default(); n * k];
        matmul(
            input_a.into_slice(),
            n as u32,
            m as u32,
            input_b.into_slice(),
            m as u32,
            k as u32,
            &MatMulConfig::default(),
            output_case_1.into_slice_mut(),
        )
        .unwrap();

        // Case 2: Aᵗ * Bᵗ using transposed flags
        // Aᵗ: m x n, Bᵗ: m x k
        let mut output_case_2 = vec![P::default(); n * k];
        matmul(
            input_a.into_slice(),
            m as u32,
            n as u32,
            input_b.into_slice(),
            m as u32,
            k as u32,
            &cfg_transposed,
            output_case_2.into_slice_mut(),
        )
        .unwrap();

        // Case 3: Manually transpose A and B, then compute Aᵗ * Bᵗ
        let mut a_transposed = vec![P::default(); n * m];
        matrix_transpose(
            input_a.into_slice(),
            n as u32,
            m as u32,
            &VecOpsConfig::default(),
            a_transposed.into_slice_mut(),
        )
        .unwrap();

        let mut b_transposed = vec![P::default(); n * m];
        matrix_transpose(
            input_b.into_slice(),
            n as u32,
            m as u32,
            &VecOpsConfig::default(),
            b_transposed.into_slice_mut(),
        )
        .unwrap();

        let mut output_case_3 = vec![P::default(); n * k];
        matmul(
            a_transposed.into_slice(),
            m as u32,
            n as u32,
            b_transposed.into_slice(),
            m as u32,
            k as u32,
            &MatMulConfig::default(),
            output_case_3.into_slice_mut(),
        )
        .unwrap();

        // Case 2 and 3 should match
        assert_eq!(
            output_case_2, output_case_3,
            "Results should match between using transposed flags and manual transposition"
        );

        // Case 1 and 2 should differ (almost always, given random inputs)
        assert_ne!(
            output_case_1, output_case_2,
            "Results should differ when transposition is applied"
        );

        output_case_2
    };

    // Run test on both main and reference devices
    let device_out = test_single_device(true);
    let ref_out = test_single_device(false);

    // Ensure consistent results across devices
    assert_eq!(device_out, ref_out);
}
