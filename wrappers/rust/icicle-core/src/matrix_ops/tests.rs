use crate::field::PrimeField;
use crate::{
    matrix_ops::{matmul, matrix_transpose, MatrixOps, VecOpsConfig},
    polynomial_ring::PolynomialRing,
    traits::GenerateRandom,
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
pub fn check_matmul_device_memory<P: PolynomialRing + MatrixOps<P>>()
where
    P::Base: PrimeField,
    P: GenerateRandom,
{
    let cfg = VecOpsConfig::default();

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
        let mut output_host_case_1 = vec![P::zero(); out_size];
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
        let mut output_host_case_2 = vec![P::zero(); out_size];
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

        let mut output_host_case_3 = vec![P::zero(); out_size];
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
        let mut output_host_case_4 = vec![P::zero(); out_size];
        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut output_host_case_4))
            .unwrap();

        assert_eq!(output_host_case_1, output_host_case_4);

        // case (5) mamtmul mixed memory model for inputs, host memory output
        let mut output_host_case_5 = vec![P::zero(); out_size];
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
pub fn check_matrix_transpose_device_memory<P: PolynomialRing + MatrixOps<P>>()
where
    P::Base: PrimeField,
    P: GenerateRandom,
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
        let mut output_host_case_1 = vec![P::zero(); matrix_size];
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
        let mut output_host_case_2 = vec![P::zero(); matrix_size];
        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut output_host_case_2))
            .unwrap();
        assert_eq!(output_host_case_1, output_host_case_2);

        // --- Case 3: Device → Host ---
        let mut device_mem_input = DeviceVec::<P>::device_malloc(matrix_size).unwrap();
        device_mem_input
            .copy_from_host(HostSlice::from_slice(&input_matrix))
            .unwrap();

        let mut output_host_case_3 = vec![P::zero(); matrix_size];
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

        let mut output_host_case_4_restored = vec![P::zero(); matrix_size];
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
    assert_eq!(device_out, ref_out);
}
