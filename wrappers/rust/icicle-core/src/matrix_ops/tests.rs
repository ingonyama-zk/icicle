use crate::{
    matrix_ops::{matmul, MatrixOps, VecOpsConfig},
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
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
    P::Base: FieldImpl,
    P: GenerateRandom<P>,
{
    let cfg = VecOpsConfig::default();

    let n = 1 << 5;
    let m = 1 << 6;
    let k = 1 << 4;
    let out_size = n * k;
    let input_a = P::generate_random(n * m);
    let input_b = P::generate_random(m * k);

    let mut test_single_device = |main_device: bool| {
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
