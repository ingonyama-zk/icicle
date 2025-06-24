use crate::{
    matrix_ops::*,
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
};

use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostSlice},
    test_utilities,
};

/// Ensure hostmemory and devicememory behaviour matches.
/// Correctness is already ensured by the C++ tests.
pub fn test_matmul_devicememory<P: PolynomialRing + MatrixOps<P>>()
where
    P::Base: FieldImpl,
    P: GenerateRandom<P>,
{
    let cfg = VecOpsConfig::default();

    // Create a test vector with alternating values
    let N = 1 << 10;
    let M = 1 << 8;
    let K = 1 << 9;
    let out_size = N * K;
    let input_a = P::generate_random(N * M);
    let input_b = P::generate_random(M * K);

    let test_single_device = |main_device: bool| {
        if main_device {
            test_utilities::test_set_main_device();
        } else {
            test_utilities::test_set_ref_device();
        }
        // (1) matmul host memory inputs -> host_memory outputs
        let mut output_host = vec![P::zero(); out_size];
        P::matmul(
            HostSlice::from_slice(&input_a),
            N as u32,
            M as u32,
            HostSlice::from_slice(&input_b),
            M as u32,
            K as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host),
        )
        .unwrap();

        // (2) matmul host memory inputs -> device memory output
        let mut output_device = vec![P::zero(); out_size];
        let mut device_mem_output = unsafe { DeviceSlice::<P>::from_mut_slice(&mut output_device) };

        P::matmul(
            HostSlice::from_slice(&input_a),
            N as u32,
            M as u32,
            HostSlice::from_slice(&input_b),
            M as u32,
            K as u32,
            &cfg,
            device_mem_output,
        )
        .unwrap();

        // compare (1) and (2)
        let mut host_buffer = vec![P::zero(); out_size];
        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut host_buffer))
            .unwrap();
        assert_eq!(output_host, host_buffer);

        // (3) matmul device memory inputs, host memory outputs
        let mut device_mem_output = unsafe { DeviceSlice::<P>::from_mut_slice(&mut output_device) };
        let device_mem_a = unsafe { DeviceSlice::<P>::from_slice(&input_a) };
        let device_mem_b = unsafe { DeviceSlice::<P>::from_slice(&input_b) };
        let mut output_host_2 = output_host.clone();
        P::matmul(
            device_mem_a,
            N as u32,
            M as u32,
            device_mem_b,
            M as u32,
            K as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_2),
        )
        .unwrap();

        // compare (1) and (3)
        assert_eq!(output_host, output_host_2);

        // (4) matmul device memory inputs, device memory output
        let mut output_device = vec![P::zero(); out_size];
        let mut device_mem_output = unsafe { DeviceSlice::<P>::from_mut_slice(&mut output_device) };

        device_mem_output
            .copy_to_host(HostSlice::from_mut_slice(&mut host_buffer))
            .unwrap();

        P::matmul(
            HostSlice::from_slice(&input_a),
            N as u32,
            M as u32,
            HostSlice::from_slice(&input_b),
            M as u32,
            K as u32,
            &cfg,
            device_mem_output,
        )
        .unwrap();

        assert_eq!(output_host, host_buffer);
        // (5) mamtmul mixed memory model for inputs, host memory ouput
        let mut device_mem_output = unsafe { DeviceSlice::<P>::from_mut_slice(&mut output_device) };
        let device_mem_a = unsafe { DeviceSlice::<P>::from_slice(&input_a) };
        let mut output_host_3 = output_host.clone();
        P::matmul(
            HostSlice::from_slice(&input_a),
            N as u32,
            M as u32,
            device_mem_b,
            M as u32,
            K as u32,
            &cfg,
            HostSlice::from_mut_slice(&mut output_host_3),
        )
        .unwrap();

        // compare (1) and (5)
        assert_eq!(output_host, output_host_3);

        output_host
    };

    // Compare main and ref devices
    let device_out = test_single_device(true);
    let ref_out = test_single_device(false);

    assert_eq!(device_out, ref_out);
}
