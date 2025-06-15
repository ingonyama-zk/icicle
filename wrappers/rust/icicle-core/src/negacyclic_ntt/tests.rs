use crate::negacyclic_ntt::{NegacyclicNtt, NegacyclicNttConfig};
use crate::ntt::NTTDir;
use crate::{
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    test_utilities,
};

/// Basic roundtrip test for NTT + inverse NTT
pub fn test_negacyclic_ntt_roundtrip<P: PolynomialRing + NegacyclicNtt<P> + Clone + PartialEq + core::fmt::Debug>()
where
    P::Base: FieldImpl,
    P: GenerateRandom<P>,
{
    let cfg = NegacyclicNttConfig::default();

    // Create a test vector with alternating values
    let size = 1 << 10;
    let input = P::generate_random(size);

    let test_single_device = |main_device: bool| {
        if main_device {
            test_utilities::test_set_main_device();
        } else {
            test_utilities::test_set_ref_device();
        }
        // (1) ntt host memory -> host_memory
        let mut output = vec![P::zero(); size];
        P::ntt(
            HostSlice::from_slice(&input),
            NTTDir::kForward,
            &cfg,
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();
        assert_ne!(input, output);

        // (2) ntt host memory -> device memory
        let mut device_mem = DeviceVec::<P>::device_malloc(size).unwrap();
        P::ntt(HostSlice::from_slice(&input), NTTDir::kForward, &cfg, &mut device_mem).unwrap();

        // (3) compare (1) and (2)
        let mut host_buffer = vec![P::zero(); size];
        device_mem
            .copy_to_host(HostSlice::from_mut_slice(&mut host_buffer))
            .unwrap();
        assert_eq!(output, host_buffer);

        // (4) intt device memory -> host memory
        let mut roundtrip = vec![P::zero(); size];
        P::ntt(
            &device_mem,
            NTTDir::kInverse,
            &cfg,
            HostSlice::from_mut_slice(&mut roundtrip),
        )
        .unwrap();
        assert_eq!(input, roundtrip);

        // (5) intt inplace device memory and compare to input
        P::ntt_inplace(&mut device_mem, NTTDir::kInverse, &cfg).unwrap();
        device_mem
            .copy_to_host(HostSlice::from_mut_slice(&mut host_buffer))
            .unwrap();
        assert_eq!(input, host_buffer);

        output
    };

    // Compare main and ref devices
    let device_out = test_single_device(true);
    let ref_out = test_single_device(false);

    assert_eq!(device_out, ref_out);
}
