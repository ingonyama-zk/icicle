use crate::negacyclic_ntt::{NegacyclicNtt, NegacyclicNttConfig};
use crate::ntt::NTTDir;
use crate::polynomial_ring::PolynomialRing;
use crate::traits::{FieldImpl, GenerateRandom};
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice};

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
    let mut output = vec![P::zero(); size];
    let mut roundtrip = vec![P::zero(); size];

    let input_slice = HostSlice::from_slice(&input);
    let output_slice = HostSlice::from_mut_slice(&mut output);
    let roundtrip_slice = HostSlice::from_mut_slice(&mut roundtrip);

    P::ntt(input_slice, NTTDir::kForward, &cfg, output_slice).unwrap();
    P::ntt(output_slice, NTTDir::kInverse, &cfg, roundtrip_slice).unwrap();

    assert_ne!(input, output);
    assert_eq!(input, roundtrip);
}
