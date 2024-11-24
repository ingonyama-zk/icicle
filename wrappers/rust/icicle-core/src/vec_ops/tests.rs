use crate::traits::GenerateRandom;
use crate::vec_ops::{
    add_scalars, bit_reverse, bit_reverse_inplace, mul_scalars, sub_scalars, BitReverseConfig, FieldImpl, VecOps,
    VecOpsConfig,
};
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};

use super::accumulate_scalars;

pub fn check_vec_ops_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let test_size = 1 << 14;

    let mut a = F::Config::generate_random(test_size);
    let mut a_clone = a.clone();
    let b = F::Config::generate_random(test_size);
    let ones = vec![F::one(); test_size];
    let mut result = vec![F::zero(); test_size];
    let mut result2 = vec![F::zero(); test_size];
    let mut result3 = vec![F::zero(); test_size];
    let a = HostSlice::from_mut_slice(&mut a);
    let a_clone = HostSlice::from_mut_slice(&mut a_clone);
    let b = HostSlice::from_slice(&b);
    let ones = HostSlice::from_slice(&ones);
    let result = HostSlice::from_mut_slice(&mut result);
    let result2 = HostSlice::from_mut_slice(&mut result2);
    let result3 = HostSlice::from_mut_slice(&mut result3);

    //--- on host ---
    let cfg = VecOpsConfig::default();
    add_scalars(a, b, result, &cfg).unwrap();

    sub_scalars(result, b, result2, &cfg).unwrap();

    assert_eq!(a[1], result2[1]);

    mul_scalars(a, ones, result3, &cfg).unwrap();

    assert_eq!(a[1], result3[1]);

    add_scalars(a, b, result, &cfg).unwrap(); // on host

    accumulate_scalars(a, b, &cfg).unwrap(); // on host in-place

    assert_eq!(a[1], result[1]);

    //--- on device ---
    let cfg = VecOpsConfig::default();
    let mut a_device = DeviceVec::cuda_malloc(test_size).unwrap();

    a_device
        .copy_from_host(a_clone)
        .unwrap();

    let mut b_device = DeviceVec::cuda_malloc(test_size).unwrap();
    b_device
        .copy_from_host(b)
        .unwrap();

    let mut result4_on_device = DeviceVec::cuda_malloc(test_size).unwrap();

    add_scalars(&a_device[..], &b_device[..], &mut result4_on_device[..], &cfg).unwrap(); // on device

    let mut result4_from_device = vec![F::zero(); test_size];
    result4_on_device
        .copy_to_host(HostSlice::from_mut_slice(&mut result4_from_device[..]))
        .unwrap();

    assert_eq!(result[1], result4_from_device[1]);

    let cfg = VecOpsConfig::default();
    let mut a_device = DeviceVec::cuda_malloc(test_size).unwrap();

    a_device
        .copy_from_host(a_clone)
        .unwrap();

    let mut b_device = DeviceVec::cuda_malloc(test_size).unwrap();
    b_device
        .copy_from_host(b)
        .unwrap();

    accumulate_scalars(&mut a_device[..], &b_device[..], &cfg).unwrap(); // on device in-place

    let mut a_from_device = vec![F::zero(); test_size];
    a_device
        .copy_to_host(HostSlice::from_mut_slice(&mut a_from_device[..]))
        .unwrap();

    assert_eq!(a[1], a_from_device[1]);
}

pub fn check_bit_reverse<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    const LOG_SIZE: u32 = 20;
    const TEST_SIZE: usize = 1 << LOG_SIZE;
    let input_vec = F::Config::generate_random(TEST_SIZE);
    let input = HostSlice::from_slice(&input_vec);
    let mut intermediate = DeviceVec::<F>::cuda_malloc(TEST_SIZE).unwrap();
    let cfg = BitReverseConfig::default();
    bit_reverse(input, &cfg, &mut intermediate[..]).unwrap();

    let mut intermediate_host = [F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut intermediate_host[..]))
        .unwrap();
    let index_reverser = |i: usize| i.reverse_bits() >> (usize::BITS - LOG_SIZE);
    intermediate_host
        .iter()
        .enumerate()
        .for_each(|(i, val)| assert_eq!(val, &input_vec[index_reverser(i)]));

    let mut result = vec![F::one(); TEST_SIZE];
    let result = HostSlice::from_mut_slice(&mut result);
    let cfg = BitReverseConfig::default();
    bit_reverse(&intermediate[..], &cfg, result).unwrap();
    assert_eq!(input.as_slice(), result.as_slice());
}

pub fn check_bit_reverse_inplace<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    const LOG_SIZE: u32 = 20;
    const TEST_SIZE: usize = 1 << LOG_SIZE;
    let input_vec = F::Config::generate_random(TEST_SIZE);
    let input = HostSlice::from_slice(&input_vec);
    let mut intermediate = DeviceVec::<F>::cuda_malloc(TEST_SIZE).unwrap();
    intermediate
        .copy_from_host(input)
        .unwrap();
    let cfg = BitReverseConfig::default();
    bit_reverse_inplace(&mut intermediate[..], &cfg).unwrap();

    let mut intermediate_host = [F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut intermediate_host[..]))
        .unwrap();
    let index_reverser = |i: usize| i.reverse_bits() >> (usize::BITS - LOG_SIZE);
    intermediate_host
        .iter()
        .enumerate()
        .for_each(|(i, val)| assert_eq!(val, &input_vec[index_reverser(i)]));

    bit_reverse_inplace(&mut intermediate[..], &cfg).unwrap();
    let mut result_host = vec![F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut result_host[..]))
        .unwrap();
    assert_eq!(input.as_slice(), result_host.as_slice());
}
