use crate::traits::GenerateRandom;
use crate::vec_ops::{
    add_scalars, bit_reverse, bit_reverse_inplace, mul_scalars, sub_scalars, BitReverseConfig, FieldImpl, VecOps,
    VecOpsConfig,
};
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};

pub fn check_vec_ops_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let test_size = 1 << 14;

    let a = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);
    let ones = vec![F::one(); test_size];
    let mut result = vec![F::zero(); test_size];
    let mut result2 = vec![F::zero(); test_size];
    let mut result3 = vec![F::zero(); test_size];
    let a = HostSlice::from_slice(&a);
    let b = HostSlice::from_slice(&b);
    let ones = HostSlice::from_slice(&ones);
    let result = HostSlice::from_mut_slice(&mut result);
    let result2 = HostSlice::from_mut_slice(&mut result2);
    let result3 = HostSlice::from_mut_slice(&mut result3);

    let cfg = VecOpsConfig::default();
    add_scalars(a, b, result, &cfg).unwrap();

    sub_scalars(result, b, result2, &cfg).unwrap();

    assert_eq!(a[0], result2[0]);

    mul_scalars(a, ones, result3, &cfg).unwrap();

    assert_eq!(a[0], result3[0]);
}

pub fn check_bit_reverse<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    const TEST_SIZE: usize = 1 << 20;
    let input = F::Config::generate_random(TEST_SIZE);
    let input = HostSlice::from_slice(&input);
    let mut intermediate_result = DeviceVec::<F>::cuda_malloc(TEST_SIZE).unwrap();
    let mut cfg = BitReverseConfig::default();
    cfg.are_outputs_on_device = true;
    bit_reverse(input, &cfg, &mut intermediate_result[..]).unwrap();

    let mut result = vec![F::one(); TEST_SIZE];
    let result = HostSlice::from_mut_slice(&mut result);
    let mut cfg = BitReverseConfig::default();
    cfg.are_outputs_on_device = false;
    cfg.are_inputs_on_device = true;
    bit_reverse(&intermediate_result[..], &cfg, result).unwrap();
    assert_eq!(input.as_slice(), result.as_slice());
}

pub fn check_bit_reverse_inplace<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    const TEST_SIZE: usize = 1 << 20;
    let input = F::Config::generate_random(TEST_SIZE);
    let input = HostSlice::from_slice(&input);
    let mut intermediate = DeviceVec::<F>::cuda_malloc(TEST_SIZE).unwrap();
    intermediate
        .copy_from_host(&input)
        .unwrap();
    let mut cfg = BitReverseConfig::default();
    cfg.are_inputs_on_device = true;
    bit_reverse_inplace(&mut intermediate[..], &cfg).unwrap();
    bit_reverse_inplace(&mut intermediate[..], &cfg).unwrap();
    let mut result_host = vec![F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut result_host[..]))
        .unwrap();
    assert_eq!(input.as_slice(), result_host.as_slice());
}
