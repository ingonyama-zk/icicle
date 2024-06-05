use crate::traits::GenerateRandom;
use crate::vec_ops::{add_scalars, mul_scalars, sub_scalars, FieldImpl, VecOps, VecOpsConfig};
use icicle_cuda_runtime::memory::HostSlice;

use super::accumulate_scalars;

pub fn check_vec_ops_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let test_size = 1 << 14;

    let mut a = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);
    let ones = vec![F::one(); test_size];
    let mut result = vec![F::zero(); test_size];
    let mut result2 = vec![F::zero(); test_size];
    let mut result3 = vec![F::zero(); test_size];
    let a = HostSlice::from_mut_slice(&mut a);
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

    add_scalars(a, b, result, &cfg).unwrap();
    
    accumulate_scalars(a, b, &cfg).unwrap();

    assert_eq!(a[0], result[0]);
}