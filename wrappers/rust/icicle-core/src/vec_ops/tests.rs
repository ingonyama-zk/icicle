use crate::traits::GenerateRandom;
use crate::vec_ops::{add_scalars, mul_scalars, sub_scalars, FieldImpl, HostOrDeviceSlice, VecOps, VecOpsConfig};

pub fn check_vec_ops_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let test_size = 1 << 14;

    let a = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
    let b = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
    let ones = HostOrDeviceSlice::on_host(vec![F::one(); test_size]);
    let mut result = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);
    let mut result2 = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);
    let mut result3 = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);

    let cfg = VecOpsConfig::default();
    add_scalars(&a, &b, &mut result, &cfg).unwrap();

    sub_scalars(&result, &b, &mut result2, &cfg).unwrap();

    assert_eq!(a[0..1][0], result2[0..1][0]);

    mul_scalars(&a, &ones, &mut result3, &cfg).unwrap();

    assert_eq!(a[0..1][0], result3[0..1][0]);
}
