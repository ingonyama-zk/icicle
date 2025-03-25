use crate::{
    decomposition::BalancedDecomposition,
    traits::{FieldImpl, GenerateRandom},
    vec_ops::VecOpsConfig,
};

use icicle_runtime::memory::{DeviceVec, HostSlice};

pub fn check_balanced_decomposition<F>()
where
    F: FieldImpl,
    F::Config: BalancedDecomposition<F> + GenerateRandom<F>,
{
    let batch = 5;
    let size = 1 << 10;
    let total_size = batch * size;
    let bases = [2, 3, 4, 16, 77];

    let input = F::Config::generate_random(total_size as usize); // vec![F::zero(); size];
    let mut recomposed = vec![F::zero(); total_size as usize];

    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = batch as i32;

    for base in bases {
        let digits_per_element = F::Config::compute_nof_digits(base);
        let mut decomposed = DeviceVec::<F>::device_malloc((total_size * digits_per_element) as usize).unwrap();

        F::Config::decompose(HostSlice::from_slice(&input), &mut decomposed[..], base, &cfg).unwrap();
        // In C++ tests we also check here that the digits are in the correct range. Skipping this check here.
        F::Config::recompose(&decomposed[..], HostSlice::from_mut_slice(&mut recomposed), base, &cfg).unwrap();
        assert_eq!(input, recomposed);
    }
}
