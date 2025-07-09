use crate::polynomial_ring::PolynomialRing;
use crate::{balanced_decomposition, traits::GenerateRandom, vec_ops::VecOpsConfig};

use icicle_runtime::memory::{DeviceVec, HostSlice};

pub fn check_balanced_decomposition<F>()
where
    F: PolynomialRing + balanced_decomposition::BalancedDecomposition<F> + GenerateRandom,
{
    let batch = 5;
    let size = 1 << 10;
    let total_size = batch * size;
    let bases = [2, 3, 4, 16, 77];

    let input = F::generate_random(total_size as usize);
    let mut recomposed = vec![F::zero(); total_size as usize];

    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = batch as i32;

    for base in bases {
        let digits_per_element = balanced_decomposition::count_digits::<F>(base);
        let mut decomposed = DeviceVec::<F>::device_malloc((total_size * digits_per_element) as usize).unwrap();

        balanced_decomposition::decompose::<F>(HostSlice::from_slice(&input), &mut decomposed[..], base, &cfg).unwrap();
        // In C++ tests we also check here that the digits are in the correct range. Skipping this check here.
        balanced_decomposition::recompose::<F>(&decomposed[..], HostSlice::from_mut_slice(&mut recomposed), base, &cfg)
            .unwrap();
        assert_eq!(input, recomposed);
    }
}
