use crate::field::PrimeField;
use crate::{
    norm,
    traits::{Arithmetic, GenerateRandom},
    vec_ops::VecOpsConfig,
};

use icicle_runtime::memory::HostSlice;
use rand::Rng;

pub fn check_norm<F>()
where
    F: PrimeField,
    F: norm::Norm<F> + GenerateRandom + Arithmetic,
{
    let batch = 5;
    let size = 1 << 10;
    let total_size = batch * size;

    let mut input = vec![F::zero(); total_size];
    let mut input_u128 = vec![0; total_size];
    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = batch as i32;

    let q_mock = 4289678649214369793_u64;
    let sqrt_q_mock = (q_mock as f64).sqrt() as u32;

    for i in 0..total_size {
        let rand_u32 = rand::thread_rng().gen_range(0..=u32::MAX);
        input[i] = F::from_u32(rand_u32 % sqrt_q_mock);
        input_u128[i] = (rand_u32 % sqrt_q_mock) as u128;
    }

    // Test L2 norm
    {
        let actual_norm = input_u128
            .iter()
            .map(|x| x * x)
            .sum::<u128>();
        let bound = (actual_norm as f64).sqrt() as u64 + 1;

        let mut expected = vec![false; batch];
        let result = HostSlice::from_mut_slice(&mut expected);
        norm::check_norm_bound(HostSlice::from_slice(&input), norm::NormType::L2, bound, &cfg, result).unwrap();

        assert!(
            result
                .iter()
                .all(|x| *x),
            "L2 norm check failed with bound {}",
            bound
        );
    }

    // Test LInfinity norm
    {
        let bound = (input_u128
            .iter()
            .max()
            .unwrap()
            + 1) as u64;

        let mut expected = vec![false; batch];
        let result = HostSlice::from_mut_slice(&mut expected);
        norm::check_norm_bound(
            HostSlice::from_slice(&input),
            norm::NormType::LInfinity,
            bound,
            &cfg,
            result,
        )
        .unwrap();

        assert!(
            result
                .iter()
                .all(|x| *x),
            "LInfinity norm check failed with bound {}",
            bound
        );
    }
}
