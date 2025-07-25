use crate::{
    norm,
    ring::IntegerRing,
    traits::{Arithmetic, GenerateRandom},
    vec_ops::VecOpsConfig,
};

use icicle_runtime::memory::{IntoIcicleSlice, IntoIcicleSliceMut};
use rand::Rng;

pub fn check_norm<T>()
where
    T: IntegerRing,
    T: norm::Norm<T> + GenerateRandom + Arithmetic,
{
    let batch = 5;
    let size = 1 << 10;
    let total_size = batch * size;

    let mut input = vec![T::zero(); total_size];
    let mut input_u128 = vec![0; total_size];
    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = batch as i32;

    let q_mock = 4289678649214369793_u64;
    let sqrt_q_mock = (q_mock as f64).sqrt() as u32;

    for i in 0..total_size {
        let rand_u32 = rand::thread_rng().gen_range(0..=u32::MAX);
        input[i] = T::from(rand_u32 % sqrt_q_mock);
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
        let result = expected.into_slice_mut();
        norm::check_norm_bound(input.into_slice(), norm::NormType::L2, bound, &cfg, result).unwrap();

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
        let result = expected.into_slice_mut();
        norm::check_norm_bound(input.into_slice(), norm::NormType::LInfinity, bound, &cfg, result).unwrap();

        assert!(
            result
                .iter()
                .all(|x| *x),
            "LInfinity norm check failed with bound {}",
            bound
        );
    }
}
