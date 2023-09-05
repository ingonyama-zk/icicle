extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion, black_box};

use icicle_utils::test_bn254_pse::{
    commit_batch_bn254,msm_batch_bn254,generate_random_points_bn254, generate_random_scalars_bn254,
};
use icicle_utils::utils::*;
#[cfg(feature = "g2")]
use icicle_utils::{commit_batch_g2, field::ExtensionField};

use rustacuda::prelude::*;

const LOG_MSM_SIZES: [usize; 1] = [12];
const BATCH_SIZES: [usize; 1] = [1];

fn bench_msm(c: &mut Criterion) {
    let seed = None;
    let mut group = c.benchmark_group("MSM");
    for log_msm_size in LOG_MSM_SIZES {
        for batch_size in BATCH_SIZES {
            let msm_size = 1 << log_msm_size;
            let scalars = generate_random_scalars_bn254(msm_size*batch_size, get_rng(seed));
            let points = generate_random_points_bn254(msm_size*batch_size, get_rng(seed));

            println!("len:{:?}",scalars.len());
            println!("batch_size:{}",batch_size);

            group
                .sample_size(100)
                .bench_function(
                    &format!("GPU-ONLY: MSM of size 2^{} in batch {}", log_msm_size, batch_size),
                    |b| b.iter(||black_box(msm_batch_bn254 (&points ,&scalars , batch_size,0))),
                );

            // #[cfg(feature = "g2")]
            // group
            //     .sample_size(10)
            //     .bench_function(
            //         &format!("G2 MSM of size 2^{} in batch {}", log_msm_size, batch_size),
            //         |b| b.iter(|| commit_batch_g2(&mut d_g2_points, &mut d_scalars, batch_size)),
            //     );
        }
    }
}

criterion_group!(msm_benches, bench_msm);
criterion_main!(msm_benches);
