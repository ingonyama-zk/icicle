extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use icicle_utils::{set_up_scalars, generate_random_points, commit_batch, get_rng, field::BaseField};
#[cfg(feature = "g2")]
use icicle_utils::{commit_batch_g2, field::ExtensionField};

use rustacuda::prelude::*;


const LOG_MSM_SIZES: [usize; 1] = [12];
const BATCH_SIZES: [usize; 2] = [128, 256];

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSM");
    for log_msm_size in LOG_MSM_SIZES {
        for batch_size in BATCH_SIZES {
            let msm_size = 1 << log_msm_size;
            let (scalars, _, _) = set_up_scalars(msm_size, 0, false);
            let batch_scalars = vec![scalars; batch_size].concat();
            let mut d_scalars = DeviceBuffer::from_slice(&batch_scalars[..]).unwrap();

            let points = generate_random_points::<BaseField>(msm_size, get_rng(None));
            let batch_points = vec![points; batch_size].concat();
            let mut d_points = DeviceBuffer::from_slice(&batch_points[..]).unwrap();

            #[cfg(feature = "g2")]
            let g2_points = generate_random_points::<ExtensionField>(msm_size, get_rng(None));
            #[cfg(feature = "g2")]
            let g2_batch_points = vec![g2_points; batch_size].concat();
            #[cfg(feature = "g2")]
            let mut d_g2_points = DeviceBuffer::from_slice(&g2_batch_points[..]).unwrap();

            group.sample_size(30).bench_function(
                &format!("MSM of size 2^{} in batch {}", log_msm_size, batch_size),
                |b| b.iter(|| commit_batch(&mut d_points, &mut d_scalars, batch_size))
            );

            #[cfg(feature = "g2")]
            group.sample_size(10).bench_function(
                &format!("G2 MSM of size 2^{} in batch {}", log_msm_size, batch_size),
                |b| b.iter(|| commit_batch_g2(&mut d_g2_points, &mut d_scalars, batch_size))
            );
        }
    }
}

criterion_group!(msm_benches, bench_msm);
criterion_main!(msm_benches);