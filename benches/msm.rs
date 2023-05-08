extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use icicle_utils::{set_up_scalars, generate_random_points, commit_batch, get_rng};
use rustacuda::prelude::*;


const LOG_MSM_SIZES: [usize; 1] = [12];
const BATCH_SIZES: [usize; 2] = [128, 256];

fn bench_msm(c: &mut Criterion) {
    for log_msm_size in LOG_MSM_SIZES {
        for batch_size in BATCH_SIZES {
            let msm_size = 1 << log_msm_size;
            let (scalars, _, _) = set_up_scalars(msm_size, 0, false);
            let batch_scalars = vec![scalars; batch_size].concat();
            let mut d_scalars = DeviceBuffer::from_slice(&batch_scalars[..]).unwrap();
            let points = generate_random_points(msm_size, get_rng(None));
            let batch_points = vec![points; batch_size].concat();
            let mut d_points = DeviceBuffer::from_slice(&batch_points[..]).unwrap();

            c.bench_function(
                &format!("MSM of size 2^{} in batch {}", log_msm_size, batch_size),
                |b| b.iter(|| commit_batch(&mut d_points, &mut d_scalars, batch_size))
            );
        }
    }
}

criterion_group!(msm_benches, bench_msm);
criterion_main!(msm_benches);