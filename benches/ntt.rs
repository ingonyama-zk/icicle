extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use icicle_utils::from_cuda::bls12_381::{interpolate_scalars_batch, interpolate_points_batch, set_up_scalars, set_up_points};


const LOG_NTT_SIZES: [usize; 1] = [15];
const BATCH_SIZES: [usize; 2] = [8, 16];

fn bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT");
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;
            let (_, mut d_evals, mut d_domain) = set_up_scalars(ntt_size * batch_size, log_ntt_size, true);
            let (_, mut d_points_evals, _) = set_up_points(ntt_size * batch_size, log_ntt_size, true);

            group.sample_size(100).bench_function(
                &format!("Scalar NTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| interpolate_scalars_batch(&mut d_evals, &mut d_domain, batch_size))
            );

            group.sample_size(10).bench_function(
                &format!("EC NTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| interpolate_points_batch(&mut d_points_evals, &mut d_domain, batch_size))
            );
        }
    }
}

criterion_group!(ntt_benches, bench_ntt);
criterion_main!(ntt_benches);

