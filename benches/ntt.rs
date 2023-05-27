extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use icicle_utils::test_bls12_381::{interpolate_scalars_batch_bls12_381, interpolate_points_batch_bls12_381, set_up_scalars_bls12_381, set_up_points_bls12_381};


const LOG_NTT_SIZES: [usize; 1] = [15];
const BATCH_SIZES: [usize; 2] = [8, 16];

fn bench_point_ntt(c: &mut Criterion) {
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;
            let (_, mut d_evals, mut d_domain) = set_up_points_bls12_381(ntt_size * batch_size, log_ntt_size, true);

            c.bench_function(
                &format!("EC NTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| interpolate_points_batch_bls12_381(&mut d_evals, &mut d_domain, batch_size))
            );
        }
    }
}

fn bench_scalar_ntt(c: &mut Criterion) {
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;
            let (_, mut d_evals, mut d_domain) = set_up_scalars_bls12_381(ntt_size * batch_size, log_ntt_size, true);

            c.bench_function(
                &format!("Scalar NTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| interpolate_scalars_batch_bls12_381(&mut d_evals, &mut d_domain, batch_size))
            );
        }
    }
}

criterion_group!(ntt_benches, bench_point_ntt, bench_scalar_ntt);
criterion_main!(ntt_benches);
