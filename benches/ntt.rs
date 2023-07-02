extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use icicle_utils::test_bls12_381::{interpolate_scalars_batch_bls12_381, interpolate_points_batch_bls12_381, set_up_scalars_bls12_381, set_up_points_bls12_381, evaluate_scalars_batch_bls12_381, evaluate_points_batch_bls12_381};


const LOG_NTT_SIZES: [usize; 3] = [20, 9, 10];
const BATCH_SIZES: [usize; 3] = [1, 512, 1024];

fn bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT");
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;

            if ntt_size * batch_size > 1 << 25 {
                continue;
            }

            let (_, mut d_evals, mut d_domain) = set_up_scalars_bls12_381(ntt_size * batch_size, log_ntt_size, true);

            group.sample_size(1000).bench_function(
                &format!("Scalar NTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| evaluate_scalars_batch_bls12_381(&mut d_evals, &mut d_domain, batch_size))
            );
            
            group.sample_size(1000).bench_function(
                &format!("Scalar iNTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| interpolate_scalars_batch_bls12_381(&mut d_evals, &mut d_domain, batch_size))
            );

            if ntt_size * batch_size > 1 << 18{
                continue;
            }

            let (_, mut d_points_evals, _) = set_up_points_bls12_381(ntt_size * batch_size, log_ntt_size, true);
            
            group.sample_size(10).bench_function(
                &format!("EC NTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| interpolate_points_batch_bls12_381(&mut d_points_evals, &mut d_domain, batch_size))
            );

            group.sample_size(10).bench_function(
                &format!("EC iNTT of size 2^{} in batch {}", log_ntt_size, batch_size),
                |b| b.iter(|| evaluate_points_batch_bls12_381(&mut d_points_evals, &mut d_domain, batch_size))
            );
        }
    }
}

criterion_group!(ntt_benches, bench_ntt);
criterion_main!(ntt_benches);

