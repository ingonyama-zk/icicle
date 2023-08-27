use std::time::Instant;

use icicle_utils::{curves::bls12_381::ScalarField_BLS12_381, test_bls12_381::*};
use rustacuda::prelude::DeviceBuffer;

const LOG_NTT_SIZES: [usize; 3] = [20, 10, 9];
const BATCH_SIZES: [usize; 3] = [1, 1 << 9, 1 << 10];

const MAX_POINTS_LOG2: usize = 18;
const MAX_SCALARS_LOG2: usize = 26;

fn bench_lde() {
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;

            fn ntt_scalars_batch_bls12_381(
                d_inout: &mut DeviceBuffer<ScalarField_BLS12_381>,
                d_twiddles: &mut DeviceBuffer<ScalarField_BLS12_381>,
                batch_size: usize,
            ) -> i32 {
                ntt_inplace_batch_bls12_381(d_inout, d_twiddles, batch_size, false, 0);
                0
            }

            fn intt_scalars_batch_bls12_381(
                d_inout: &mut DeviceBuffer<ScalarField_BLS12_381>,
                d_twiddles: &mut DeviceBuffer<ScalarField_BLS12_381>,
                batch_size: usize,
            ) -> i32 {
                ntt_inplace_batch_bls12_381(d_inout, d_twiddles, batch_size, true, 0);
                0
            }

            // copy
            bench_ntt_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                evaluate_scalars_batch_bls12_381,
                "NTT",
                false,
                100,
            );

            bench_ntt_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                interpolate_scalars_batch_bls12_381,
                "iNTT",
                true,
                100,
            );

            bench_ntt_template(
                MAX_POINTS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_points_bls12_381,
                evaluate_points_batch_bls12_381,
                "EC NTT",
                false,
                20,
            );

            bench_ntt_template(
                MAX_POINTS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_points_bls12_381,
                interpolate_points_batch_bls12_381,
                "EC iNTT",
                true,
                20,
            );

            // inplace
            bench_ntt_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                ntt_scalars_batch_bls12_381,
                "NTT inplace",
                false,
                100,
            );

            bench_ntt_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                intt_scalars_batch_bls12_381,
                "iNTT inplace",
                true,
                100,
            );
        }
    }
}

fn bench_ntt_template<E, S, R>(
    log_max_size: usize,
    ntt_size: usize,
    batch_size: usize,
    log_ntt_size: usize,
    set_data: fn(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<E>, DeviceBuffer<E>, DeviceBuffer<S>),
    bench_fn: fn(d_evaluations: &mut DeviceBuffer<E>, d_domain: &mut DeviceBuffer<S>, batch_size: usize) -> R,
    id: &str,
    inverse: bool,
    samples: usize,
) -> Option<(Vec<E>, R)> {
    let count = ntt_size * batch_size;

    let bench_id = format!("{} of size 2^{} in batch {}", id, log_ntt_size, batch_size);

    if count > 1 << log_max_size {
        println!("Bench size exceeded: {}", bench_id);
        return None;
    }

    println!("{}", bench_id);

    let (input, mut d_evals, mut d_domain) = set_data(ntt_size * batch_size, log_ntt_size, inverse);

    let first = bench_fn(&mut d_evals, &mut d_domain, batch_size);

    let start = Instant::now();
    for _ in 0..samples {
        bench_fn(&mut d_evals, &mut d_domain, batch_size);
    }
    let elapsed = start.elapsed();
    println!(
        "{} {:0?} us x {} = {:?}",
        bench_id,
        elapsed.as_micros() as f32 / (samples as f32),
        samples,
        elapsed
    );

    Some((input, first))
}

fn main() {
    bench_lde();
}
