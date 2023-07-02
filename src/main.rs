// #![allow(warnings, unused)]

use std::time::{Duration, Instant};

use ark_std::{end_timer, start_timer};

use icicle_utils::{
    curves::bls12_381::ScalarField_BLS12_381,
    test_bls12_381::{
        evaluate_points_batch_bls12_381,
        evaluate_scalars_batch_bls12_381,
        generate_random_scalars_bls12_381, get_rng_bls12_381, interpolate_points_batch_bls12_381,
        interpolate_scalars_batch_bls12_381, intt_batch_bls12_381, ntt_batch_bls12_381,
        set_up_points_bls12_381, set_up_scalars_bls12_381,
    },
};
use rustacuda::prelude::DeviceBuffer;

const LOG_NTT_SIZES: [usize; 3] = [20, 10, 9]; //, 23, 9, 10, 11, 12, 18];
const BATCH_SIZES: [usize; 3] = [1, 1 << 9, 1 << 10]; //, 4, 8, 16, 256, 512, 1024, 1 << 14];
                                              // const LOG_NTT_SIZES: [usize; 4] = [10, 12, 23, 24]; //, 23, 9, 10, 11, 12, 18];
                                              // const BATCH_SIZES: [usize; 6] = [1, 2, 4, 16, 256, 1<<16]; //, 4, 8, 16, 256, 512, 1024, 1 << 14];

const MAX_POINTS_LOG2: usize = 18;
const MAX_SCALARS_LOG2: usize = 26;

fn bench_lde() {
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;

            bench_ntt_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                evaluate_scalars_batch_bls12_381,
                "NTT",
                false,
                1000,
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
                1000,
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
        }
    }
}

fn bench_ntt_template<E, S>(
    log_max_size: usize,
    ntt_size: usize,
    batch_size: usize,
    log_ntt_size: usize,
    set_data: fn(
        test_size: usize,
        log_domain_size: usize,
        inverse: bool,
    ) -> (Vec<E>, DeviceBuffer<E>, DeviceBuffer<S>),
    bench_fn: fn(
        d_evaluations: &mut DeviceBuffer<E>,
        d_domain: &mut DeviceBuffer<S>,
        batch_size: usize,
    ) -> DeviceBuffer<E>,
    id: &str,
    inverse: bool,
    samples: usize,
) -> Option<(Vec<E>, DeviceBuffer<E>)> {
    let count = ntt_size * batch_size;

    let bench_id = format!("{} of size 2^{} in batch {}", id, log_ntt_size, batch_size);

    if count > 1 << log_max_size {
        println!("Bench size exceeded: {}", bench_id);
        return None;
    }

    println!("{}", bench_id);

    let (input, mut d_evals, mut d_domain) = set_data(ntt_size * batch_size, log_ntt_size, inverse);
    //range_push!("{}", bench_id);
    let first = bench_fn(&mut d_evals, &mut d_domain, batch_size);
    //start_timer!(bench_id);
    let start = Instant::now();
    for i in 0..samples {
        bench_fn(&mut d_evals, &mut d_domain, batch_size);
    }
    //end_timer!(bench_id);
    let elapsed = start.elapsed();
    println!(
        "{} {:0?} us x {} = {:?}",
        bench_id,
        elapsed.as_micros() as f32 / (samples as f32),
        samples,
        elapsed
    );
    //range_pop!();

    Some((input, first))
}

fn arith_run() {
    use std::str::FromStr;

    let bench_npow = std::env::var("ARITH_BENCH_NPOW").unwrap_or("3".to_string());
    let lg_domain_size = i32::from_str(&bench_npow).unwrap() as u32;

    let blocks = 2048;
    let threads = 128;
    let domain_size = 10usize.pow(lg_domain_size);
    let name = format!("FR ADD 10**{}", lg_domain_size);
    println!("{}", name);

    //bench_add_fr(domain_size, blocks, threads);

    let name = format!("FR MUL 10**{}", lg_domain_size);
    println!("{}", name);
    //bench_mul_fr(domain_size, blocks, threads);
}

fn main() {
    //arith_run();
    bench_lde();
}
