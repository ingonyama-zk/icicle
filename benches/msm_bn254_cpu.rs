#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use halo2_proofs::{arithmetic::best_multiexp,
                   halo2curves::{bn256::{Bn256, Fr},
                                 group::{{Curve, Group},
                                         ff::Field},
                                 pairing::Engine}
};

use icicle_utils::utils::get_rng;

fn criterion_benchmark(c: &mut Criterion) {
    const MIN_K: u32 = 20;
    const MAX_K: u32 = 23;

    let seed = None;

    let mut group = c.benchmark_group("MSM");
    for k in MIN_K..=MAX_K {
        let n = 1 << k;
        let coeffs = (0..n).map(|_| Fr::random(get_rng(seed))).collect::<Vec<_>>();
        let bases: Vec<_> = (0..n)
            .map(|_| <Bn256 as Engine>::G1::random(get_rng(seed)).to_affine())
            .collect();


        group.bench_function(BenchmarkId::new("CPU", k), |b| {
            b.iter(|| {
                best_multiexp(&coeffs, &bases);
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);