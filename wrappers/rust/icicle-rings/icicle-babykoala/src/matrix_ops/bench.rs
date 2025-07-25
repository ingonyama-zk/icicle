use criterion::{criterion_group, criterion_main, Criterion};
use icicle_babykoala::polynomial_ring;

use icicle_core::{matrix_ops::*, polynomial_ring::PolynomialRing, traits::GenerateRandom};

use icicle_runtime::memory::{IntoIcicleSlice, IntoIcicleSliceMut};
use icicle_runtime::test_utilities;

static DEVICES: [&str; 2] = ["ref", "main"];
// static devices: [&str; 1] = ["ref"];

fn x_r_n_r_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "256_n_r_battery_N={}_R={}_device={}"
        };
    }
    for log_n in 15..20 {
        for log_r in 8..10 {
            // generate inputs
            let n = 1 << log_n;
            let r = 1 << log_r;

            let out_size = r * r;
            let input_a = P::generate_random(r * n);
            let input_b = P::generate_random(n * r);
            let mut output_host = vec![P::zero(); out_size];
            let cfg = MatMulConfig::default();

            for device_id in DEVICES {
                // set device

                if device_id == "ref" {
                    test_utilities::test_set_ref_device();
                } else {
                    test_utilities::test_set_main_device();
                }

                // generate instances
                c.bench_function(&format!(testid!(), n, r, device_id), |b| {
                    b.iter(|| {
                        P::matmul(
                            input_a.into_slice(),
                            n as u32,
                            n as u32,
                            input_b.into_slice(),
                            n as u32,
                            n as u32,
                            &cfg,
                            output_host.into_slice_mut(),
                        )
                    })
                });
            }
        }
    }
}

fn x256_n_r_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "256_n_r_battery_N={}_R={}_device={}"
        };
    }
    for log_n in 15..20 {
        for log_r in 8..10 {
            // generate inputs
            let n = 1 << log_n;
            let r = 1 << log_r;

            let out_size = 256 * r;
            let input_a = P::generate_random(256 * n);
            let input_b = P::generate_random(n * r);
            let mut output_host = vec![P::zero(); out_size];
            let cfg = MatMulConfig::default();

            for device_id in DEVICES {
                // set device

                if device_id == "ref" {
                    test_utilities::test_set_ref_device();
                } else {
                    test_utilities::test_set_main_device();
                }

                // generate instances
                c.bench_function(&format!(testid!(), n, r, device_id), |b| {
                    b.iter(|| {
                        P::matmul(
                            input_a.into_slice(),
                            n as u32,
                            n as u32,
                            input_b.into_slice(),
                            n as u32,
                            n as u32,
                            &cfg,
                            output_host.into_slice_mut(),
                        )
                    })
                });
            }
        }
    }
}

fn x256_n_1_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "256_n_1_battery_N={}_device={}"
        };
    }

    for log_n in 15..20 {
        // generate inputs
        let n = 1 << log_n;

        let out_size = n;
        let input_a = P::generate_random(256 * n);
        let input_b = P::generate_random(n);
        let mut output_host = vec![P::zero(); out_size];
        let cfg = MatMulConfig::default();

        for device_id in DEVICES {
            // set device

            if device_id == "ref" {
                test_utilities::test_set_ref_device();
            } else {
                test_utilities::test_set_main_device();
            }

            // generate instances
            c.bench_function(&format!(testid!(), n, device_id), |b| {
                b.iter(|| {
                    P::matmul(
                        input_a.into_slice(),
                        n as u32,
                        n as u32,
                        input_b.into_slice(),
                        n as u32,
                        n as u32,
                        &cfg,
                        output_host.into_slice_mut(),
                    )
                })
            });
        }
    }
}

fn square_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "square_battery_N={}_device={}"
        };
    }
    for log_n in 15..20 {
        // generate inputs
        let n = 1 << log_n;

        let out_size = n * n;
        let input_a = P::generate_random(n * n);
        let input_b = P::generate_random(n * n);
        let mut output_host = vec![P::zero(); out_size];
        let cfg = MatMulConfig::default();

        for device_id in DEVICES {
            // set device

            if device_id == "ref" {
                test_utilities::test_set_ref_device();
            } else {
                test_utilities::test_set_main_device();
            }

            // generate instances
            c.bench_function(&format!(testid!(), n, device_id), |b| {
                b.iter(|| {
                    P::matmul(
                        input_a.into_slice(),
                        n as u32,
                        n as u32,
                        input_b.into_slice(),
                        n as u32,
                        n as u32,
                        &cfg,
                        output_host.into_slice_mut(),
                    )
                })
            });
        }
    }
}

criterion_group! {
    name = matmul_benches;
    config = Criterion::default().significance_level(0.05).sample_size(10);
    targets = square_battery<polynomial_ring::PolyRing>, x256_n_1_battery<polynomial_ring::PolyRing>, x256_n_r_battery<polynomial_ring::PolyRing>, x_r_n_r_battery<polynomial_ring::PolyRing>
}
criterion_main!(matmul_benches);
