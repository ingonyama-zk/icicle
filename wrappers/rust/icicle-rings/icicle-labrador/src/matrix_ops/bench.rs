use criterion::{criterion_group, criterion_main, Criterion};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_labrador::polynomial_ring;
use std::hint::black_box;

use icicle_core::{
    matrix_ops::*,
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
};

use icicle_runtime::{
    memory::{DeviceSlice, HostSlice},
    test_utilities,
};

static devices: [&str; 2] = ["ref", "main"];
// static devices: [&str; 1] = ["ref"];

fn xR_N_R_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "256_n_r_battery_N={}_R={}_device={}"
        };
    }
    for log_n in 15..20 {
        for log_r in 8..10 {
            // generate inputs
            let N = 1 << log_n;
            let R = 1 << log_r;

            let out_size = R * R;
            let input_a = P::generate_random(R * N);
            let input_b = P::generate_random(N * R);
            let mut output_host = vec![P::zero(); out_size];
            let cfg = VecOpsConfig::default();

            for deviceID in devices {
                // set device

                if deviceID == "ref" {
                    test_utilities::test_set_ref_device();
                } else {
                    test_utilities::test_set_main_device();
                }

                // generate instances
                c.bench_function(&format!(testid!(), N, R, deviceID), |b| {
                    b.iter(|| {
                        P::matmul(
                            HostSlice::from_slice(&input_a),
                            N as u32,
                            N as u32,
                            HostSlice::from_slice(&input_b),
                            N as u32,
                            N as u32,
                            &cfg,
                            HostSlice::from_mut_slice(&mut output_host),
                        )
                    })
                });
            }
        }
    }
}

fn x256_n_r_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "256_n_r_battery_N={}_R={}_device={}"
        };
    }
    for log_n in 15..20 {
        for log_r in 8..10 {
            // generate inputs
            let N = 1 << log_n;
            let R = 1 << log_r;

            let out_size = 256 * R;
            let input_a = P::generate_random(256 * N);
            let input_b = P::generate_random(N * R);
            let mut output_host = vec![P::zero(); out_size];
            let cfg = VecOpsConfig::default();

            for deviceID in devices {
                // set device

                if deviceID == "ref" {
                    test_utilities::test_set_ref_device();
                } else {
                    test_utilities::test_set_main_device();
                }

                // generate instances
                c.bench_function(&format!(testid!(), N, R, deviceID), |b| {
                    b.iter(|| {
                        P::matmul(
                            HostSlice::from_slice(&input_a),
                            N as u32,
                            N as u32,
                            HostSlice::from_slice(&input_b),
                            N as u32,
                            N as u32,
                            &cfg,
                            HostSlice::from_mut_slice(&mut output_host),
                        )
                    })
                });
            }
        }
    }
}

fn x256_n_1_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "256_n_1_battery_N={}_device={}"
        };
    }

    for log_n in 15..20 {
        // generate inputs
        let N = 1 << log_n;

        let out_size = N;
        let input_a = P::generate_random(256 * N);
        let input_b = P::generate_random(N);
        let mut output_host = vec![P::zero(); out_size];
        let cfg = VecOpsConfig::default();

        for deviceID in devices {
            // set device

            if deviceID == "ref" {
                test_utilities::test_set_ref_device();
            } else {
                test_utilities::test_set_main_device();
            }

            // generate instances
            c.bench_function(&format!(testid!(), N, deviceID), |b| {
                b.iter(|| {
                    P::matmul(
                        HostSlice::from_slice(&input_a),
                        N as u32,
                        N as u32,
                        HostSlice::from_slice(&input_b),
                        N as u32,
                        N as u32,
                        &cfg,
                        HostSlice::from_mut_slice(&mut output_host),
                    )
                })
            });
        }
    }
}

fn square_battery<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(c: &mut Criterion) {
    macro_rules! testid {
        () => {
            "square_battery_N={}_device={}"
        };
    }
    for log_n in 15..20 {
        // generate inputs
        let N = 1 << log_n;

        let out_size = N * N;
        let input_a = P::generate_random(N * N);
        let input_b = P::generate_random(N * N);
        let mut output_host = vec![P::zero(); out_size];
        let cfg = VecOpsConfig::default();

        for deviceID in devices {
            // set device

            if deviceID == "ref" {
                test_utilities::test_set_ref_device();
            } else {
                test_utilities::test_set_main_device();
            }

            // generate instances
            c.bench_function(&format!(testid!(), N, deviceID), |b| {
                b.iter(|| {
                    P::matmul(
                        HostSlice::from_slice(&input_a),
                        N as u32,
                        N as u32,
                        HostSlice::from_slice(&input_b),
                        N as u32,
                        N as u32,
                        &cfg,
                        HostSlice::from_mut_slice(&mut output_host),
                    )
                })
            });
        }
    }
}

criterion_group! {
    name = matmul_benches;
    config = Criterion::default().significance_level(0.05).sample_size(10);
    targets = square_battery<polynomial_ring::PolyRing>, x256_n_1_battery<polynomial_ring::PolyRing>, x256_n_r_battery<polynomial_ring::PolyRing>, xR_N_R_battery<polynomial_ring::PolyRing>
}
criterion_main!(matmul_benches);
