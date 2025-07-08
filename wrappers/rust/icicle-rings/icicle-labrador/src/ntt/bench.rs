use criterion::{criterion_group, criterion_main, Criterion};
use icicle_labrador::polynomial_ring;

use icicle_core::{
    negacyclic_ntt::*,
    ntt::NTTDir,
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
};

use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    test_utilities,
};

static devices: [&str; 2] = ["main", "ref"];
//static devices: [&str; 1] = ["ref"];
//static devices: [&str; 1] = ["main"];

const ON_DEVICE: bool = true;

/// Initialize test devices and set the main device for benchmarking
fn initialize() {
    test_utilities::test_load_and_init_devices();
    test_utilities::test_set_main_device();
}

/// Benchmark NTT operations for given size on both reference and main devices
fn benchmark_ntt<P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>>(
    c: &mut Criterion,
    test_id: &str,
    n: usize,
    size: usize,
    dir: NTTDir,
) where
    P::Base: FieldImpl,
{
    let test_name = test_id
        .replace("N={}", &format!("N={}", n))
        .replace("_device={}", "");

    let cfg = NegacyclicNttConfig::default();

    // Generate input data
    let input_polys = P::generate_random(size);
    let mut output_polys = vec![P::zero(); size];

    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }

        if ON_DEVICE {
            // Allocate device memory and copy data OUTSIDE the benchmark timing
            let mut input_device = DeviceVec::<P>::device_malloc(size).unwrap();
            let mut output_device = DeviceVec::<P>::device_malloc(size).unwrap();

            input_device
                .copy_from_host(HostSlice::from_slice(&input_polys))
                .unwrap();

            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Only benchmark the actual computation
                    ntt(&input_device, dir, &cfg, &mut output_device)
                })
            });
        } else {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    ntt(
                        HostSlice::from_slice(&input_polys),
                        dir,
                        &cfg,
                        HostSlice::from_mut_slice(&mut output_polys),
                    )
                })
            });
        }
    }
}

/// Generate benchmark functions that iterate over N for both NTT directions
macro_rules! bench_ntt {
    ($f:ident, $id:literal, $n_range:expr, $dir:expr) => {
        fn $f<P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>>(c: &mut Criterion) 
        where 
            P::Base: FieldImpl,
        {
            initialize();
            for log_n in $n_range {
                let N = 1 << log_n;
                benchmark_ntt::<P>(c, $id, N, N, $dir);
            }
        }
    };
}

// Forward NTT benchmark for log_n from 10 to 20
bench_ntt!(
    ntt_forward_battery,
    "ntt_forward_battery_N={}_device={}",
    10..21,
    NTTDir::kForward
);

// Inverse NTT benchmark for log_n from 10 to 20
bench_ntt!(
    ntt_inverse_battery,
    "ntt_inverse_battery_N={}_device={}",
    10..21,
    NTTDir::kInverse
);

// Combined benchmark that tests both directions
fn ntt_both_directions<P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>>(c: &mut Criterion) 
where 
    P::Base: FieldImpl,
{
    initialize();
    for log_n in 10..21 {
        let N = 1 << log_n;
        benchmark_ntt::<P>(c, "ntt_forward_N={}_device={}", N, N, NTTDir::kForward);
        benchmark_ntt::<P>(c, "ntt_inverse_N={}_device={}", N, N, NTTDir::kInverse);
    }
}

// Inplace NTT benchmark
fn benchmark_ntt_inplace<P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>>(
    c: &mut Criterion,
    test_id: &str,
    n: usize,
    size: usize,
    dir: NTTDir,
) where
    P::Base: FieldImpl,
{
    let test_name = test_id
        .replace("N={}", &format!("N={}", n))
        .replace("_device={}", "");

    let cfg = NegacyclicNttConfig::default();

    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }

        if ON_DEVICE {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // NOTE: For inplace operations, we must allocate and copy fresh data each iteration
                    // since the operation modifies the input data
                    let input_polys = P::generate_random(size);
                    let mut input_device = DeviceVec::<P>::device_malloc(size).unwrap();
                    input_device
                        .copy_from_host(HostSlice::from_slice(&input_polys))
                        .unwrap();

                    ntt_inplace(&mut input_device, dir, &cfg)
                })
            });
        } else {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    let input_polys = P::generate_random(size);
                    let mut input_host = input_polys.clone();
                    ntt_inplace(HostSlice::from_mut_slice(&mut input_host), dir, &cfg)
                })
            });
        }
    }
}

// Inplace NTT benchmarks
fn ntt_inplace_forward<P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>>(c: &mut Criterion) 
where 
    P::Base: FieldImpl,
{
    initialize();
    for log_n in 10..21 {
        let N = 1 << log_n;
        benchmark_ntt_inplace::<P>(c, "ntt_inplace_forward_N={}_device={}", N, N, NTTDir::kForward);
    }
}

fn ntt_inplace_inverse<P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>>(c: &mut Criterion) 
where 
    P::Base: FieldImpl,
{
    initialize();
    for log_n in 10..21 {
        let N = 1 << log_n;
        benchmark_ntt_inplace::<P>(c, "ntt_inplace_inverse_N={}_device={}", N, N, NTTDir::kInverse);
    }
}

criterion_group! { name = ntt_benches; config = Criterion::default().significance_level(0.05).sample_size(10);
    targets = ntt_forward_battery<polynomial_ring::PolyRing>, ntt_inverse_battery<polynomial_ring::PolyRing> //, ntt_both_directions<polynomial_ring::PolyRing>, ntt_inplace_forward<polynomial_ring::PolyRing>, ntt_inplace_inverse<polynomial_ring::PolyRing>
    //targets = ntt_forward_battery<polynomial_ring::PolyRing>
}
criterion_main!(ntt_benches); 