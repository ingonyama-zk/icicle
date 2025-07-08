use criterion::{criterion_group, criterion_main, Criterion};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_labrador::polynomial_ring;

use icicle_core::{
    balanced_decomposition::*,
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

/// Benchmark balanced decomposition for given size and base on both reference and main devices
fn benchmark_decompose<P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>>(
    c: &mut Criterion,
    test_id: &str,
    n: usize,
    size: usize,
    base: u32,
) where
    P::Base: FieldImpl,
{
    let test_name = test_id
        .replace("N={}", &format!("N={}", n))
        .replace("T={}", &format!("T={}", base))
        .replace("_device={}", "");

    let cfg = VecOpsConfig::default();

    // Generate input data
    let input_polys = P::generate_random(size);
    let digits_per_element = count_digits::<P>(base);
    let output_size = size * digits_per_element as usize;
    let mut output_polys = vec![P::zero(); output_size];

    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }

        if ON_DEVICE {
            // Allocate device memory and copy data OUTSIDE the benchmark timing
            let mut input_device = DeviceVec::<P>::device_malloc(size).unwrap();
            let mut output_device = DeviceVec::<P>::device_malloc(output_size).unwrap();

            input_device
                .copy_from_host(HostSlice::from_slice(&input_polys))
                .unwrap();

            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Only benchmark the actual computation
                    decompose(&input_device, &mut output_device, base, &cfg)
                })
            });
        } else {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    decompose(
                        HostSlice::from_slice(&input_polys),
                        HostSlice::from_mut_slice(&mut output_polys),
                        base,
                        &cfg,
                    )
                })
            });
        }
    }
}

/// Benchmark balanced recomposition for given size and base on both reference and main devices
fn benchmark_recompose<P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>>(
    c: &mut Criterion,
    test_id: &str,
    n: usize,
    size: usize,
    base: u32,
) where
    P::Base: FieldImpl,
{
    let test_name = test_id
        .replace("N={}", &format!("N={}", n))
        .replace("T={}", &format!("T={}", base))
        .replace("_device={}", "");

    let cfg = VecOpsConfig::default();

    // Generate input data (decomposed form)
    let digits_per_element = count_digits::<P>(base);
    let input_size = size * digits_per_element as usize;
    let input_polys = P::generate_random(input_size);
    let mut output_polys = vec![P::zero(); size];

    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }

        if ON_DEVICE {
            // Allocate device memory and copy data OUTSIDE the benchmark timing
            let mut input_device = DeviceVec::<P>::device_malloc(input_size).unwrap();
            let mut output_device = DeviceVec::<P>::device_malloc(size).unwrap();

            input_device
                .copy_from_host(HostSlice::from_slice(&input_polys))
                .unwrap();

            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Only benchmark the actual computation
                    recompose(&input_device, &mut output_device, base, &cfg)
                })
            });
        } else {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    recompose(
                        HostSlice::from_slice(&input_polys),
                        HostSlice::from_mut_slice(&mut output_polys),
                        base,
                        &cfg,
                    )
                })
            });
        }
    }
}

// Decomposition benchmarks - loop over t values with constant N
fn decompose_battery<P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>>(c: &mut Criterion) 
where 
    P::Base: FieldImpl,
{
    initialize();
    let log_n = 14; // N = 2^14 = 16384
    let N = 1 << log_n;
    
    for base in [2, 4, 6] {
        benchmark_decompose::<P>(c, "decompose_battery_N={}_T={}_device={}", N, N, base);
    }
}

// Recomposition benchmarks - loop over t values with constant N
fn recompose_battery<P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>>(c: &mut Criterion) 
where 
    P::Base: FieldImpl,
{
    initialize();
    let log_n = 14; // N = 2^14 = 16384
    let N = 1 << log_n;
    
    for base in [2, 4, 6] {
        benchmark_recompose::<P>(c, "recompose_battery_N={}_T={}_device={}", N, N, base);
    }
}

// Combined roundtrip benchmark (decompose then recompose)
fn benchmark_roundtrip<P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>>(
    c: &mut Criterion,
    test_id: &str,
    n: usize,
    size: usize,
    base: u32,
) where
    P::Base: FieldImpl,
{
    let test_name = test_id
        .replace("N={}", &format!("N={}", n))
        .replace("T={}", &format!("T={}", base))
        .replace("_device={}", "");

    let cfg = VecOpsConfig::default();

    // Generate input data
    let input_polys = P::generate_random(size);
    let digits_per_element = count_digits::<P>(base);
    let decomposed_size = size * digits_per_element as usize;
    let mut decomposed_polys = vec![P::zero(); decomposed_size];
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
            let mut decomposed_device = DeviceVec::<P>::device_malloc(decomposed_size).unwrap();
            let mut output_device = DeviceVec::<P>::device_malloc(size).unwrap();

            input_device
                .copy_from_host(HostSlice::from_slice(&input_polys))
                .unwrap();

            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Only benchmark the actual computation
                    // Decompose
                    decompose(&input_device, &mut decomposed_device, base, &cfg).unwrap();
                    
                    // Recompose
                    recompose(&decomposed_device, &mut output_device, base, &cfg)
                })
            });
        } else {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Decompose
                    decompose(
                        HostSlice::from_slice(&input_polys),
                        HostSlice::from_mut_slice(&mut decomposed_polys),
                        base,
                        &cfg,
                    ).unwrap();
                    
                    // Recompose
                    recompose(
                        HostSlice::from_slice(&decomposed_polys),
                        HostSlice::from_mut_slice(&mut output_polys),
                        base,
                        &cfg,
                    )
                })
            });
        }
    }
}

// Roundtrip benchmarks - loop over t values with constant N
fn roundtrip_battery<P: PolynomialRing + BalancedDecomposition<P> + GenerateRandom<P>>(c: &mut Criterion) 
where 
    P::Base: FieldImpl,
{
    initialize();
    let log_n = 14; // N = 2^14 = 16384
    let N = 1 << log_n;
    
    for base in [2, 4, 6] {
        benchmark_roundtrip::<P>(c, "roundtrip_battery_N={}_T={}_device={}", N, N, base);
    }
}

criterion_group! { name = balanced_decomposition_benches; config = Criterion::default().significance_level(0.05).sample_size(10);
    //targets = decompose_battery<polynomial_ring::PolyRing>, recompose_battery<polynomial_ring::PolyRing>, roundtrip_battery<polynomial_ring::PolyRing>
    targets = decompose_battery<polynomial_ring::PolyRing>
}
criterion_main!(balanced_decomposition_benches); 