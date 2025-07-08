use criterion::{criterion_group, criterion_main, Criterion};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_labrador::{polynomial_ring, ring};

use icicle_core::{
    jl_projection::*,
    polynomial_ring::PolynomialRing,
    traits::FieldImpl,
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

/// Benchmark JL projection for scalar ring elements
fn benchmark_jl_projection_scalar<T>(
    c: &mut Criterion,
    test_id: &str,
    input_size: usize,
    output_size: usize,
) 
where
    T: FieldImpl,
    T::Config: JLProjection<T>,
{
    let test_name = test_id
        .replace("INPUT_SIZE={}", &format!("INPUT_SIZE={}", input_size))
        .replace("OUTPUT_SIZE={}", &format!("OUTPUT_SIZE={}", output_size))
        .replace("_device={}", "");

    let cfg = VecOpsConfig::default();
    
    // Generate input data and seed
    // Note: JL projection performance doesn't depend on input values, so we use ones
    let input_host = vec![T::one(); input_size];
    let mut output_host = vec![T::zero(); output_size];
    let seed = b"jl_projection_benchmark_seed_12345";

    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }

        if ON_DEVICE {
            // Allocate device memory and copy data OUTSIDE the benchmark timing
            let mut input_device = DeviceVec::<T>::device_malloc(input_size).unwrap();
            let mut output_device = DeviceVec::<T>::device_malloc(output_size).unwrap();

            input_device
                .copy_from_host(HostSlice::from_slice(&input_host))
                .unwrap();

            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Only benchmark the actual computation
                    jl_projection(&input_device, seed, &cfg, &mut output_device).unwrap();
                })
            });
        } else {
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    jl_projection(
                        HostSlice::from_slice(&input_host),
                        seed,
                        &cfg,
                        HostSlice::from_mut_slice(&mut output_host),
                    ).unwrap();
                })
            });
        }
    }
}

/// Benchmark JL matrix row generation for polynomial rings
fn benchmark_jl_matrix_rows_polyring<P>(
    c: &mut Criterion,
    test_id: &str,
    row_size: usize,
    num_rows: usize,
) 
where
    P: PolynomialRing + JLProjectionPolyRing<P>,
{
    let test_name = test_id
        .replace("ROW_SIZE={}", &format!("ROW_SIZE={}", row_size))
        .replace("NUM_ROWS={}", &format!("NUM_ROWS={}", num_rows))
        .replace("_device={}", "");

    let cfg = VecOpsConfig::default();
    let seed = b"jl_projection_benchmark_seed_12345";
    let conjugate = false; // Test without conjugation

    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }

        if ON_DEVICE {
            // Allocate device memory OUTSIDE the benchmark timing
            let mut output_device = DeviceVec::<P>::device_malloc(row_size * num_rows).unwrap();

            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    // Only benchmark the actual computation
                    get_jl_matrix_rows_as_polyring(
                        seed,
                        row_size,
                        0,
                        num_rows,
                        conjugate,
                        &cfg,
                        &mut output_device,
                    ).unwrap();
                })
            });
        } else {
            let mut output_host = vec![P::zero(); row_size * num_rows];
            
            c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                b.iter(|| {
                    get_jl_matrix_rows_as_polyring(
                        seed,
                        row_size,
                        0,
                        num_rows,
                        conjugate,
                        &cfg,
                        HostSlice::from_mut_slice(&mut output_host),
                    ).unwrap();
                })
            });
        }
    }
}

/// Benchmark JL projection for different input sizes (mirroring matmul n=15..25)
fn jl_projection_scalar_battery(c: &mut Criterion) {
    initialize();
    
    for log_n in 15..25 {
        let input_size = 1 << log_n;
        let output_size = 256; // Common JL projection output size
        
        benchmark_jl_projection_scalar::<ring::ScalarRing>(
            c,
            "jl_projection_scalar_INPUT_SIZE={}_OUTPUT_SIZE={}_device={}",
            input_size,
            output_size,
        );
    }
}

/// Benchmark JL matrix row generation for polynomial rings with different row sizes
fn jl_matrix_rows_polyring_battery(c: &mut Criterion) {
    initialize();
    
    for log_n in 15..25 {
        let row_size = 1 << log_n;
        let num_rows = 256; // Common number of rows
        
        benchmark_jl_matrix_rows_polyring::<polynomial_ring::PolyRing>(
            c,
            "jl_matrix_rows_polyring_ROW_SIZE={}_NUM_ROWS={}_device={}",
            row_size,
            num_rows,
        );
    }
}

/// Benchmark JL projection with varying output sizes (typical dimensionality reduction)
fn jl_projection_varying_output_battery(c: &mut Criterion) {
    initialize();
    
    let input_size = 1 << 18; 
    let output_sizes = [128, 256, 512, 1024]; // Different output dimensions
    
    for &output_size in &output_sizes {
        benchmark_jl_projection_scalar::<ring::ScalarRing>(
            c,
            "jl_projection_varying_output_INPUT_SIZE={}_OUTPUT_SIZE={}_device={}",
            input_size,
            output_size,
        );
    }
}

/// Benchmark JL matrix row generation for different row sizes
fn jl_matrix_rows_battery(c: &mut Criterion) {
    initialize();
    
    for log_n in 15..25 {
        let row_size = 1 << log_n;
        let num_rows = 256;
        let start_row = 0;
        
        let test_name = format!("jl_matrix_rows_ROW_SIZE={}_NUM_ROWS={}", row_size, num_rows);
        let cfg = VecOpsConfig::default();
        let seed = b"jl_matrix_rows_benchmark_seed_12345";
        
        for device_id in devices {
            if device_id == "ref" {
                test_utilities::test_set_ref_device();
            } else {
                test_utilities::test_set_main_device();
            }

            if ON_DEVICE {
                let mut output_device = DeviceVec::<ring::ScalarRing>::device_malloc(row_size * num_rows).unwrap();

                c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                    b.iter(|| {
                        get_jl_matrix_rows(seed, row_size, start_row, num_rows, &cfg, &mut output_device).unwrap();
                    })
                });
            } else {
                let mut output_host = vec![ring::ScalarRing::zero(); row_size * num_rows];
                
                c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
                    b.iter(|| {
                        get_jl_matrix_rows(
                            seed,
                            row_size,
                            start_row,
                            num_rows,
                            &cfg,
                            HostSlice::from_mut_slice(&mut output_host),
                        ).unwrap();
                    })
                });
            }
        }
    }
}

criterion_group! { 
    name = jl_projection_benches; 
    config = Criterion::default().significance_level(0.05).sample_size(10);
    //targets = jl_projection_scalar_battery, jl_matrix_rows_polyring_battery, jl_projection_varying_output_battery, jl_matrix_rows_battery
    //targets = jl_matrix_rows_polyring_battery
    targets = jl_projection_varying_output_battery
}

criterion_main!(jl_projection_benches);
