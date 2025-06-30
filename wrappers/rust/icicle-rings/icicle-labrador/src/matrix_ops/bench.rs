use criterion::{criterion_group, criterion_main, Criterion};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_core::{
    matrix_ops::*,
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_labrador::polynomial_ring;
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    test_utilities,
};
use std::hint::black_box;

static devices: [&str; 2] = ["main", "ref"];
//static devices: [&str; 1] = ["ref"];
//static devices: [&str; 1] = ["main"];

const LOG_N: std::ops::Range<i32> = 15..18;
const LOG_R: std::ops::Range<i32> = 8..10;

const ON_DEVICE: bool = true;

/// Initialize test devices and set the main device for benchmarking
fn initialize() {
    test_utilities::test_load_and_init_devices();
    test_utilities::test_set_main_device();
}

/// Benchmark matrix multiplication for given dimensions on both reference and main devices
fn benchmark<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(
    c: &mut Criterion,
    test_id: &str,
    n: usize,
    r: Option<usize>,
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
) {
    let test_name = match r {
        Some(r_val) => test_id
            .replace("N={}", &format!("N={}", n))
            .replace("R={}", &format!("R={}", r_val)),
        None => test_id.replace("N={}", &format!("N={}", n)),
    }
    .replace("_device={}", "");
    
    let cfg = VecOpsConfig::default();
    
    // Generate input data
    let input_a_host = P::generate_random(rows_a * cols_a);
    let input_b_host = P::generate_random(rows_b * cols_b);
    let mut output_host = vec![P::zero(); rows_a * cols_b];
    
    for device_id in devices {
        if device_id == "ref" {
            test_utilities::test_set_ref_device();
        } else {
            test_utilities::test_set_main_device();
        }
        
        c.bench_function(&format!("{}_device={}", test_name, device_id), |b| {
            b.iter(|| {
                if ON_DEVICE {
                    // Allocate device memory and copy data
                    let mut input_a_device = DeviceVec::<P>::device_malloc(rows_a * cols_a).unwrap();
                    let mut input_b_device = DeviceVec::<P>::device_malloc(rows_b * cols_b).unwrap();
                    let mut output_device = DeviceVec::<P>::device_malloc(rows_a * cols_b).unwrap();
                    
                    input_a_device.copy_from_host(HostSlice::from_slice(&input_a_host)).unwrap();
                    input_b_device.copy_from_host(HostSlice::from_slice(&input_b_host)).unwrap();
                    
                    P::matmul(
                        &input_a_device,
                        rows_a as u32,
                        cols_a as u32,
                        &input_b_device,
                        rows_b as u32,
                        cols_b as u32,
                        &cfg,
                        &mut output_device,
                    )
                } else {
                    P::matmul(
                        HostSlice::from_slice(&input_a_host),
                        rows_a as u32,
                        cols_a as u32,
                        HostSlice::from_slice(&input_b_host),
                        rows_b as u32,
                        cols_b as u32,
                        &cfg,
                        HostSlice::from_mut_slice(&mut output_host),
                    )
                }
            })
        });
    }
}

/// Generate benchmark functions that iterate over N and R
macro_rules! bench_dual {
    ($f:ident, $id:literal, $n_range:expr, $r_range:expr, $ra:tt, $ca:tt, $rb:tt, $cb:tt) => {
        fn $f<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(c: &mut Criterion) {
            initialize();
            for log_n in $n_range {
                for log_r in $r_range {
                    let (N, R) = (1 << log_n, 1 << log_r);
                    benchmark::<P>(c, $id, N, Some(R), bench_dual!(@eval $ra, N, R), bench_dual!(@eval $ca, N, R), bench_dual!(@eval $rb, N, R), bench_dual!(@eval $cb, N, R));
                }
            }
        }
    };
    (@eval N, $n:expr, $r:expr) => { $n };
    (@eval R, $n:expr, $r:expr) => { $r };
    (@eval $lit:literal, $n:expr, $r:expr) => { $lit };
}

/// Generate benchmark functions that iterate over N
macro_rules! bench_single {
    ($f:ident, $id:literal, $n_range:expr, $ra:tt, $ca:tt, $rb:tt, $cb:tt) => {
        fn $f<P: PolynomialRing + MatrixOps<P> + GenerateRandom<P>>(c: &mut Criterion) {
            initialize();
            for log_n in $n_range {
                let N = 1 << log_n;
                benchmark::<P>(c, $id, N, None, bench_single!(@eval $ra, N), bench_single!(@eval $ca, N), bench_single!(@eval $rb, N), bench_single!(@eval $cb, N));
            }
        }
    };
    (@eval N, $n:expr) => { $n };
    (@eval $lit:literal, $n:expr) => { $lit };
}

// Benchmark R×N by N×R matrix multiplication
bench_dual!(
    xR_N_R_battery,
    "R_N_R_battery_N={}_R={}_device={}",
    LOG_N,
    LOG_R,
    R,
    N,
    N,
    R
);
// Benchmark 256×N by N×R matrix multiplication
bench_dual!(
    x256_n_r_battery,
    "256_n_r_battery_N={}_R={}_device={}",
    LOG_N,
    LOG_R,
    256,
    N,
    N,
    R
);
// Benchmark 256×N by N×1 matrix multiplication
bench_single!(x256_n_1_battery, "256_n_1_battery_N={}_device={}", LOG_N, 256, N, N, 1);
// Benchmark N×N by N×N square matrix multiplication
bench_single!(square_battery, "square_battery_N={}_device={}", LOG_N, N, N, N, N);

criterion_group! { name = matmul_benches; config = Criterion::default().significance_level(0.05).sample_size(10); targets =  x256_n_1_battery<polynomial_ring::PolyRing>, x256_n_r_battery<polynomial_ring::PolyRing>, xR_N_R_battery<polynomial_ring::PolyRing>, square_battery<polynomial_ring::PolyRing> }
criterion_main!(matmul_benches);
