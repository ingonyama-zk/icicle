use icicle_runtime::{
    memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
    stream::IcicleStream,
};

// Using both bn254 and bls12-377 curves
use icicle_bls12_377::curve::{
    ScalarField as BLS12377ScalarField, CurveCfg as BLS12377CurveCfg, G1Projective as BLS12377G1Projective,
};
use icicle_bn254::curve::{ScalarField, CurveCfg, G1Projective, G2CurveCfg, G2Projective};

use clap::Parser;
use icicle_core::{curve::Curve, msm, traits::GenerateRandom};

#[derive(Parser, Debug)]
struct Args {
    /// Lower bound (inclusive) of MSM sizes to run for
    #[arg(short, long, default_value_t = 10)]
    lower_bound_log_size: u8,

    /// Upper bound of MSM sizes to run for
    #[arg(short, long, default_value_t = 10)]
    upper_bound_log_size: u8,

    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

// Load backend and set device
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    let lower_bound = args.lower_bound_log_size;
    let upper_bound = args.upper_bound_log_size;
    println!("Running Icicle Examples: Rust MSM");
    let upper_size = 1 << upper_bound;

    println!("Generating random inputs on host for bn254...");
    let upper_points = CurveCfg::generate_random_affine_points(upper_size);
    let g2_upper_points = G2CurveCfg::generate_random_affine_points(upper_size);
    let upper_scalars = ScalarField::generate_random(upper_size);

    println!("Generating random inputs on host for bls12377...");
    let upper_points_bls12377 = BLS12377CurveCfg::generate_random_affine_points(upper_size);
    let upper_scalars_bls12377 = BLS12377ScalarField::generate_random(upper_size);

    for i in lower_bound..=upper_bound {
        let log_size = i;
        let size = 1 << log_size;
        println!(
            "---------------------- MSM size 2^{} = {} ------------------------",
            log_size, size
        );

        // Setting Bn254 points and scalars
        let points = upper_points[..size].into_slice();
        let g2_points = g2_upper_points[..size].into_slice();
        let scalars = upper_scalars[..size].into_slice();

        // Setting bls12377 points and scalars
        let points_bls12377 = upper_points_bls12377[..size].into_slice();
        let scalars_bls12377 = upper_scalars_bls12377[..size].into_slice();

        println!("Configuring bn254 MSM...");
        let mut msm_results = DeviceVec::<G1Projective>::malloc(1);
        let mut g2_msm_results = DeviceVec::<G2Projective>::malloc(1);
        let mut stream = IcicleStream::create().unwrap();
        let mut g2_stream = IcicleStream::create().unwrap();
        let mut cfg = msm::MSMConfig::default();
        let mut g2_cfg = msm::MSMConfig::default();
        cfg.stream_handle = *stream;
        cfg.is_async = true;
        g2_cfg.stream_handle = *g2_stream;
        g2_cfg.is_async = true;

        println!("Configuring bls12377 MSM...");
        let mut msm_results_bls12377 = DeviceVec::<BLS12377G1Projective>::malloc(1);
        let mut stream_bls12377 = IcicleStream::create().unwrap();
        let mut cfg_bls12377 = msm::MSMConfig::default();
        cfg_bls12377.stream_handle = *stream_bls12377;
        cfg_bls12377.is_async = true;

        println!("Executing bn254 MSM on device...");
        msm::msm(scalars, points, &cfg, msm_results.into_slice_mut()).unwrap();
        msm::msm(scalars, g2_points, &g2_cfg, g2_msm_results.into_slice_mut()).unwrap();

        println!("Executing bls12377 MSM on device...");
        msm::msm(scalars_bls12377, points_bls12377, &cfg_bls12377, msm_results_bls12377.into_slice_mut()).unwrap();

        println!("Moving results to host...");
        stream
            .synchronize()
            .unwrap();
        let msm_host_result = msm_results.to_host_vec();
        println!("bn254 result: {:#?}", msm_host_result);

        g2_stream
            .synchronize()
            .unwrap();
        let g2_msm_host_result = g2_msm_results.to_host_vec();
        println!("G2 bn254 result: {:#?}", g2_msm_host_result);

        stream_bls12377
            .synchronize()
            .unwrap();
        let msm_host_result_bls12377 = msm_results_bls12377.to_host_vec();
        println!("bls12377 result: {:#?}", msm_host_result_bls12377);

        println!("Cleaning up bn254...");
        stream
            .destroy()
            .unwrap();
        g2_stream
            .destroy()
            .unwrap();

        println!("Cleaning up bls12377...");
        stream_bls12377
            .destroy()
            .unwrap();
        println!("");
    }
}
