use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};

// Using both bn254 and bls12-377 curves
use icicle_bls12_377::curve::{
    CurveCfg as BLS12377CurveCfg, G1Projective as BLS12377G1Projective, ScalarCfg as BLS12377ScalarCfg,
};
use icicle_bn254::curve::{CurveCfg, G1Projective, G2CurveCfg, G2Projective, ScalarCfg};

use clap::Parser;
use icicle_core::{curve::Curve, msm, traits::GenerateRandom};

#[derive(Parser, Debug)]
struct Args {
    /// Lower bound (inclusive) of MSM sizes to run for
    #[arg(short, long, default_value_t = 24)]
    lower_bound_log_size: u8,

    /// Upper bound of MSM sizes to run for
    #[arg(short, long, default_value_t = 24)]
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
    let upper_points0 = CurveCfg::generate_random_affine_points(upper_size);
    let upper_points1 = CurveCfg::generate_random_affine_points(upper_size);
    let g2_upper_points = G2CurveCfg::generate_random_affine_points(upper_size);
    let upper_scalars = ScalarCfg::generate_random(upper_size);
    let upper_scalars0 = ScalarCfg::generate_random(upper_size);
    let upper_scalars1 = ScalarCfg::generate_random(upper_size);

    println!("Generating random inputs on host for bls12377...");
    let upper_points_bls12377 = BLS12377CurveCfg::generate_random_affine_points(upper_size);
    let upper_scalars_bls12377 = BLS12377ScalarCfg::generate_random(upper_size);

    for i in lower_bound..=upper_bound {
        let log_size = i;
        let size = 1 << log_size;
        println!(
            "---------------------- MSM size 2^{} = {} ------------------------",
            log_size, size
        );

        // Setting Bn254 points and scalars
        let points = HostSlice::from_slice(&upper_points[..size]);
        let points0 = HostSlice::from_slice(&upper_points0[..size]);
        let points1 = HostSlice::from_slice(&upper_points1[..size]);
        let g2_points = HostSlice::from_slice(&g2_upper_points[..size]);
        let scalars = HostSlice::from_slice(&upper_scalars[..size]);
        let scalars0 = HostSlice::from_slice(&upper_scalars0[..size]);
        let scalars1 = HostSlice::from_slice(&upper_scalars1[..size]);

        // Setting bls12377 points and scalars
        let points_bls12377 = HostSlice::from_slice(&upper_points_bls12377[..size]);
        let scalars_bls12377 = HostSlice::from_slice(&upper_scalars_bls12377[..size]);

        println!("Configuring bn254 MSM...");
        let mut msm_results0 = DeviceVec::<G1Projective>::device_malloc(1).unwrap();
        let mut msm_results1 = DeviceVec::<G1Projective>::device_malloc(1).unwrap();
        let mut g2_msm_results = DeviceVec::<G2Projective>::device_malloc(1).unwrap();
        let mut stream0 = IcicleStream::create().unwrap();
        let mut stream1 = IcicleStream::create().unwrap();
        let mut g2_stream = IcicleStream::create().unwrap();
        let mut cfg0 = msm::MSMConfig::default();
        let mut cfg1 = msm::MSMConfig::default();
        let mut g2_cfg = msm::MSMConfig::default();
        cfg0.stream_handle = *stream0;
        cfg0.is_async = true;
        cfg1.stream_handle = *stream1;
        cfg1.is_async = true;
        g2_cfg.stream_handle = *g2_stream;
        g2_cfg.is_async = true;

        println!("Configuring bls12377 MSM...");
        let mut msm_results_bls12377 = DeviceVec::<BLS12377G1Projective>::device_malloc(1).unwrap();
        let mut stream_bls12377 = IcicleStream::create().unwrap();
        let mut cfg_bls12377 = msm::MSMConfig::default();
        cfg_bls12377.stream_handle = *stream_bls12377;
        cfg_bls12377.is_async = true;

        println!("Executing bn254 MSM on device...");
        println!("G1 0");
        let start_time0 = std::time::Instant::now();
        msm::msm(scalars0, points0, &cfg0, &mut msm_results0[..]).unwrap();
        println!("G1 0 time: {:?}", start_time0.elapsed());
        println!("G1 1");
        let start_time1 = std::time::Instant::now();
        msm::msm(scalars1, points1, &cfg1, &mut msm_results1[..]).unwrap();
        println!("G1 1 time: {:?}", start_time1.elapsed());
        println!("G2");
        let start_time2 = std::time::Instant::now();
        msm::msm(scalars, g2_points, &g2_cfg, &mut g2_msm_results[..]).unwrap();
        println!("G2 time: {:?}", start_time2.elapsed());
        println!("Executing bls12377 MSM on device...");
        msm::msm(
            scalars_bls12377,
            points_bls12377,
            &cfg_bls12377,
            &mut msm_results_bls12377[..],
        )
        .unwrap();

        println!("Moving results to host...");
        let mut msm_host_result0 = vec![G1Projective::zero(); 1];
        let mut msm_host_result1 = vec![G1Projective::zero(); 1];
        let mut g2_msm_host_result = vec![G2Projective::zero(); 1];
        let mut msm_host_result_bls12377 = vec![BLS12377G1Projective::zero(); 1];

        stream0
            .synchronize()
            .unwrap();
        stream1
            .synchronize()
            .unwrap();
        msm_results0
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result0[..]))
            .unwrap();
        msm_results1
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result1[..]))
            .unwrap();
        println!("bn254 result: {:#?}", msm_host_result0);
        println!("bn254 result: {:#?}", msm_host_result1);

        g2_stream
            .synchronize()
            .unwrap();
        g2_msm_results
            .copy_to_host(HostSlice::from_mut_slice(&mut g2_msm_host_result[..]))
            .unwrap();
        println!("G2 bn254 result: {:#?}", g2_msm_host_result);

        stream_bls12377
            .synchronize()
            .unwrap();
        msm_results_bls12377
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result_bls12377[..]))
            .unwrap();
        println!("bls12377 result: {:#?}", msm_host_result_bls12377);

        println!("Cleaning up bn254...");
        stream0
            .destroy()
            .unwrap();
        stream1
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
