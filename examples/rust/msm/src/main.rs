use icicle_bn254::curve::{
    CurveCfg,
    ScalarCfg,
    G1Projective,
    G2CurveCfg,
    G2Projective 
};

use icicle_bls12_377::curve::{
    CurveCfg as BLS12377CurveCfg,
    ScalarCfg as BLS12377ScalarCfg,
    G1Projective as BLS12377G1Projective
};

use icicle_cuda_runtime::{
    stream::CudaStream,
    memory::HostOrDeviceSlice
};

use icicle_core::{
    msm,
    curve::Curve,
    traits::GenerateRandom
};

#[cfg(feature = "arkworks")]
use icicle_core::traits::ArkConvertible;

#[cfg(feature = "arkworks")]
use ark_bn254::{
    G1Projective as Bn254ArkG1Projective,
    G1Affine as Bn254G1Affine,
    Fr as Bn254Fr
};
#[cfg(feature = "arkworks")]
use ark_bls12_377::{
    G1Projective as Bls12377ArkG1Projective,
    G1Affine as Bls12377G1Affine,
    Fr as Bls12377Fr
};
#[cfg(feature = "arkworks")]
use ark_ec::scalar_mul::variable_base::VariableBaseMSM;

#[cfg(feature = "profile")]
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Lower bound (inclusive) of MSM sizes to run for
    #[arg(short, long, default_value_t = 19)]
    lower_bound_log_size: u8,

    /// Upper bound of MSM sizes to run for
    #[arg(short, long, default_value_t = 23)]
    upper_bound_log_size: u8,
}

fn main() {
    let args = Args::parse();
    let lower_bound = args.lower_bound_log_size;
    let upper_bound = args.upper_bound_log_size;
    println!("Running Icicle Examples: Rust MSM");
    let upper_size = 1 << (upper_bound);
    println!("Generating random inputs on host for bn254...");
    let upper_points = CurveCfg::generate_random_affine_points(upper_size);
    let g2_upper_points = G2CurveCfg::generate_random_affine_points(upper_size);
    let upper_scalars = ScalarCfg::generate_random(upper_size);
    
    println!("Generating random inputs on host for bls12377...");
    let upper_points_bls12377 = BLS12377CurveCfg::generate_random_affine_points(upper_size);
    let upper_scalars_bls12377 = BLS12377ScalarCfg::generate_random(upper_size);

    for i in lower_bound..=upper_bound { 
        let log_size = i;
        let size = 1 << log_size;
        println!("---------------------- MSM size 2^{}={} ------------------------", log_size, size);
        // Setting Bn254 points and scalars
        let points = HostOrDeviceSlice::Host(upper_points[..size].to_vec());
        let g2_points = HostOrDeviceSlice::Host(g2_upper_points[..size].to_vec());
        let scalars = HostOrDeviceSlice::Host(upper_scalars[..size].to_vec());
        
        // Setting bls12377 points and scalars
        // let points_bls12377 = &upper_points_bls12377[..size];
        let points_bls12377 =  HostOrDeviceSlice::Host(upper_points_bls12377[..size].to_vec()); //  &upper_points_bls12377[..size];
        let scalars_bls12377 = HostOrDeviceSlice::Host(upper_scalars_bls12377[..size].to_vec());

        println!("Configuring bn254 MSM...");
        let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        let mut g2_msm_results: HostOrDeviceSlice<'_, G2Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        let stream = CudaStream::create().unwrap();
        let g2_stream = CudaStream::create().unwrap();
        let mut cfg = msm::get_default_msm_config::<CurveCfg>();
        let mut g2_cfg = msm::get_default_msm_config::<G2CurveCfg>();
        cfg.ctx.stream = &stream;
        g2_cfg.ctx.stream = &g2_stream;
        cfg.is_async = true;
        g2_cfg.is_async = true;

        println!("Configuring bls12377 MSM...");
        let mut msm_results_bls12377: HostOrDeviceSlice<'_, BLS12377G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        let stream_bls12377 = CudaStream::create().unwrap();
        let mut cfg_bls12377 = msm::get_default_msm_config::<BLS12377CurveCfg>();
        cfg_bls12377.ctx.stream = &stream_bls12377;
        cfg_bls12377.is_async = true;

        println!("Executing bn254 MSM on device...");
        #[cfg(feature = "profile")]
        let start = Instant::now();
        msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
        #[cfg(feature = "profile")]
        println!("ICICLE BN254 MSM on size 2^{log_size} took: {} ms", start.elapsed().as_millis());
        msm::msm(&scalars, &g2_points, &g2_cfg, &mut g2_msm_results).unwrap();


        println!("Executing bls12377 MSM on device...");
        #[cfg(feature = "profile")]
        let start = Instant::now();
        msm::msm(&scalars_bls12377, &points_bls12377, &cfg_bls12377, &mut msm_results_bls12377 ).unwrap();
        #[cfg(feature = "profile")]
        println!("ICICLE BLS12377 MSM on size 2^{log_size} took: {} ms", start.elapsed().as_millis());

        println!("Moving results to host..");
        let mut msm_host_result = vec![G1Projective::zero(); 1];
        let mut g2_msm_host_result = vec![G2Projective::zero(); 1];
        let mut msm_host_result_bls12377 = vec![BLS12377G1Projective::zero(); 1];
        
        stream
            .synchronize()
            .unwrap();
        g2_stream
            .synchronize()
            .unwrap();
        msm_results
            .copy_to_host(&mut msm_host_result[..])
            .unwrap();
        g2_msm_results
            .copy_to_host(&mut g2_msm_host_result[..])
            .unwrap();
        println!("bn254 result: {:#?}", msm_host_result);
        println!("G2 bn254 result: {:#?}", g2_msm_host_result);
        
        stream_bls12377
            .synchronize()
            .unwrap();
        msm_results_bls12377
            .copy_to_host(&mut msm_host_result_bls12377[..])
            .unwrap();
        println!("bls12377 result: {:#?}", msm_host_result_bls12377);
        
        #[cfg(feature = "arkworks")]
        {
            println!("Checking against arkworks...");
            let ark_points: Vec<Bn254G1Affine> = points.as_slice().iter().map(|&point| point.to_ark()).collect();
            let ark_scalars: Vec<Bn254Fr> = scalars.as_slice().iter().map(|scalar| scalar.to_ark()).collect();

            let ark_points_bls12377: Vec<Bls12377G1Affine> = points_bls12377.as_slice().iter().map(|point| point.to_ark()).collect();
            let ark_scalars_bls12377: Vec<Bls12377Fr> = scalars_bls12377.as_slice().iter().map(|scalar| scalar.to_ark()).collect();

            #[cfg(feature = "profile")]
            let start = Instant::now();
            let bn254_ark_msm_res = Bn254ArkG1Projective::msm(&ark_points, &ark_scalars).unwrap();
            println!("Arkworks Bn254 result: {:#?}", bn254_ark_msm_res);
            #[cfg(feature = "profile")]
            println!("Ark BN254 MSM on size 2^{log_size} took: {} ms", start.elapsed().as_millis());

            #[cfg(feature = "profile")]
            let start = Instant::now();
            let bls12377_ark_msm_res = Bls12377ArkG1Projective::msm(&ark_points_bls12377, &ark_scalars_bls12377).unwrap();
            println!("Arkworks Bls12377 result: {:#?}", bls12377_ark_msm_res);
            #[cfg(feature = "profile")]
            println!("Ark BLS12377 MSM on size 2^{log_size} took: {} ms", start.elapsed().as_millis());

            let bn254_icicle_msm_res_as_ark = msm_host_result[0].to_ark();
            let bls12377_icicle_msm_res_as_ark = msm_host_result_bls12377[0].to_ark();

            println!("Bn254 MSM is correct: {}", bn254_ark_msm_res.eq(&bn254_icicle_msm_res_as_ark));
            println!("Bls12377 MSM is correct: {}", bls12377_ark_msm_res.eq(&bls12377_icicle_msm_res_as_ark));
        }
        
        println!("Cleaning up bn254...");
        stream
            .destroy()
            .unwrap();
        println!("Cleaning up bls12377...");
        stream_bls12377
            .destroy()
            .unwrap();
        println!("");
    }
}
