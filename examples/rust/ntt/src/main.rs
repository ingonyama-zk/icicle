use icicle_bn254::curve::{
    ScalarCfg,
    ScalarField,
};

use icicle_bls12_377::curve::{
    ScalarCfg as BLS12377ScalarCfg,
    ScalarField as BLS12377ScalarField
};

use icicle_cuda_runtime::{
    stream::CudaStream,
    memory::HostOrDeviceSlice,
    device_context::get_default_device_context
};

use icicle_core::{
    ntt::{self, NTT},
    traits::{GenerateRandom, FieldImpl}
};

use icicle_core::traits::ArkConvertible;

use ark_bn254::Fr as Bn254Fr;
use ark_bls12_377::Fr as Bls12377Fr;
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::cmp::{Ord, Ordering};
use std::convert::TryInto;

#[cfg(feature = "profile")]
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Size of NTT to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    size: u8,
}

fn main() {
    let args = Args::parse();
    println!("Running Icicle Examples: Rust NTT");
    let log_size = args.size;
    let size = 1 << log_size;
    println!("---------------------- NTT size 2^{}={} ------------------------", log_size, size);
    // Setting Bn254 points and scalars
    println!("Generating random inputs on host for bn254...");
    let scalars = HostOrDeviceSlice::Host(ScalarCfg::generate_random(size));
    let mut ntt_results: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::cuda_malloc(size).unwrap();
    
    // Setting bls12377 points and scalars
    println!("Generating random inputs on host for bls12377...");
    let scalars_bls12377 = HostOrDeviceSlice::Host(BLS12377ScalarCfg::generate_random(size));
    let mut ntt_results_bls12377: HostOrDeviceSlice<'_, BLS12377ScalarField> = HostOrDeviceSlice::cuda_malloc(size).unwrap();
    
    println!("Setting up bn254 Domain...");
    let icicle_omega = <Bn254Fr as FftField>::get_root_of_unity(size.try_into().unwrap()).unwrap();
    let ctx = get_default_device_context();
    ScalarCfg::initialize_domain(ScalarField::from_ark(icicle_omega), &ctx).unwrap();

    println!("Configuring bn254 NTT...");
    let stream = CudaStream::create().unwrap();
    let mut cfg = ntt::get_default_ntt_config::<ScalarField>();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;

    println!("Setting up bls12377 Domain...");
    let icicle_omega = <Bls12377Fr as FftField>::get_root_of_unity(size.try_into().unwrap()).unwrap();
    // reusing ctx from above
    BLS12377ScalarCfg::initialize_domain(BLS12377ScalarField::from_ark(icicle_omega), &ctx).unwrap();

    println!("Configuring bls12377 NTT...");
    let stream_bls12377 = CudaStream::create().unwrap();
    let mut cfg_bls12377 = ntt::get_default_ntt_config::<BLS12377ScalarField>();
    cfg_bls12377.ctx.stream = &stream_bls12377;
    cfg_bls12377.is_async = true;

    println!("Executing bn254 NTT on device...");
    #[cfg(feature = "profile")]
    let start = Instant::now();
    ntt::ntt(&scalars, ntt::NTTDir::kForward, &cfg, &mut ntt_results).unwrap();
    #[cfg(feature = "profile")]
    println!("ICICLE BN254 NTT on size 2^{log_size} took: {} μs", start.elapsed().as_micros());

    println!("Executing bls12377 NTT on device...");
    #[cfg(feature = "profile")]
    let start = Instant::now();
    ntt::ntt(&scalars_bls12377, ntt::NTTDir::kForward, &cfg_bls12377, &mut ntt_results_bls12377).unwrap();
    #[cfg(feature = "profile")]
    println!("ICICLE BLS12377 NTT on size 2^{log_size} took: {} μs", start.elapsed().as_micros());

    println!("Moving results to host..");
    stream
        .synchronize()
        .unwrap();
    let mut host_bn254_results = vec![ScalarField::zero(); size];
    ntt_results
        .copy_to_host(&mut host_bn254_results[..])
        .unwrap();
    
    stream_bls12377
        .synchronize()
        .unwrap();
    let mut host_bls12377_results = vec![BLS12377ScalarField::zero(); size];
    ntt_results_bls12377
        .copy_to_host(&mut host_bls12377_results[..])
        .unwrap();
    
    println!("Checking against arkworks...");
    let mut ark_scalars: Vec<Bn254Fr> = scalars.as_slice().iter().map(|scalar| scalar.to_ark()).collect();
    let bn254_domain = <Radix2EvaluationDomain<Bn254Fr> as EvaluationDomain<Bn254Fr>>::new(size).unwrap();
    
    let mut ark_scalars_bls12377: Vec<Bls12377Fr> = scalars_bls12377.as_slice().iter().map(|scalar| scalar.to_ark()).collect();
    let bls12_377_domain = <Radix2EvaluationDomain<Bls12377Fr> as EvaluationDomain<Bls12377Fr>>::new(size).unwrap();
    
    #[cfg(feature = "profile")]
    let start = Instant::now();
    bn254_domain.fft_in_place(&mut ark_scalars);
    #[cfg(feature = "profile")]
    println!("Ark BN254 NTT on size 2^{log_size} took: {} ms", start.elapsed().as_millis());

    #[cfg(feature = "profile")]
    let start = Instant::now();
    bls12_377_domain.fft_in_place(&mut ark_scalars_bls12377);
    #[cfg(feature = "profile")]
    println!("Ark BLS12377 NTT on size 2^{log_size} took: {} ms", start.elapsed().as_millis());

    host_bn254_results
        .iter()
        .zip(ark_scalars.iter())
        .for_each(|(icicle_scalar, &ark_scalar)| {
            assert_eq!(ark_scalar.cmp(&icicle_scalar.to_ark()), Ordering::Equal);
        });
    println!("Bn254 NTT is correct");
    
    host_bls12377_results
        .iter()
        .zip(ark_scalars_bls12377.iter())
        .for_each(|(icicle_scalar, &ark_scalar)| {
            assert_eq!(ark_scalar.cmp(&icicle_scalar.to_ark()), Ordering::Equal);
        });

    println!("Bls12377 NTT is correct");
    
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
