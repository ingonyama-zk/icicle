use icicle_runtime::memory::{DeviceVec, HostSlice};

use icicle_bls12_377::curve::{ScalarCfg as BLS12377ScalarCfg, ScalarField as BLS12377ScalarField};
use icicle_bn254::curve::{ScalarCfg as Bn254ScalarCfg, ScalarField as Bn254ScalarField};

use clap::Parser;
use icicle_core::{
    ntt::{self, initialize_domain},
    traits::{FieldImpl, GenerateRandom},
};
use std::convert::TryInto;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    /// Size of NTT to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    size: u8,

    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,

    /// Backend installation directory
    #[arg(short, long, default_value = "")]
    backend_install_dir: String,
}

// Load backend and set device
fn try_load_and_set_backend_device(args: &Args) {
    if !args
        .backend_install_dir
        .is_empty()
    {
        println!("trying to load backend from {}", &args.backend_install_dir);
        icicle_runtime::runtime::load_backend(&args.backend_install_dir, true /*recursive */).unwrap();
    }
    println!("Setting device {}", args.device_type);
    icicle_runtime::set_device(&icicle_runtime::Device::new(&args.device_type, 0)).unwrap();
}

fn main() {
    let args = Args::parse();
    try_load_and_set_backend_device(&args);

    println!("Running Icicle Examples: Rust NTT");
    let log_size = args.size;
    let size = 1 << log_size;
    println!(
        "---------------------- NTT size 2^{}={} ------------------------",
        log_size, size
    );
    // Setting Bn254 points and scalars
    println!("Generating random inputs on host for bn254...");
    let scalars = Bn254ScalarCfg::generate_random(size);
    let mut ntt_results = DeviceVec::<Bn254ScalarField>::device_malloc(size).unwrap();

    // Setting bls12377 points and scalars
    println!("Generating random inputs on host for bls12377...");
    let scalars_bls12377 = BLS12377ScalarCfg::generate_random(size);
    let mut ntt_results_bls12377 = DeviceVec::<BLS12377ScalarField>::device_malloc(size).unwrap();

    println!("Setting up bn254 Domain...");
    initialize_domain(
        ntt::get_root_of_unity::<Bn254ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    println!("Configuring bn254 NTT...");
    let cfg = ntt::NTTConfig::<Bn254ScalarField>::default();

    println!("Setting up bls12377 Domain...");
    // reusing ctx from above
    initialize_domain(
        ntt::get_root_of_unity::<BLS12377ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    println!("Configuring bls12377 NTT...");
    let cfg_bls12377 = ntt::NTTConfig::<BLS12377ScalarField>::default();

    println!("Executing bn254 NTT on device...");
    let start = Instant::now();
    ntt::ntt(
        HostSlice::from_slice(&scalars),
        ntt::NTTDir::kForward,
        &cfg,
        &mut ntt_results[..],
    )
    .unwrap();
    println!(
        "ICICLE BN254 NTT on size 2^{log_size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );

    println!("Executing bls12377 NTT on device...");
    let start = Instant::now();
    ntt::ntt(
        HostSlice::from_slice(&scalars_bls12377),
        ntt::NTTDir::kForward,
        &cfg_bls12377,
        &mut ntt_results_bls12377[..],
    )
    .unwrap();
    println!(
        "ICICLE BLS12377 NTT on size 2^{log_size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );

    println!("Moving results to host..");
    let mut host_bn254_results = vec![Bn254ScalarField::zero(); size];
    ntt_results
        .copy_to_host(HostSlice::from_mut_slice(&mut host_bn254_results[..]))
        .unwrap();

    let mut host_bls12377_results = vec![BLS12377ScalarField::zero(); size];
    ntt_results_bls12377
        .copy_to_host(HostSlice::from_mut_slice(&mut host_bls12377_results[..]))
        .unwrap();
}
