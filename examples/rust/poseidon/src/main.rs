use icicle_bls12_381::poseidon;

use icicle_cuda_runtime::device_context::get_default_device_context;

use icicle_core::poseidon::{load_optimized_poseidon_constants, poseidon_hash_many, PoseidonConfig};

#[cfg(feature = "profile")]
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Size of Poseidon input to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    size: u8,
}

fn main() {
    let size = args.size;
    let test_size = 1 << size;

    println!("Running Icicle Examples: Rust Poseidon Hash");
    let arity = 2u32;
    println!("---------------------- Loading optimized Poseidon constants for arity={} ------------------------", arity);
    let ctx = get_default_device_context();
    let constants = load_optimized_poseidon_constants::<F>(arity, &ctx).unwrap();
    let config = PoseidonConfig::default();

    println!("---------------------- Input size 2^{}={} ------------------------", size, test_size);
    let inputs = vec![F::one(); test_size * arity as usize];
    let outputs = vec![F::zero(); test_size];
    let mut input_slice = HostOrDeviceSlice::on_host(inputs);
    let mut output_slice = HostOrDeviceSlice::on_host(outputs);

    println!("Executing BLS12-381 Poseidon Hash on device...");
    #[cfg(feature = "profile")]
    let start = Instant::now();
    poseidon_hash_many::<F>(
        &mut input_slice,
        &mut output_slice,
        test_size as u32,
        arity as u32,
        &constants,
        &config,
    )
    .unwrap();
    #[cfg(feature = "profile")]
    println!("ICICLE BLS12-381 Poseidon Hash on size 2^{size} took: {} μs", start.elapsed().as_micros());
}
