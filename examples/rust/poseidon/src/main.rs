use icicle_bls12_381::curve::ScalarField as F;

use icicle_cuda_runtime::device_context::DeviceContext;

use icicle_core::hash::{SpongeHash, HashConfig};
use icicle_core::poseidon::Poseidon;
use icicle_core::traits::FieldImpl;
use icicle_cuda_runtime::memory::HostSlice;

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
    let args = Args::parse();
    let size = args.size;
    let test_size = 1 << size;

    println!("Running Icicle Examples: Rust Poseidon Hash");
    let arity = 2;
    println!(
        "---------------------- Loading optimized Poseidon constants for arity={} ------------------------",
        arity
    );
    let ctx = DeviceContext::default();
    let poseidon = Poseidon::load(arity, &ctx).unwrap();
    let config = HashConfig::default();

    println!(
        "---------------------- Input size 2^{}={} ------------------------",
        size, test_size
    );
    let mut inputs = vec![F::one(); test_size * arity as usize];
    let mut outputs = vec![F::zero(); test_size];
    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    println!("Executing BLS12-381 Poseidon Hash on device...");
    #[cfg(feature = "profile")]
    let start = Instant::now();
    poseidon.hash_many(
        input_slice,
        output_slice,
        test_size,
        arity,
        1,
        &config,
    )
    .unwrap();
    #[cfg(feature = "profile")]
    println!(
        "ICICLE BLS12-381 Poseidon Hash on size 2^{size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );
}
