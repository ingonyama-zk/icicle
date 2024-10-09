use clap::Parser;
use hex;
use icicle_babybear::field::ScalarCfg as BabybearCfg;
use icicle_core::{hash::HashConfig, traits::GenerateRandom};
use icicle_hash::keccak::Keccak256;
use icicle_runtime::memory::HostSlice;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
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

fn keccak_hash_example() {
    // (1) construct a keccak-256 hasher
    // NOTE that any other hasher can be used here including sha3, blake2s, poseidon.
    let keccak_hasher = Keccak256::new(0 /*=default input size */).unwrap();

    // (2) hash a string
    let input_str = "I like ICICLE! it's so fast and easy";
    let expected_hash = "9fac4e3dc249b59cc57bdec04c132073f0f6ef3a216ef5b3b75815292fb7a45e";
    let mut output = vec![0u8; 32]; // 32B for output. TODO Can I use something else?

    keccak_hasher
        .hash(
            HostSlice::from_slice(input_str.as_bytes()),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    let output_as_hex_str = hex::encode(output);
    println!("hash(`{}`)={:?}", input_str, &output_as_hex_str);
    assert_eq!(expected_hash, output_as_hex_str);

    // (3) hash field elements
    let input_field_elements = BabybearCfg::generate_random(128);
    let mut output = vec![0u8; 32]; // 32B for output. TODO Can I use something else?
    keccak_hasher
        .hash(
            HostSlice::from_slice(&input_field_elements),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();
    let output_as_hex_str = hex::encode(output);
    println!("hash(```babybear field elements```)={:?}", &output_as_hex_str);

    // (4) hash field elements batch
    let batch = 1 << 10;
    let single_hash_input_nof_elements = 1 << 12;
    let total_input_elements = batch * single_hash_input_nof_elements;
    let input_field_elements_batch = BabybearCfg::generate_random(total_input_elements);
    // NOTE that a digest is allocated per batch. The size of the output is what defines the batch for the hash function
    let mut output = vec![0x8; 32 * batch];
    let start = Instant::now();
    keccak_hasher
        .hash(
            HostSlice::from_slice(&input_field_elements_batch),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();
    println!(
        "hashing {} batches of {} field elements on selected device took (from host memory): {} ms",
        batch,
        single_hash_input_nof_elements,
        start
            .elapsed()
            .as_millis()
    );
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);
    try_load_and_set_backend_device(&args);

    // Keccak hashing example
    keccak_hash_example()

    // Merkle-tree example - TODO
}
