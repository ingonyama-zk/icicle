use clap::Parser;
use hex;
use icicle_babybear::field::ScalarCfg as BabybearCfg;
use icicle_core::{hash::HashConfig, traits::GenerateRandom};
use icicle_hash::keccak::Keccak256;
use icicle_runtime::memory::HostSlice;
use std::time::Instant;

/// Command-line argument parser
#[derive(Parser, Debug)]
struct Args {
    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

/// Load backend and set the device based on input arguments
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device to {}", args.device_type);

    // Create and set the device
    let device = icicle_runtime::Device::new(&args.device_type, 0); // device_id = 0
    icicle_runtime::set_device(&device).unwrap();
}

/// Example of Keccak-256 hashing using the ICICLE framework
fn keccak_hash_example() {
    // 1. Create a Keccak-256 hasher instance
    let keccak_hasher = Keccak256::new(0 /*=default input size */).unwrap();
    // Note: the default input size is useful in some cases. Can be ignored in this example.

    // 2. Hash a simple string
    let input_str = "I like ICICLE! it's so fast and easy";
    let expected_hash = "9fac4e3dc249b59cc57bdec04c132073f0f6ef3a216ef5b3b75815292fb7a45e";
    let mut output = vec![0u8; 32]; // Output buffer for the hash

    keccak_hasher
        .hash(
            HostSlice::from_slice(input_str.as_bytes()),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    // Convert the output to a hex string and print the result
    let output_as_hex_str = hex::encode(output);
    println!("Hash(`{}`) = {:?}", input_str, &output_as_hex_str);
    assert_eq!(expected_hash, output_as_hex_str);

    // 3. Hash field elements (Babybear field elements)
    let input_field_elements = BabybearCfg::generate_random(128); // Generate random field elements
    let mut output = vec![0u8; 32]; // Output buffer for the hash

    keccak_hasher
        .hash(
            HostSlice::from_slice(&input_field_elements),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    // Convert the output to a hex string and print the result
    let output_as_hex_str = hex::encode(output);
    println!("Hash(```babybear field elements```) = {:?}", &output_as_hex_str);

    // 4. Hash field elements in batch
    let batch = 1 << 10; // 1024
    let single_hash_nof_elements = 1 << 12; // 4096
    let total_input_elements = batch * single_hash_nof_elements;
    let input_field_elements_batch = BabybearCfg::generate_random(total_input_elements);

    // The size of the output determines the batch size for the hash function
    let mut output = vec![0u8; 32 * batch]; // Output buffer for batch hashing. Doesn't have to be u8.

    let start = Instant::now(); // Start timer for performance measurement
    keccak_hasher
        .hash(
            HostSlice::from_slice(&input_field_elements_batch),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();

    // Print the time taken for hashing in milliseconds
    println!(
        "Hashing {} batches of {} field elements took: {} ms",
        batch,
        single_hash_nof_elements,
        start
            .elapsed()
            .as_millis()
    );

    // NOTE: like other ICICLE apis, this also works with DeviceSlice for on-device data.
}

fn main() {
    // Parse command-line arguments
    let args = Args::parse();
    println!("{:?}", args);

    // Load backend and set the device
    try_load_and_set_backend_device(&args);

    // Execute the Keccak hashing example
    keccak_hash_example();

    // TODO: Implement a Merkle tree example
}
