# ICICLE Example: Keccak-256 Hashing and Merkle-Tree

## Key Takeaway

This example demonstrates how to perform Keccak-256 hashing on both the CPU and CUDA backends using the ICICLE framework. The example includes hashing string data, Babybear field elements, and batched field elements, showcasing how to efficiently use the ICICLE runtime to process data on different devices.

> [!NOTE]
> This example uses the Keccak-256 hash function, but you can easily substitute it with other hash algorithms supported by ICICLE (e.g., SHA3, Blake2s, Poseidon). Just replace the Keccak256 type in the example with your desired hashing algorithm.

## Batch Size and Input Flexibility

The **batch size** for hashing is automatically derived from the size of the output buffer, meaning it scales according to the size of the output you need. 
This design also allows the same hashing function to efficiently handle any input type, whether it’s simple strings, scalar field elements, or elliptic curve points.

## Usage

```rust
// Initialize the Keccak-256 hasher
let keccak_hasher = Keccak256::new(0).unwrap();

// Hash a simple string
let input_str = "I like ICICLE! it's so fast and easy";
let mut output = vec![0u8; 32]; // Output buffer for hash result
keccak_hasher.hash(
    HostSlice::from_slice(input_str.as_bytes()),
    &HashConfig::default(),
    HostSlice::from_mut_slice(&mut output),
).unwrap();

// Hash Babybear field elements
let input_field_elements = BabybearCfg::generate_random(128);
let mut output = vec![0u8; 32]; // Output buffer for hash result
keccak_hasher.hash(
    HostSlice::from_slice(&input_field_elements),
    &HashConfig::default(),
    HostSlice::from_mut_slice(&mut output),
).unwrap();
```

This example demonstrates how to hash both simple strings and field elements using the Keccak-256 hashing algorithm, which is commonly used in cryptographic applications.

## What's in this example

1. Set up the ICICLE backend (CPU or CUDA).
2. String Hashing: Demonstrates hashing of basic strings using Keccak-256.
3. Field Element Hashing: Hash Babybear field elements.
4. Batch Hashing: Hash a large batch of field elements and measure the performance.
5. (TODO merkle tree)

## Running the Example

To run the example, use the following commands based on your backend setup:

```sh
# For CPU
./run.sh -d CPU

# For CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
```

This will execute the example using either the CPU or CUDA backend, allowing you to test the conversion and computation process between Arkworks and ICICLE.
