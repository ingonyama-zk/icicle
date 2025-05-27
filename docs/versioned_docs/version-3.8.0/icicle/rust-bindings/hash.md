# ICICLE Hashing in Rust

:::note
For a general overview of ICICLE's hashing logic and supported algorithms, check out the [ICICLE Hashing Overview](../primitives/hash.md).
:::

## Overview

The ICICLE library provides Rust bindings for hashing using a variety of cryptographic hash functions. These hash functions are optimized for both general-purpose data and cryptographic operations such as multi-scalar multiplication, commitment generation, and Merkle tree construction.

This guide will show you how to use the ICICLE hashing API in Rust with examples for common hash algorithms, such as Keccak-256, Keccak-512, SHA3-256, SHA3-512, Blake2s, and Poseidon.

## Importing Hash Functions

To use the hashing functions in Rust, you will need to import the specific hash algorithm module from the ICICLE Rust bindings. For example:

```rust
use icicle_hash::keccak::Keccak256;
use icicle_hash::keccak::Keccak512;
use icicle_hash::sha3::Sha3_256;
use icicle_hash::sha3::Sha3_512;
use icicle_hash::blake2s::Blake2s;
use icicle_core::poseidon::Poseidon;
```

## API Usage

### 1. Creating a Hasher Instance

Each hash algorithm can be instantiated by calling its respective constructor. The new function takes an optional default input size, which can be set to 0 unless required for a specific use case.

Example for Keccak-256:

```rust
let keccak_hasher = Keccak256::new(0 /* default input size */).unwrap();
```

### 2. Hashing a Simple String

Once you have created a hasher instance, you can hash any input data, such as strings or byte arrays, and store the result in an output buffer.
Here’s how to hash a simple string using Keccak-256:

```rust
use icicle_hash::keccak::Keccak256;
use icicle_runtime::memory::HostSlice;
use icicle_core::hash::HashConfig;
use hex;

let input_str = "I like ICICLE! it's so fast and easy";
let mut output = vec![0u8; 32]; // 32-byte output buffer

let keccak_hasher = Keccak256::new(0 /* default input size */).unwrap();
keccak_hasher
    .hash(
        HostSlice::from_slice(input_str.as_bytes()),  // Input data
        &HashConfig::default(),                       // Default configuration
        HostSlice::from_mut_slice(&mut output),       // Output buffer
    )
    .unwrap();

// convert the output to a hex string for easy readability
let output_as_hex_str = hex::encode(output);
println!("Hash(`{}`) = {:?}", input_str, &output_as_hex_str);

```

### 3. Poseidon Example (field elements) and batch hashing

The Poseidon hash is designed for cryptographic field elements and curves, making it ideal for use cases such as zero-knowledge proofs (ZKPs).
Poseidon hash using babybear field:

```rust
use icicle_babybear::field::{ScalarCfg, ScalarField};
use icicle_core::hash::HashConfig;
use icicle_core::poseidon::{Poseidon, PoseidonHasher};
use icicle_core::traits::FieldImpl;
use icicle_runtime::memory::HostSlice;

let batch = 1 << 10; // Number of hashes to compute in a single batch
let t = 3; // Poseidon parameter that specifies the arity (number of inputs) for each hash function
let mut outputs = vec![ScalarField::zero(); batch]; // Output array sized for the batch count

// Case (1): Hashing without a domain tag
// Generates 'batch * t' random input elements as each hash needs 't' inputs
let inputs = ScalarCfg::generate_random(batch * t);
let poseidon_hasher = Poseidon::new::<ScalarField>(t as u32, None /*=domain-tag*/).unwrap(); // Instantiate Poseidon without domain tag

poseidon_hasher
    .hash(
        HostSlice::from_slice(&inputs),           // Input slice for the hash function
        &HashConfig::default(),                   // Default hashing configuration
        HostSlice::from_mut_slice(&mut outputs),  // Output slice to store hash results
    )
    .unwrap();

// Case (2): Hashing with a domain tag
// Generates 'batch * (t - 1)' inputs, as domain tag counts as one input in each hash
let inputs = ScalarCfg::generate_random(batch * (t - 1));
let domain_tag = ScalarField::zero(); // Example domain tag (can be any valid field element)
let poseidon_hasher_with_domain_tag = Poseidon::new::<ScalarField>(t as u32, Some(&domain_tag) /*=domain-tag*/).unwrap();

poseidon_hasher_with_domain_tag
    .hash(
        HostSlice::from_slice(&inputs),           // Input slice with 't - 1' elements per hash
        &HashConfig::default(),                   // Default hashing configuration
        HostSlice::from_mut_slice(&mut outputs),  // Output slice to store hash results
    )
    .unwrap();
```

