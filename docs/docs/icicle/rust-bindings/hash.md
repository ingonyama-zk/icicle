# ICICLE Hashing in Rust

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

    let batch = 1 << 10;
    let arity = 3;
    let inputs = ScalarCfg::generate_random(batch * arity);
    let mut outputs = vec![ScalarField::zero(); batch]; // note output array is sized for batch

    let poseidon_hasher = Poseidon::new::<ScalarField>(arity as u32).unwrap();

    poseidon_hasher
        .hash(
            HostSlice::from_slice(&inputs),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut outputs),
        )
        .unwrap();
```

