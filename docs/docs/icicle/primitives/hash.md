# ICICLE Hashing Logic

## Overview

ICICLE’s hashing system is designed to be flexible, efficient, and optimized for both general-purpose and cryptographic operations. Hash functions are essential in operations such as generating commitments, constructing Merkle trees, executing the Sumcheck protocol, and more.

ICICLE provides an easy-to-use interface for hashing on both CPU and GPU, with transparent backend selection. You can choose between several hash algorithms such as Keccak-256, Keccak-512, SHA3-256, SHA3-512, Blake2s, and Poseidon, which are optimized for processing both general data and cryptographic field elements or elliptic curve points.

## Hashing Logic

Hashing in ICICLE involves creating a hasher instance for the desired algorithm, configuring the hash function if needed, and then processing the data. Data can be provided as strings, arrays, or field elements, and the output is collected in a buffer that automatically adapts to the size of the hashed data.

## Batch Hashing

For scenarios where large datasets need to be hashed efficiently, ICICLE supports batch hashing. The batch size is automatically derived from the output size, making it adaptable and optimized for parallel computation on the GPU (when using the CUDA backend). This is useful for Merkle-trees and more.

## Supported Hash Algorithms

ICICLE supports the following hash functions:

1.  **Keccak-256**
2.	**Keccak-512**
3.	**SHA3-256**
4.	**SHA3-512**
5.	**Blake2s**
6.	**Poseidon**

### Keccak and SHA3

[Keccak](https://keccak.team/files/Keccak-implementation-3.2.pdf) is a cryptographic hash function designed by Guido Bertoni, Joan Daemen, Michaël Peeters, and Gilles Van Assche. It was selected as the winner of the NIST hash function competition, becoming the basis for the [SHA-3 standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf).

Keccak can take input messages of any length and produce a fixed-size hash. It uses the sponge construction, which absorbs the input data and squeezes out the final hash value. The permutation function, operating on a state array, applies iterative rounds of operations to derive the hash.

### Blake2s

Blake2s is an optimized cryptographic hash function that provides high performance while ensuring strong security. Blake2s is ideal for hashing small data (such as field elements), especially when speed is crucial. It produces a 256-bit (32-byte) output and is often used in cryptographic protocols.


### Poseidon

Poseidon is a hash function designed specifically for cryptographic field elements and elliptic curve points. It is optimized for zero-knowledge proofs (ZKPs) and is often used in ZK-SNARK systems. Poseidon’s strength lies in its efficiency when working with cryptographic data, making it ideal for scenarios like Merkle tree construction and proof generation.


## Using Hash API

### 1. Creating a Hasher Object

First, you need to create a hasher object for the specific hash function you want to use:

```cpp
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"
#include "icicle/hash/poseidon.h"

// Create hasher instances for different algorithms
auto keccak256 = Keccak256::create();
auto keccak512 = Keccak512::create();
auto sha3_256 = Sha3_256::create();
auto sha3_512 = Sha3_512::create();
auto blake2s = Blake2s::create();
auto poseidon = Poseidon::create<scalar_t>(arity); // Poseidon requires specifying the field type and arity
```

### 2. Hashing Data

Once you have a hasher object, you can hash any input data by passing the input, its size, a configuration, and an output buffer:
   
```cpp
/**
 * @brief Perform a hash operation.
 *
 * This function delegates the hash operation to the backend.
 *
 * @param input Pointer to the input data as bytes.
 * @param size The number of bytes to hash. If 0, the default chunk size is used.
 * @param config Configuration options for the hash operation.
 * @param output Pointer to the output data as bytes.
 * @return An error code of type eIcicleError indicating success or failure.
 */
eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const;

/**
 * @brief Perform a hash operation using typed data.
 *
 * Converts input and output types to `std::byte` pointers and delegates the call to the backend.
 *
 * @tparam PREIMAGE The type of the input data.
 * @tparam IMAGE The type of the output data.
 * @param input Pointer to the input data.
 * @param size The number of elements of type `PREIMAGE` to hash.
 * @param config Configuration options for the hash operation.
 * @param output Pointer to the output data.
 * @return An error code of type eIcicleError indicating success or failure.
 */
template <typename PREIMAGE, typename IMAGE>
eIcicleError hash(const PREIMAGE* input, uint64_t size, const HashConfig& config, IMAGE* output) const;
```

Example Usage:

```cpp
// Using the Blake2s hasher
const std::string input = "Hello, I am Blake2s!";
const uint64_t output_size = 32; // Blake2s outputs 32 bytes
auto output = std::make_unique<std::byte[]>(output_size);
auto config = default_hash_config();

eIcicleErr err = blake2s.hash(input.data(), input.size(), config, output.get());

// Alternatively, use another hasher (e.g., Keccak256, SHA3-512)
```

### 3. Batch Hashing

To perform batch hashing, set the `config.batch` field to indicate the number of batches. This allows for multiple inputs to be hashed in parallel:

```cpp
auto config = default_hash_config();
config.batch = 2;

const std::string input = "0123456789abcdef"; // This is a batch of "01234567" and "89abcdef"
auto output = std::make_unique<std::byte[]>(32 * config.batch); // Allocate output for 2 batches

eIcicleErr err = keccak256.hash(input.data(), input.size() / config.batch, config, output.get());
```

### Supported Bindings

- [Rust](../rust-bindings/hash)
- Go bindings soon
