# ICICLE Hashing Logic

## Overview

ICICLE’s hashing system is designed to be flexible, efficient, and optimized for both general-purpose and cryptographic operations. Hash functions are essential in operations such as generating commitments, constructing Merkle trees, executing the Sumcheck protocol, and more.

ICICLE provides an easy-to-use interface for hashing on both CPU and GPU, with transparent backend selection. You can choose between several hash algorithms such as Keccak-256, Keccak-512, SHA3-256, SHA3-512, Blake2s, Poseidon, Poseidon2 and more, which are optimized for processing both general data and cryptographic field elements or elliptic curve points.

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
7.	**Poseidon2**

:::info
Additional hash functions might be added in the future. Stay tuned!
:::

### Keccak and SHA3

[Keccak](https://keccak.team/files/Keccak-implementation-3.2.pdf) is a cryptographic hash function designed by Guido Bertoni, Joan Daemen, Michaël Peeters, and Gilles Van Assche. It was selected as the winner of the NIST hash function competition, becoming the basis for the [SHA-3 standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf).

Keccak can take input messages of any length and produce a fixed-size hash. It uses the sponge construction, which absorbs the input data and squeezes out the final hash value. The permutation function, operating on a state array, applies iterative rounds of operations to derive the hash.

### Blake2s

[Blake2s](https://www.rfc-editor.org/rfc/rfc7693.txt) is an optimized cryptographic hash function that provides high performance while ensuring strong security. Blake2s is ideal for hashing small data (such as field elements), especially when speed is crucial. It produces a 256-bit (32-byte) output and is often used in cryptographic protocols.


### Poseidon

[Poseidon](https://eprint.iacr.org/2019/458) is a cryptographic hash function designed specifically for field elements. It is highly optimized for zero-knowledge proofs (ZKPs) and is commonly used in ZK-SNARK systems. Poseidon’s main strength lies in its arithmetization-friendly design, meaning it can be efficiently expressed as arithmetic constraints within a ZK-SNARK circuit.

Traditional hash functions, such as SHA-2, are difficult to represent within ZK circuits because they involve complex bitwise operations that don’t translate efficiently into arithmetic operations. Poseidon, however, is specifically designed to minimize the number of constraints required in these circuits, making it significantly more efficient for use in ZK-SNARKs and other cryptographic protocols that require hashing over field elements.

Currently the Poseidon implementation is the Optimized Poseidon (https://hackmd.io/@jake/poseidon-spec#Optimized-Poseidon). Optimized Poseidon significantly decreases the calculation time of the hash.

The optional `domain_tag` pointer parameter enables domain separation, allowing isolation of hash outputs across different contexts or applications.


### Poseidon2

[Poseidon2](https://eprint.iacr.org/2023/323.pdf) is a cryptographic hash function designed specifically for field elements.
It is an improved version of the original [Poseidon](https://eprint.iacr.org/2019/458) hash, offering better performance on modern hardware. Poseidon2 is optimized for use with elliptic curve cryptography and finite fields, making it ideal for decentralized systems like blockchain. Its main advantage is balancing strong security with efficient computation, which is crucial for applications that require fast, reliable hashing.

The optional `domain_tag` pointer parameter enables domain separation, allowing isolation of hash outputs across different contexts or applications.

The supported values of number of states (***t*** or ***T*** as defined in https://eprint.iacr.org/2023/323.pdf) are 2, 3, 4, 8, 12, 16, 20 and 24. Note that ***t*** of 8, 12, 16, 20 and 24 is supported only for the small fields (babybear and m31).
The alpha, number of full rounds and partial rounds, rounds constants, MDS matrix, and partial matrix for each field and ***t*** could be found in the appropriate file in the https://github.com/ingonyama-zk/icicle/tree/main/icicle/include/icicle/hash/poseidon2_constants/constants folder.

In the current version the padding is not supported and should be performed by the user.

## Using Hash API

### 1. Creating a Hasher Object

First, you need to create a hasher object for the specific hash function you want to use:

```cpp
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"
#include "icicle/hash/poseidon.h"
#include "icicle/hash/poseidon2.h"

// Create hasher instances for different algorithms
auto keccak256 = Keccak256::create();
auto keccak512 = Keccak512::create();
auto sha3_256 = Sha3_256::create();
auto sha3_512 = Sha3_512::create();
auto blake2s = Blake2s::create();
// Poseidon requires specifying the field-type and t parameter (supported 3,5,9,12) as defined by the Poseidon paper.
auto poseidon = Poseidon::create<scalar_t>(t); 
// Optionally, Poseidon can accept a domain-tag, which is a field element used to separate applications or contexts.
// The domain tag acts as the first input to the hash function, with the remaining t-1 inputs following it.
scalar_t domain_tag = scalar_t::zero(); // Example using zero; this can be set to any valid field element.
auto poseidon_with_domain_tag = Poseidon::create<scalar_t>(t, &domain_tag);
// Poseidon2 requires specifying the field-type and t parameter (supported 2, 3, 4, 8, 12, 16, 20, 24) as defined by
// the Poseidon2 paper. For large fields (field width >= 252) the supported values of t are 2, 3, 4.
auto poseidon2 = Poseidon2::create<scalar_t>(t); 
// Optionally, Poseidon2 can accept a domain-tag, which is a field element used to separate applications or contexts.
// The domain tag acts as the first input to the hash function, with the remaining t-1 inputs following it.
scalar_t domain_tag = scalar_t::zero(); // Example using zero; this can be set to any valid field element.
auto poseidon2_with_domain_tag = Poseidon2::create<scalar_t>(t, &domain_tag);
// This version of the hasher with a domain tag expects t-1 inputs per hasher.
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
 * @param size The number of elements of type `PREIMAGE` to a single hasher.
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

### 4. Poseidon sponge function

Currently the poseidon sponge function (sponge function description could be found in Sec 2.1 of https://eprint.iacr.org/2019/458.pdf ) isn't implemented.

### 5. Poseidon2 sponge function

Currently the poseidon2 sponge function (sponge function description could be found in Sec 2.1 of https://eprint.iacr.org/2019/458.pdf ) isn't implemented.

### Supported Bindings

- [Rust](../rust-bindings/hash)
- [Go](../golang-bindings/hash)
