# Keccak

TODO update for V3

[Keccak](https://keccak.team/files/Keccak-implementation-3.2.pdf) is a cryptographic hash function designed by Guido Bertoni, Joan Daemen, Michaël Peeters, and Gilles Van Assche. It was selected as the winner of the NIST hash function competition, becoming the basis for the [SHA-3 standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf).

Keccak operates on a message input of any length and produces a fixed-size hash output. The hash function is built upon the sponge construction, which involves absorbing the input data followed by squeezing out the hash value.

At its core, Keccak consists of a permutation function operating on a state array. The permutation function employs a round function that operates iteratively on the state array. Each round consists of five main steps:

- **Theta:** This step introduces diffusion by performing a bitwise XOR operation between the state and a linear combination of its neighboring columns.
- **Rho:** This step performs bit rotation operations on each lane of the state array.
- **Pi:** This step rearranges the positions of the lanes in the state array.
- **Chi:** This step applies a nonlinear mixing operation to each lane of the state array.
- **Iota:** This step introduces a round constant to the state array.

## Using Keccak

ICICLE Keccak supports batch hashing, which can be utilized for constructing a merkle tree or running multiple hashes in parallel.

### Supported Bindings

- [Golang](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/hash/keccak)
- [Rust](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-hash)

### Example usage

This is an example of running 1024 Keccak-256 hashes in parallel, where input strings are of size 136 bytes:

```rust
use icicle_core::hash::HashConfig;
use icicle_cuda_runtime::memory::HostSlice;
use icicle_hash::keccak::keccak256;

let config = HashConfig::default();
let input_block_len = 136;
let number_of_hashes = 1024;

let preimages = vec![1u8; number_of_hashes * input_block_len];
let mut digests = vec![0u8; number_of_hashes * 64];

let preimages_slice = HostSlice::from_slice(&preimages);
let digests_slice = HostSlice::from_mut_slice(&mut digests);

keccak256(
    preimages_slice,
    input_block_len as u32,
    number_of_hashes as u32,
    digests_slice,
    &config,
)
.unwrap();
```

### Merkle Tree

You can build a keccak merkle tree using the corresponding functions:

```rust
use icicle_core::tree::{merkle_tree_digests_len, TreeBuilderConfig};
use icicle_cuda_runtime::memory::HostSlice;
use icicle_hash::keccak::build_keccak256_merkle_tree;

let mut config = TreeBuilderConfig::default();
config.arity = 2;
let height = 22;
let input_block_len = 136;
let leaves = vec![1u8; (1 << height) * input_block_len];
let mut digests = vec![0u64; merkle_tree_digests_len((height + 1) as u32, 2, 1)];

let leaves_slice = HostSlice::from_slice(&leaves);
let digests_slice = HostSlice::from_mut_slice(&mut digests);

build_keccak256_merkle_tree(leaves_slice, digests_slice, height, input_block_len, &config).unwrap();
```

In the example above, a binary tree of height 22 is being built. Each leaf is considered to be a 136 byte long array. The leaves and digests are aligned in a flat array. You can also use keccak512 in `build_keccak512_merkle_tree` function.