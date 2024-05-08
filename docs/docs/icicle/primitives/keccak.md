# Keccak

[Keccak](https://keccak.team/files/Keccak-implementation-3.2.pdf) is a cryptographic hash function designed by Guido Bertoni, Joan Daemen, Michaël Peeters, and Gilles Van Assche. It was selected as the winner of the NIST hash function competition, becoming the basis for the [SHA-3 standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf).

Keccak operates on a message input of any length and produces a fixed-size hash output. The hash function is built upon the sponge construction, which involves absorbing the input data followed by squeezing out the hash value.

At its core, Keccak consists of a permutation function operating on a state array. The permutation function employs a round function that operates iteratively on the state array. Each round consists of five main steps:

- **Theta:** This step introduces diffusion by performing a bitwise XOR operation between the state and a linear combination of its neighboring columns.
- **Rho:** This step performs bit rotation operations on each lane of the state array.
- **Pi:** This step rearranges the positions of the lanes in the state array.
- **Chi:** This step applies a nonlinear mixing operation to each lane of the state array.
- **Iota:** This step introduces a round constant to the state array.

## Using Keccak

ICICLE Keccak supports batch hashing, which can be utilized for constructing a merkle tree.

### Supported Bindings

- [Golang](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/golang/hash/keccak)
- [Rust](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-hash)