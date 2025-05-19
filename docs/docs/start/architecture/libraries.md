# ICICLE libraries

ICICLE is composed of two main logical parts:
1. [**ICICLE device library**](#icicle-device)
2. [**ICICLE template core library**](#icicle-core)

## ICICLE device

The ICICLE device library serves as an abstraction layer for interacting with various hardware devices. It provides a comprehensive interface for tasks such as setting the active device, querying device-specific information like free and total memory, determining the number of available devices, and managing memory allocation. Additionally, it offers functionality for copying data to and from devices, managing task queues (streams) for efficient device utilization, and abstracting the complexities of device management away from the user.

See programmers guide for more details. [C++](start/programmers_guide/cpp.md#device-management), [Rust](start/programmers_guide/rust.md#device-management), [Go](start/programmers_guide/go.md) 

## ICICLE Core

ICICLE Core is a template library written in C++ that implements fundamental cryptographic operations, including field and curve arithmetic, as well as higher-level APIs such as MSM and NTT.

The Core can be [instantiated](/start/programmers_guide/build_from_source) for different fields, curves, and other cryptographic components, allowing you to tailor it to your specific needs. You can link your application to one or more ICICLE libraries, depending on the requirements of your project. For example, you might only need the babybear library or a combination of babybear and a Merkle tree builder.


### Rust
Each library has a corresponding crate. See [programmers guide](start/programmers_guide/general.md) for more details. 

### Supported curves, fields and operations

#### Supported curves and operations

| Operation\Curve                           | [bn254](https://neuromancer.sk/std/bn/bn254) | [bls12-377](https://neuromancer.sk/std/bls/BLS12-377) | [bls12-381](https://neuromancer.sk/std/bls/BLS12-381) | [bw6-761](https://eprint.iacr.org/2020/351) | grumpkin |
| ----------------------------------------- | :------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :-----------------------------------------: | :------: |
| [MSM](api/cpp/msm)                   |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| G2 MSM                                    |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [NTT](api/cpp/ntt)                   |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| ECNTT                                     |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [Vector operations](api/cpp/vec_ops) |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| [Polynomials](./polynomials/overview)     |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [Poseidonapi(cpp/hash#poseidon)      |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| [Poseidon2api(cpp/hash#poseidon2)    |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |

#### Supported fields and operations

| Operation\Field                           | [babybear](https://eprint.iacr.org/2023/824.pdf) | [Stark252](https://docs.starknet.io/architecture-and-concepts/cryptography/#stark-field) |  m31  |  Koalabear  | Goldilocks |
| ----------------------------------------- | :----------------------------------------------: | :------------------------------------------------------------------------------------------------: | :---: | :---------: | :--------: |
| [Vector operations](api/cpp/vec_ops) |                        ✅                         |                                                 ✅                                                  |  ✅   |     ✅      |     ✅     |
| [Polynomials](./polynomials/overview)     |                        ✅                         |                                                 ✅                                                  |  ❌   |     ✅      |     ✅     |
| [NTTapi(cpp/ntt)                     |                        ✅                         |                                                 ✅                                                  |  ❌   |     ✅      |     ✅     |
| Extension Field                           |                        ✅                         |                                                 ❌                                                  |  ✅   |     ✅      |     ✅     |
| [Poseidonapi(cpp/hash#poseidon)      |                        ✅                         |                                                 ✅                                                  |  ✅   |     ✅      |     ❌     |
| [Poseidon2api(cpp/hash#poseidon2)    |                        ✅                         |                                                 ✅                                                  |  ✅   |     ✅      |     ✅    |


### Misc

| Operation                                 |                 Description                  |
| ----------------------------------------- | :------------------------------------------: |
| [Keccakapi(cpp/hash#keccak-and-sha3) |       supporting 256b and 512b digest        |
| [SHA3api(cpp/hash#keccak-and-sha3)   |       supporting 256b and 512b digest        |
| [Blake2sapi(cpp/hash#blake2s)        |                digest is 256b                |
| [Blake3api(cpp/hash#blake3)        |                digest is 256b                |
| [Merkle-Treeapi(cpp/merkle)          | works with any combination of hash functions |






## Backend
Each backend must implement the device API interface.
Each backend may implement
- One or more ICICLE library. For example implement only bn254 curve.
- One or more APIs in this library. For example MSM only.

See [CUDA backend](/start/architecture/install_gpu_backend) and [Build Your Own Backend](/start/architecture/build_your_own_backend.md) for more info about implementing a backend.
