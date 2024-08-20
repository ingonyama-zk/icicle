# ICICLE libraries

ICICLE is composed of two main logical parts:
1. [**ICICLE device library**](#icicle-device)
2. [**ICICLE template core library**](#icicle-core)

## ICICLE device

The ICICLE device library serves as an abstraction layer for interacting with various hardware devices. It provides a comprehensive interface for tasks such as setting the active device, querying device-specific information like free and total memory, determining the number of available devices, and managing memory allocation. Additionally, it offers functionality for copying data to and from devices, managing task queues (streams) for efficient device utilization, and abstracting the complexities of device management away from the user. 

See programmers guide for more details. [C++](./programmers_guide/cpp#device-management), [Rust](./programmers_guide/rust#device-management), [Go TODO](./programmers_guide/go)

## ICICLE Core

ICICLE Core is a template library written in C++ that implements fundamental cryptographic operations, including field and curve arithmetic, as well as higher-level APIs such as MSM and NTT.

The Core can be [instantiated](./getting_started) for different fields, curves, and other cryptographic components, allowing you to tailor it to your specific needs. You can link your application to one or more ICICLE libraries, depending on the requirements of your project. For example, you might only need the babybear library or a combination of babybear and a Merkle tree builder.


### Rust
Each library has a corresponding crate. See [programmers guide](./programmers_guide/general.md) for more details.

### Supported curves, fields and operations

#### Supported curves and operations

| Operation\Curve                                     | [bn254](https://neuromancer.sk/std/bn/bn254) | [bls12-377](https://neuromancer.sk/std/bls/BLS12-377) | [bls12-381](https://neuromancer.sk/std/bls/BLS12-381) | [bw6-761](https://eprint.iacr.org/2020/351) | grumpkin |
| --------------------------------------------------- | :------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :-----------------------------------------: | :------: |
| [MSM](./primitives/msm)                             |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| G2 MSM                                              |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [NTT](./primitives/ntt)                             |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| ECNTT                                               |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [Vector operations](./primitives/vec_ops)           |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| [Polynomials](./polynomials/overview)               |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [Poseidon](primitives/poseidon)                     |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| [Merkle Tree](primitives/poseidon#the-tree-builder) |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |

#### Supported fields and operations

| Operation\Field                           | [babybear](https://eprint.iacr.org/2023/824.pdf) | [Stark252](https://docs.starknet.io/documentation/architecture_and_concepts/Cryptography/p-value/) |
| ----------------------------------------- | :----------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| [Vector operations](./primitives/vec_ops) |                        ✅                         |                                                 ✅                                                  |
| [Polynomials](./polynomials/overview)     |                        ✅                         |                                                 ✅                                                  |
| [NTT](primitives/ntt)                     |                        ✅                         |                                                 ✅                                                  |
| Extension Field                           |                        ✅                         |                                                 ❌                                                  |

#### Supported hashes

| Hash   |  Sizes   |
| ------ | :------: |
| Keccak | 256, 512 |

## Backend
Each backend must implement the device API interface.
Each backend may implement
- One or more ICICLE library. For example implement only bn254 curve. 
- One or more APIs in this library. For example MSM only.

See [CUDA backend](./install_cuda_backend.md) and [Build Your Own Backend](./build_your_own_backend.md) for more info about implementing a backend.
