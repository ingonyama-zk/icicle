# ICICLE libraries

TODO : review this page and see the hashes really exist

## ICICLE device

The ICICLE device library serves as an abstraction layer for interacting with various hardware devices. It provides a comprehensive interface for tasks such as setting the active device, querying device-specific information like free and total memory, determining the number of available devices, and managing memory allocation. Additionally, it offers functionality for copying data to and from devices, managing task queues (streams) for efficient device utilization, and abstracting the complexities of device management away from the user. 

TODO update links

[C++ device APIs](https://github.com/ingonyama-zk/icicle/blob/yshekel/V3/icicle_v3/include/icicle/runtime.h)

[Rust icicle-runtime crate](https://github.com/ingonyama-zk/icicle/tree/yshekel/V3/wrappers/rust_v3/icicle-runtime)

TODO Golang

## ICICLE Core

ICICLE Core is a template library written in C++ that implements fundamental cryptographic operations, including field and curve arithmetic, as well as higher-level APIs such as MSM and NTT.

The Core can be [instantiated](./getting_started) for different fields, curves, and other cryptographic components, allowing you to tailor it to your specific needs. You can link your application to one or more ICICLE libraries, depending on the requirements of your project. For example, you might only need the babybear library or a combination of babybear and a Merkle tree builder.

:::note
Every instantiation is essentially a dispatch layer that is calling backend apis based on the current thread device.
:::


### Rust
Each library has a corresponding crate. See [examples](./using_icicle.md) for more details.

### Supported curves, fields and operations

#### Supported curves and operations

| Operation\Curve                                     | [bn254](https://neuromancer.sk/std/bn/bn254) | [bls12-377](https://neuromancer.sk/std/bls/BLS12-377) | [bls12-381](https://neuromancer.sk/std/bls/BLS12-381) | [bw6-761](https://eprint.iacr.org/2020/351) | grumpkin |
| --------------------------------------------------- | :------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :-----------------------------------------: | :------: |
| [MSM](./primitives/msm)                             |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| G2                                                  |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [NTT](./primitives/ntt)                             |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| ECNTT                                               |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| VecOps                                              |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| [Polynomials](./polynomials/overview)               |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ❌     |
| [Poseidon](primitives/poseidon)                     |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |
| [Merkle Tree](primitives/poseidon#the-tree-builder) |                      ✅                       |                           ✅                           |                           ✅                           |                      ✅                      |    ✅     |

#### Supported fields and operations

| Operation\Field                       | [babybear](https://eprint.iacr.org/2023/824.pdf) | [Stark252](https://docs.starknet.io/documentation/architecture_and_concepts/Cryptography/p-value/) |
| ------------------------------------- | :----------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| VecOps                                |                        ✅                         |                                                 ✅                                                  |
| [Polynomials](./polynomials/overview) |                        ✅                         |                                                 ✅                                                  |
| [NTT](primitives/ntt)                 |                        ✅                         |                                                 ✅                                                  |
| Extension Field                       |                        ✅                         |                                                 ❌                                                  |

#### Supported hashes

| Hash   |  Sizes   |
| ------ | :------: |
| Keccak | 256, 512 |

## Backend
Each backend must implement the device API interface.
Each backend may implement
- One or more ICICLE library. For example implement only bn254 curve. 
- One or more APIs in this library. For example MSM only.

See [Build Your Own Backend](./build_your_own_backend.md) for more details.