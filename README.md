# ICICLE
 **<div align="center">Icicle is a library for ZK acceleration using CUDA-enabled GPUs.</div>**

                  
![image (4)](https://user-images.githubusercontent.com/2446179/223707486-ed8eb5ab-0616-4601-8557-12050df8ccf7.png)


<p align="center">
  <img src="https://github.com/ingonyama-zk/icicle/actions/workflows/main-build.yml/badge.svg?event=push" alt="Build status">
  <a href="https://discord.gg/EVVXTdt6DF">
    <img src="https://img.shields.io/discord/1063033227788423299?logo=discord" alt="Chat with us on Discord">
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=Ingo_zk">
    <img src="https://img.shields.io/twitter/follow/Ingo_zk?style=social&logo=twitter" alt="Follow us on Twitter">
  </a>
</p>

## Background

Zero Knowledge Proofs (ZKPs) are considered one of the greatest achievements of modern cryptography. Accordingly, ZKPs are expected to disrupt a number of industries and will usher in an era of trustless and privacy preserving services and infrastructure.

If we want ZK hardware today we have FPGAs or GPUs which are relatively inexpensive. However, the biggest selling point of GPUs is the software; we talk in particular about CUDA, which makes it easy to write code running on Nvidia GPUs, taking advantage of their highly parallel architecture. Together with the widespread availability of these devices, if we can get GPUs to work on ZK workloads, then we have made a giant step towards accessible and efficient ZK provers.

## Zero Knowledge on GPU

ICICLE is a CUDA implementation of general functions widely used in ZKP. ICICLE currently provides support for MSM, NTT, and ECNTT, with plans to support Hash functions soon.

### Supported primitives

- Fields
    - Scalars
    - Points
        - Projective: {x, y, z}
        - Affine: {x, y}
- Curves
    - [BLS12-381]
    - [BLS12-377]
    - [BN254]
    - [BW6-671]

## Build and usage


### Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 12.0 or newer.
- [CMake]((https://cmake.org/files/)), version 3.18 and above. Latest version is recommended.
- [GCC](https://gcc.gnu.org/install/download.html) version 9, latest version is recommended.
- Any Nvidia GPU (which supports CUDA Toolkit version 12.0 or above).

> [!NOTE] 
> It is possible to use CUDA 11 for cards which dont support CUDA 12, however we dont officially support this version and in the future there may be issues.

### Accessing Hardware

If you don't have access to a Nvidia GPU we have some options for you. 

Checkout [Google Colab](https://colab.google/). Google Colab offers a free [T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) instance and ICICLE can be used with it, reference this guide for setting up your [Google Colab workplace][GOOGLE-COLAB-ICICLE].

If you require more compute and have an interesting research project, we have [bounty and grant programs][GRANT_PROGRAM].


### Build systems

ICICLE has three build systems.

- [ICICLE core][ICICLE-CORE], C++ and CUDA
- [ICICLE Rust][ICICLE-RUST] bindings
- [ICICLE Golang][ICICLE-GO] bindings

ICICLE core always needs to be built as part of the other build systems as it contains the core ICICLE primitives implemented in CUDA. Reference these guides for the different build systems, [ICICLE core guide][ICICLE-CORE-README], [ICICLE Rust guide][ICICLE-RUST-README] and [ICICLE Golang guide][ICICLE-GO-README].

## Docker

We offer a simple Docker container so you can simply run ICICLE without setting everything up locally.

```
docker build -t <name_of_your_choice> .
docker run --gpus all -it <name_of_your_choice> /bin/bash
```

## Contributions

Join our [Discord Server][DISCORD] and find us on the icicle channel. We will be happy to work together to support your use case and talk features, bugs and design.

### Development Contributions

If you are changing code, please make sure to change your [git hooks path][HOOKS_DOCS] to the repo's [hooks directory][HOOKS_PATH] by running the following command:

```sh
git config core.hooksPath ./scripts/hooks
```

In case `clang-format` is missing on your system, you can install it  using the following command:

```sh
sudo apt install clang-format
```

This will ensure our custom hooks are run and will make it easier to follow our coding guidelines.

### Hall of Fame

- [Robik](https://github.com/robik75), for his ongoing support and mentorship
- [liuxiao](https://github.com/liuxiaobleach), for being a top notch bug smasher
- [gkigiermo](https://github.com/gkigiermo), for making it intuitive to use ICICLE in Google Colab.

## Help & Support

For help and support talk to our devs in our discord channel ["ICICLE"](https://discord.gg/EVVXTdt6DF) 


## License

ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[BLS12-381]: ./icicle/curves/bls12_381/supported_operations.cu
[BLS12-377]: ./icicle/curves/bls12_377/supported_operations.cu
[BN254]: ./icicle/curves/bn254/supported_operations.cu
[BW6-671]: ./icicle/curves/bw6_671/supported_operations.cu
[NVCC]: https://docs.nvidia.com/cuda/#installation-guides
[CRV_TEMPLATE]: ./icicle/curves/curve_template/
[CRV_CONFIG]: ./icicle/curves/index.cu
[B_SCRIPT]: ./build.rs
[LMIT]: ./LICENSE
[DISCORD]: https://discord.gg/Y4SkbDf2Ff
[googletest]: https://github.com/google/googletest/
[HOOKS_DOCS]: https://git-scm.com/docs/githooks
[HOOKS_PATH]: ./scripts/hooks/
[CMAKELISTS]: https://github.com/ingonyama-zk/icicle/blob/f0e6b465611227b858ec4590f4de5432e892748d/icicle/CMakeLists.txt#L28
[GOOGLE-COLAB-ICICLE]: https://github.com/gkigiermo/rust-cuda-colab
[GRANT_PROGRAM]: https://medium.com/@ingonyama/icicle-for-researchers-grants-challenges-9be1f040998e
[ICICLE-CORE]: ./icicle/
[ICICLE-RUST]: ./wrappers/rust/
[ICICLE-GO]: ./goicicle/
[ICICLE-CORE-README]: ./icicle/README.md
[ICICLE-RUST-README]: ./wrappers/rust/README.md
[ICICLE-GO-README]: ./goicicle/README.md


<!-- End Links -->
