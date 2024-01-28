# ICICLE

**<div align="center">ICICLE is a library for ZK acceleration using CUDA-enabled GPUs.</div>**

<p align="center">
  <img alt="ICICLE" width="300" height="300" src="https://user-images.githubusercontent.com/2446179/223707486-ed8eb5ab-0616-4601-8557-12050df8ccf7.png"/>
</p>
<p align="center">
  <a href="https://discord.gg/EVVXTdt6DF">
    <img src="https://img.shields.io/discord/1063033227788423299?logo=discord" alt="Chat with us on Discord">
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=Ingo_zk">
    <img src="https://img.shields.io/twitter/follow/Ingo_zk?style=social&logo=twitter" alt="Follow us on Twitter">
  </a>
  <img src="https://img.shields.io/badge/Machines%20running%20ICICLE-544-lightblue" alt="Machines running ICICLE">
</p>

## Background

Zero Knowledge Proofs (ZKPs) are considered one of the greatest achievements of modern cryptography. Accordingly, ZKPs are expected to disrupt a number of industries and will usher in an era of trustless and privacy preserving services and infrastructure.

We believe GPUs are as important for ZK as for AI.

- GPUs are a perfect match for ZK compute - around 97% of ZK protocol runtime is parallel by nature.
- GPUs are simple for developers to use and scale compared to other hardware platforms.
- GPUs are extremely competitive in terms of power / performance and price (3x cheaper).
- GPUs are popular and readily available.

## Getting Started

ICICLE is a CUDA implementation of general functions widely used in ZKP.

> [!NOTE]
> Developers: We highly recommend reading our [documentation]

> [!TIP]
> Try out ICICLE by running some [examples] using ICICLE in C++ and our Rust bindings 

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
- [ICICLE Rust][ICICLE-RUST] bindings, requires [Rust](https://www.rust-lang.org/) version 1.70 and above
- [ICICLE Golang][ICICLE-GO] bindings, requires [Go](https://go.dev/) version 1.20 and above

ICICLE core always needs to be built as part of the other build systems as it contains the core ICICLE primitives implemented in CUDA. Reference these guides for the different build systems, [ICICLE core guide][ICICLE-CORE-README], [ICICLE Rust guide][ICICLE-RUST-README] and [ICICLE Golang guide][ICICLE-GO-README].

### Compiling ICICLE

Running ICICLE via Rust bindings is highly recommended and simple:
- Clone this repo
  - go to our [Rust bindings][ICICLE-RUST]
  - Enter a [curve](./wrappers/rust/icicle-curves) implementation
  - run `cargo build --release` to build or `cargo test -- --test-threads=1` to build and execute tests

In any case you would want to compile and run core icicle c++ tests, just follow these setps:
- Clone this repo
  - go to [ICICLE core][ICICLE-CORE]
  - execute the small [script](https://github.com/ingonyama-zk/icicle/tree/main/icicle#running-tests) to compile via cmake and run c++ and cuda tests

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

You will also need to install [codespell](https://github.com/codespell-project/codespell?tab=readme-ov-file#installation) to check for typos.

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
[BLS12-381]: ./icicle/curves/
[BLS12-377]: ./icicle/curves/
[BN254]: ./icicle/curves/
[BW6-671]: ./icicle/curves/
[NVCC]: https://docs.nvidia.com/cuda/#installation-guides
[LMIT]: ./LICENSE
[DISCORD]: https://discord.gg/Y4SkbDf2Ff
[googletest]: https://github.com/google/googletest/
[HOOKS_DOCS]: https://git-scm.com/docs/githooks
[HOOKS_PATH]: ./scripts/hooks/
[CMAKELISTS]: https://github.com/ingonyama-zk/icicle/blob/f0e6b465611227b858ec4590f4de5432e892748d/icicle/CMakeLists.txt#L28
[GOOGLE-COLAB-ICICLE]: https://dev.ingonyama.com/icicle/colab-instructions
[GRANT_PROGRAM]: https://medium.com/@ingonyama/icicle-for-researchers-grants-challenges-9be1f040998e
[ICICLE-CORE]: ./icicle/
[ICICLE-RUST]: ./wrappers/rust/
[ICICLE-GO]: ./goicicle/
[ICICLE-CORE-README]: ./icicle/README.md
[ICICLE-RUST-README]: ./wrappers/rust/README.md
[ICICLE-GO-README]: ./goicicle/README.md
[documentation]: https://dev.ingonyama.com/icicle/overview
[examples]: ./examples/

<!-- End Links -->
