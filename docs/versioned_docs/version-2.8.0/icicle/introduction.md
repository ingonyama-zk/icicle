# Getting started with ICICLE

This guide is oriented towards developers who want to start writing code with the ICICLE libraries. If you just want to run your existing ZK circuits on GPU refer to [this guide](./integrations.md#using-icicle-integrated-provers) please.

## ICICLE repository overview

![ICICLE API overview](/img/apilevels.png)

The diagram above displays the general architecture of ICICLE and the API layers that exist. The CUDA API, which we also call ICICLE Core, is the lowest level and is comprised of CUDA kernels which implement all primitives such as MSM as well as C++ wrappers which expose these methods for different curves.

ICICLE Core compiles into a static library. This library can be used with our official Golang and Rust wrappers or linked with your C++ project. You can also implement a wrapper for it in any other language.

Based on this dependency architecture, the ICICLE repository has three main sections:

- [ICICLE Core](#icicle-core)
- [ICICLE Rust bindings](#icicle-rust-and-golang-bindings)
- [ICICLE Golang bindings](#icicle-rust-and-golang-bindings)

### ICICLE Core

[ICICLE Core](./core.md) is a library that directly works with GPU by defining CUDA kernels and algorithms that invoke them. It contains code for [fast field arithmetic](https://github.com/ingonyama-zk/icicle/tree/main/icicle/include/field/field.cuh), cryptographic primitives used in ZK such as [NTT](https://github.com/ingonyama-zk/icicle/tree/main/icicle/src/ntt/), [MSM](https://github.com/ingonyama-zk/icicle/tree/main/icicle/src/msm/), [Poseidon Hash](https://github.com/ingonyama-zk/icicle/tree/main/icicle/src/poseidon/), [Polynomials](https://github.com/ingonyama-zk/icicle/tree/main/icicle/src/polynomials/) and others.

ICICLE Core would typically be compiled into a static library and either used in a third party language such as Rust or Golang, or linked with your own C++ project.

### ICICLE Rust and Golang bindings

- [ICICLE Rust bindings](/icicle/rust-bindings)
- [ICICLE Golang bindings](/icicle/golang-bindings)

These bindings allow you to easily use ICICLE in a Rust or Golang project. Setting up Golang bindings requires a bit of extra steps compared to the Rust bindings which utilize the `cargo build` tool.

## Running ICICLE

This guide assumes that you have a Linux or Windows machine with an Nvidia GPU installed. If you don't have access to an Nvidia GPU you can access one for free on [Google Colab](https://colab.google/).

:::info note

ICICLE can only run on Linux or Windows. **MacOS is not supported**.

:::

### Prerequisites

- NVCC (version 12.0 or newer)
- cmake 3.18 and above
- GCC - version 9 or newer is recommended.
- Any Nvidia GPU
- Linux or Windows operating system.

#### Optional Prerequisites

- Docker, latest version.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

If you don't wish to install these prerequisites you can follow this tutorial using a [ZK-Container](https://github.com/ingonyama-zk/icicle/blob/main/Dockerfile) (docker container). To learn more about using ZK-Containers [read this](../ZKContainers.md).

### Setting up ICICLE and running tests

The objective of this guide is to make sure you can run the ICICLE Core, Rust and Golang tests. Achieving this will ensure you know how to setup ICICLE and run an ICICLE program. For simplicity, we will be using the ICICLE docker container as our environment, however, you may install the prerequisites on your machine and [skip](#icicle-core-1) the docker section.

#### Setting up environment with Docker

Lets begin by cloning the ICICLE repository:

```sh
git clone https://github.com/ingonyama-zk/icicle
```

We will proceed to build the docker image [found here](https://github.com/ingonyama-zk/icicle/blob/main/Dockerfile):

```sh
docker build -t icicle-demo .
docker run -it --runtime=nvidia --gpus all --name icicle_container icicle-demo
```

- `-it` runs the container in interactive mode with a terminal.
- `--gpus all` Allocate all available GPUs to the container. You can also specify which GPUs to use if you don't want to allocate all.
- `--runtime=nvidia` Use the NVIDIA runtime, necessary for GPU support.

To read more about these settings reference this [article](https://developer.nvidia.com/nvidia-container-runtime).

If you accidentally close your terminal and want to reconnect just call:

```sh
docker exec -it icicle_container bash
```

Lets make sure that we have the correct CUDA version before proceeding

```sh
nvcc --version
```

You should see something like this

```sh
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
```

Make sure the release version is at least 12.0.

#### ICICLE Core

ICICLE Core is found under [`<project_root>/icicle`](https://github.com/ingonyama-zk/icicle/tree/main/icicle). To build and run the tests first:

```sh
cd icicle
```

For this example, we are going to compile ICICLE for a `bn254` curve. However other compilation strategies are supported.

```sh
mkdir -p build
cmake -S . -B build -DCURVE=bn254 -DBUILD_TESTS=ON
cmake --build build -j
```

`-DBUILD_TESTS` option compiles the tests, without this flag `ctest` won't work.
`-DCURVE` option tells the compiler which curve to build. You can find a list of supported curves [here](https://github.com/ingonyama-zk/icicle/tree/main/icicle/cmake/CurvesCommon.cmake#L2).

The output in `build` folder should include the static libraries for the compiled curve.

To run the test

```sh
cd build/tests
ctest
```

#### ICICLE Rust

The rust bindings work by first compiling the CUDA static libraries as seen [here](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-curves/icicle-bn254/build.rs). The compilation of CUDA and the Rust library is all handled by the rust build toolchain.

Similar to ICICLE Core here we also have to compile per curve.

Lets compile curve `bn254`

```sh
cd wrappers/rust/icicle-curves/icicle-bn254
```

Now lets build our library

```sh
cargo build --release
```

This may take a couple of minutes since we are compiling both the CUDA and Rust code.

To run the tests

```sh
cargo test
```

We also include some benchmarks

```sh
cargo bench
```

#### ICICLE Golang

The Golang bindings require compiling ICICLE Core first. We supply a [build script](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/golang/build.sh) to help build what you need.

Script usage:

```sh
./build.sh [-curve=<curve>] [-field=<field>] [-hash=<hash>] [-cuda_version=<version>] [-g2] [-ecntt] [-devmode]

curve - The name of the curve to build or "all" to build all supported curves
field - The name of the field to build or "all" to build all supported fields
hash - The name of the hash to build or "all" to build all supported hashes
-g2 - Optional - build with G2 enabled 
-ecntt - Optional - build with ECNTT enabled
-devmode - Optional - build in devmode
```

:::note

If more than one curve or more than one field or more than one hash is supplied, the last one supplied will be built

:::

Once the library has been built, you can use and test the Golang bindings.

To test a specific curve, field or hash, change to it's directory and then run:

```sh
go test ./tests -count=1 -failfast -timeout 60m -p 2 -v
```

You will be able to see each test that runs, how long it takes and whether it passed or failed

### Running ICICLE examples

ICICLE examples can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/examples) these examples cover some simple use cases using C++, rust and golang.

Lets run one of our C++ examples, in this case the [MSM example](https://github.com/ingonyama-zk/icicle/blob/main/examples/c%2B%2B/msm/example.cu).

```sh
cd examples/c++/msm
./compile.sh
./run.sh
```

:::tip

Read through the compile.sh and CMakeLists.txt to understand how to link your own C++ project with ICICLE

:::

#### Running with Docker

In each example directory, ZK-container files are located in a subdirectory `.devcontainer`.

```sh
msm/
├── .devcontainer
   ├── devcontainer.json
   └── Dockerfile
```

Now lets build our docker file and run the test inside it. Make sure you have installed the [optional prerequisites](#optional-prerequisites).

```sh
docker build -t icicle-example-msm -f .devcontainer/Dockerfile .
```

Lets start and enter the container

```sh
docker run -it --rm --gpus all -v .:/icicle-example icicle-example-msm
```

Inside the container you can run the same commands:

```sh
./compile.sh
./run.sh
```

You can now experiment with our other examples, perhaps try to run a rust or golang example next.
