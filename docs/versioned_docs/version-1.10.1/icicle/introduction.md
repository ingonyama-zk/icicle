# Getting started with ICICLE

This guide is oriented towards developers who want to start writing code with the ICICLE libraries. If you just want to run your existing ZK circuits on GPU refer to [this guide](./integrations.md#using-icicle-integrations) please.

## ICICLE repository overview

![ICICLE API overview](/img/apilevels.png)

The diagram above displays the general architecture of ICICLE and the API layers that exist. The CUDA API, which we also call ICICLE Core, is the lowest level and is comprised of CUDA kernels which implement all primitives such as MSM as well as C++ wrappers which expose these methods for different curves.

ICICLE Core compiles into a static library. This library can be used with our official Golang and Rust wrappers or you can implement a wrapper for it in any language.

Based on this dependency architecture, the ICICLE repository has three main sections, each of which is independent from the other.

- ICICLE core
- ICICLE Rust bindings
- ICICLE Golang bindings

### ICICLE Core

[ICICLE core](https://github.com/ingonyama-zk/icicle/tree/main/icicle) contains all the low level CUDA code implementing primitives such as [points](https://github.com/ingonyama-zk/icicle/tree/main/icicle/primitives) and [MSM](https://github.com/ingonyama-zk/icicle/tree/main/icicle/appUtils/msm). There also exists higher level C++ wrappers to expose the low level CUDA primitives ([example](https://github.com/ingonyama-zk/icicle/blob/c1a32a9879a7612916e05aa3098f76144de4109e/icicle/appUtils/msm/msm.cu#L1)).

ICICLE Core would typically be compiled into a static library and used in a third party language such as Rust or Golang.

### ICICLE Rust and Golang bindings

- [ICICLE Rust bindings](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust)
- [ICICLE Golang bindings](https://github.com/ingonyama-zk/icicle/tree/main/goicicle)

These bindings allow you to easily use ICICLE in a Rust or Golang project. Setting up Golang bindings requires a bit of extra steps compared to the Rust bindings which utilize the `cargo build` tool.

## Running ICICLE

This guide assumes that you have a Linux or Windows machine with an Nvidia GPU installed. If you don't have access to an Nvidia GPU you can access one for free on [Google Colab](https://colab.google/).

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

The objective of this guide is to make sure you can run the ICICLE Core, Rust and Golang tests. Achieving this will ensure you know how to setup ICICLE and run a ICICLE program. For simplicity, we will be using the ICICLE docker container as our environment, however, you may install the prerequisites on your machine and follow the same commands in your terminal.

#### Setting up our environment

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

We are going to compile ICICLE for a specific curve

```sh
mkdir -p build
cmake -S . -B build -DCURVE=bn254 -DBUILD_TESTS=ON
cmake --build build
```

`-DBUILD_TESTS=ON` compiles the tests, without this flag `ctest` won't work.
`-DCURVE=bn254` tells the compiler which curve to build. You can find a list of supported curves [here](https://github.com/ingonyama-zk/icicle/tree/main/icicle/curves).

The output in `build` folder should include the static libraries for the compiled curve.

:::info

Make sure to only use `-DBUILD_TESTS=ON` for running tests as the archive output will only be available when `-DBUILD_TESTS=ON` is not supplied.

:::

To run the test

```sh
cd build
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

Golang is WIP in v1, coming soon. Please checkout a previous [release v0.1.0](https://github.com/ingonyama-zk/icicle/releases/tag/v0.1.0) for golang bindings.

### Running ICICLE examples

ICICLE examples can be found [here](https://github.com/ingonyama-zk/icicle-examples) these examples cover some simple use cases using C++, rust and golang.

In each example directory, ZK-container files are located in a subdirectory `.devcontainer`.

```sh
msm/
├── .devcontainer
   ├── devcontainer.json
   └── Dockerfile
```

Lets run one of our C++ examples, in this case the [MSM example](https://github.com/ingonyama-zk/icicle-examples/blob/main/c%2B%2B/msm/example.cu).

Clone the repository

```sh
git clone https://github.com/ingonyama-zk/icicle-examples.git
cd icicle-examples
```

Enter the test directory

```sh
cd c++/msm
```

Now lets build our docker file and run the test inside it. Make sure you have installed the [optional prerequisites](#optional-prerequisites).

```sh
docker build -t icicle-example-msm -f .devcontainer/Dockerfile .
```

Lets start and enter the container

```sh
docker run -it --rm --gpus all -v .:/icicle-example icicle-example-msm
```

to run the example

```sh
rm -rf build
mkdir -p build
cmake -S . -B build
cmake --build build
./build/example
```

You can now experiment with our other examples, perhaps try to run a rust or golang example next.

## Writing new bindings for ICICLE

Since ICICLE Core is written in CUDA / C++ its really simple to generate static libraries. These static libraries can be installed on any system and called by higher level languages such as Golang.

static libraries can be loaded into memory once and used by multiple programs, reducing memory usage and potentially improving performance. They also allow you to separate functionality into distinct modules so your static library may need to compile only specific features that you want to use.

Lets review the Golang bindings since its a pretty verbose example (compared to rust which hides it pretty well) of using static libraries. Golang has a library named `CGO` which can be used to link static libraries. Here's a basic example on how you can use cgo to link these libraries:

```go
/*
#cgo LDFLAGS: -L/path/to/shared/libs -lbn254 -lbls12_381 -lbls12_377 -lbw6_671
#include "icicle.h" // make sure you use the correct header file(s)
*/
import "C"

func main() {
  // Now you can call the C functions from the ICICLE libraries.
  // Note that C function calls are prefixed with 'C.' in Go code.

  out := (*C.BN254_projective_t)(unsafe.Pointer(p))
  in := (*C.BN254_affine_t)(unsafe.Pointer(affine))

  C.projective_from_affine_bn254(out, in)
}
```

The comments on the first line tell `CGO` which libraries to import as well as which header files to include. You can then call methods which are part of the static library and defined in the header file, `C.projective_from_affine_bn254` is an example.

If you wish to create your own bindings for a language of your choice we suggest you start by investigating how you can call static libraries.

### ICICLE Adapters

One of the core ideas behind ICICLE is that developers can gradually accelerate their provers. Many protocols are written using other cryptographic libraries and completely replacing them may be complex and time consuming.

Therefore we offer adapters for various popular libraries, these adapters allow us to convert points and scalars between different formats defined by various libraries. Here is a list:

Golang adapters:

- [Gnark crypto adapter](https://github.com/ingonyama-zk/iciclegnark)
