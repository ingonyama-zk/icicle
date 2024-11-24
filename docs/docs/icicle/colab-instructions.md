# Run ICICLE on Google Colab

Google Colab lets you use a GPU free of charge, it's an Nvidia T4 GPU with 16 GB of memory, capable of running latest CUDA (tested on Cuda 12.2)
As Colab is able to interact with shell commands, a user can also install a framework and load git repositories into Colab space.

## Prepare Colab environment

First thing to do in a notebook is to set the runtime type to a T4 GPU.

- in the upper corner click on the dropdown menu and select "change runtime type"

![Change runtime](./static/img/colab_change_runtime.png)

- In the window select "T4 GPU" and press Save

![T4 GPU](./static/img/t4_gpu.png)

Installing Rust is rather simple, just execute the following command:

```sh
!apt install rustc cargo
```

To test the installation of Rust:

```sh
!rustc --version
!cargo --version
```

A successful installation will result in a rustc and cargo version print, a faulty installation will look like this:

```sh
/bin/bash: line 1: rustc: command not found
/bin/bash: line 1: cargo: command not found
```

Now we will check the environment:

```sh
!nvcc --version
!gcc --version
!cmake --version
!nvidia-smi
```

A correct environment should print the result with no bash errors for `nvidia-smi` command and result in a **Teslt T4 GPU** type:

```sh
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

cmake version 3.27.9

CMake suite maintained and supported by Kitware (kitware.com/cmake).
Wed Jan 17 13:10:18 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
| N/A   39C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## Cloning ICICLE and running test

Now we are ready to clone ICICE repository,

```sh
!git clone https://github.com/ingonyama-zk/icicle.git
```

We can browse the repository and run tests to check the runtime environment:

```sh
!ls -la
%cd /content/icicle
```

## Download CUDA backend

First let's create a backend directory

```sh
%cd /content
!rm -rf cuda_backend/
!mkdir cuda_backend
%cd cuda_backend
```

Download and extract a backend from [ICICLE released](https://github.com/ingonyama-zk/icicle/releases) backends
In this example we are using [ICICLE Cuda backend v3.1.0](https://github.com/ingonyama-zk/icicle/releases/download/v3.1.0/icicle_3_1_0-ubuntu22-cuda122.tar.gz)

```sh
!curl -O -L https://github.com/ingonyama-zk/icicle/releases/download/v3.1.0/icicle_3_1_0-ubuntu22-cuda122.tar.gz
!tar -xvf icicle_3_1_0-ubuntu22-cuda122.tar.gz
```

## Setting CUDA backend installation directory
Point colab to the extracted cuda backend using an [environment variable](https://github.com/ingonyama-zk/icicle/blob/f638e9d3056d2a5d6271a67ba4f63973a2ba2c1a/docs/docs/icicle/getting_started.md#backend-loading)

```sh
import os
os.envvar["ICICLE_BACKEND_INSTALL_DIR"] = "/content/cuda_backend/icicle"
```

## Fun with ICICLE

Let's run a test!
Navigate to icicle/wrappers/rust/icicle-curves/icicle-bn254 and run cargo test:

```sh
%cd /content/icicle/wrappers/rust/icicle-curves/icicle-bn254/
!cargo test --release -- ntt
```

:::note

Compiling the first time may take a while

:::

Test run should end like this:

```sh
running 9 tests
[WARNING] Defaulting to Ingonyama icicle-cuda-license-server at `5053@license.icicle.ingonyama.com`. For more information about icicle-cuda-license, please contact support@ingonyama.com.
[INFO] ICICLE backend loaded from $ICICLE_BACKEND_INSTALL_DIR=/content/cuda_backend/icicle
test ecntt::tests::test_ecntt::test_ecntt_batch ... ok
test ntt::tests::test_ntt ... ok
test ntt::tests::test_ntt_arbitrary_coset ... ok
test ntt::tests::test_ntt_batch ... ok
test ntt::tests::test_ntt_coset_from_subgroup ... ok
test ntt::tests::test_ntt_coset_interpolation_nm ... ok
test ecntt::tests::test_ecntt::test_ecntt ... ok
test ntt::tests::test_ntt_device_async ... ok
test ntt::tests::test_ntt_release_domain ... ok

test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 36 filtered out; finished in 42.71s
```

Viola, ICICLE in Colab!
