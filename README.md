# ICICLE
 <div align="center">Icicle is a library for ZK acceleration using CUDA-enabled GPUs.</div>

                  
![image (4)](https://user-images.githubusercontent.com/2446179/223707486-ed8eb5ab-0616-4601-8557-12050df8ccf7.png)

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

> NOTE: _Support for BN254 and BLS12-377 are planned_

## Build and usage

> NOTE: [NVCC] is a prerequisite for building.

1. Define or select a curve for your application; we've provided a [template][CRV_TEMPLATE] for defining a curve
2. Include the curve in [`curve_config.cuh`][CRV_CONFIG]
3. Now you can build the ICICLE library using nvcc

```sh
mkdir -p build
nvcc -o build/<ENTER_DIR_NAME> ./icicle/appUtils/ntt/ntt.cu ./icicle/appUtils/msm/msm.cu ./icicle/appUtils/vector_manipulation/ve_mod_mult.cu ./icicle/primitives/projective.cu -lib -arch=native
```

### Testing the CUDA code

We are using [googletest] library for testing. To build and run the test suite for finite field and elliptic curve arithmetic, run from the `icicle` folder:

```sh
mkdir -p build
cmake -S . -B build
cmake --build build
cd build && ctest
```

### Rust Bindings

For convenience, we also provide rust bindings to the ICICLE library for the following primitives:

- MSM
- NTT
    - Forward NTT
    - Inverse NTT
- ECNTT
    - Forward ECNTT
    - Inverse NTT
- Scalar Vector Multiplication
- Point Vector Multiplication

A custom [build script][B_SCRIPT] is used to compile and link the ICICLE library. The environement variable `ARCH_TYPE` is used to determine which GPU type the library should be compiled for and it defaults to `native` when it is not set allowing the compiler to detect the installed GPU type.

> NOTE: A GPU must be detectable and therefore installed if the `ARCH_TYPE` is not set.

Once you have your parameters set, run:

```sh
cargo build --release
```

You'll find a release ready library at `target/release/libicicle_utils.rlib`.

To benchmark and test the functionality available in RUST, run:

```
cargo bench
cargo test -- --test-threads=1
```

The flag `--test-threads=1` is needed because currently some tests might interfere with one another inside the GPU.

### Example Usage

An example of using the Rust bindings library can be found in our [fast-danksharding implementation][FDI]

## Contributions

Join our [Discord Server](https://discord.gg/Y4SkbDf2Ff) and find us on the icicle channel. We will be happy to work together to support your use case and talk features, bugs and design.

### Hall of Fame

- [Robik](https://github.com/robik75), for his on-going support and mentorship 

## License

ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[BLS12-381]: ./icicle/curves/bls12_381.cuh
[NVCC]: https://docs.nvidia.com/cuda/#installation-guides
[CRV_TEMPLATE]: ./icicle/curves/curve_template.cuh
[CRV_CONFIG]: ./icicle/curves/curve_config.cuh
[B_SCRIPT]: ./build.rs
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
[LMIT]: ./LICENSE
[googletest]: https://github.com/google/googletest/

<!-- End Links -->
