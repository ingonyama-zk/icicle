# ICICLE
 **<div align="center">Icicle is a library for ZK acceleration using CUDA-enabled GPUs.</div>**

                  
![image (4)](https://user-images.githubusercontent.com/2446179/223707486-ed8eb5ab-0616-4601-8557-12050df8ccf7.png)


<p align="center">
  <img src="https://github.com/ingonyama-zk/icicle/actions/workflows/main-build.yml/badge.svg" alt="Build status">
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

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu) version 12.0 or newer.
- [CMake]((https://cmake.org/files/)), version 3.18 and above. Latest version is recommended.
- [GCC](https://gcc.gnu.org/install/download.html) version 9, latest version is recommended.
- Any Nvidia GPU (which supports CUDA Toolkit version 12.0 or above).

Note: It is possible to use CUDA 11 for cards which dont support CUDA 12, however we dont offically support this version and in the future there may be issues.

### Accessing Hardware

If you don't have access to a Nvidia GPU we have some options for you. 

Checkout [Google Colab](https://colab.google/). Google Colab offers a free [T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) instance and ICICLE can be used with it, refrence this guide for setting up your [Google Colab workplace][GOOGLE-COLAB-ICICLE].

If you require more compute and have an interesting research project, we have a [bounties and grants program][GRANT_PROGRAM].


### Build systems

ICICLE has three build systems.

- [ICICLE core][ICICLE-CORE], C++ and CUDA
- [ICICLE Rust][ICICLE-RUST] bindings
- [ICICLE Golang][ICICLE-GO] bindings

ICICLE core always needs to be build as part of the other build systems as it containse the core ICICLE primitves implemeted in Cuda. Refrence these guides for the different build systems, [ICICLE core guide][ICICLE-CORE-README], [ICICLE Rust guide][ICICLE-RUST-README] and [ICICLE Golang guide][ICICLE-GO-README].

# Supporting Additional Curves

Supporting additional curves can be done as follows:

Create a JSON file with the curve parameters. The curve is defined by the following parameters: 
- ``curve_name`` - e.g. ``bls12_381``.
- ``modulus_p`` - scalar field modulus (in decimal).
- ``bit_count_p`` - number of bits needed to represent `` modulus_p`` .
- ``limb_p`` - number of (32-bit) limbs needed to represent `` modulus_p`` (rounded up).
- ``ntt_size`` - log of the maximal size subgroup of the scalar field.
- ``modulus_q`` - base field modulus (in decimal).
- ``bit_count_q`` - number of bits needed to represent `` modulus_q`` .
- ``limb_q`` - number of (32-bit) limbs needed to represent `` modulus_q`` (rounded up).
- ``weierstrass_b`` - `b` of the curve in Weierstrauss form.
- ``weierstrass_b_g2_re`` - real part of the `b` value in of the g2 curve in Weierstrass form.
- ``weierstrass_b_g2_im`` - imaginary part of the `b` value in of the g2 curve in Weierstrass form.
- ``gen_x`` - `x` coordinate of a generator element for the curve.
- ``gen_y`` - `y` coordinate of a generator element for the curve.
- ``gen_x_re`` - real part of the `x` coordinate of generator element for the g2 curve.
- ``gen_x_im`` - imaginary part of the `x` coordinate of generator element for the g2 curve.
- ``gen_y_re`` - real part of the `y` coordinate of generator element for the g2 curve.
- ``gen_y_im`` - imaginary part of the `y` coordinate of generator element for the g2 curve.
- ``nonresidue`` - nonresidue, or `i^2`, or `u^2` - square of the element that generates quadratic extension field of the base field.

Here's an example for BLS12-381.
```
{
    "curve_name" : "bls12_381", 
    "modulus_p" : 52435875175126190479447740508185965837690552500527637822603658699938581184513,
    "bit_count_p" : 255,
    "limb_p" :  8,
    "ntt_size" : 32,
    "modulus_q" : 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787,
    "bit_count_q" : 381,
    "limb_q" : 12,
    "weierstrass_b" : 4,
    "weierstrass_b_g2_re" : 4,
    "weierstrass_b_g2_im" : 4,
    "gen_x" : 3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507,
    "gen_y" : 1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569,
    "gen_x_re" : 352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160,
    "gen_x_im" : 3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758,
    "gen_y_re" : 1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905,
    "gen_y_im" : 927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582,
    "nonresidue" : -1
}
```

Save the parameters JSON file under the ``curve_parameters`` directory.

Then run the Python script ``new_curve_script.py `` from the root folder:

```
python3 ./curve_parameters/new_curve_script.py ./curve_parameters/bls12_381.json
```

The script does the following:
- Creates a folder in ``icicle/curves`` with the curve name, which contains all of the files needed for the supported operations in cuda.
- Adds the curve's exported operations to ``icicle/curves/index.cu``. 
- Creates a file with the curve name in ``src/curves`` with the relevant objects for the curve. 
- Creates a test file with the curve name in ``src``. 

Also files from ``./icicle/curves/<curve_name>/supported_operations.cu`` should be added individually to ``add_library`` section of [``./icicle/CMakeLists.txt``][CMAKELISTS]

Testing the new curve could be done by running the tests in ``tests_curve_name`` (e.g. ``tests_bls12_381``).

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
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
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
