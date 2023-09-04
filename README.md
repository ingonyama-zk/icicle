# ICICLE
 <div align="center">Icicle is a library for ZK acceleration using CUDA-enabled GPUs.</div>

                  
![image (4)](https://user-images.githubusercontent.com/2446179/223707486-ed8eb5ab-0616-4601-8557-12050df8ccf7.png)


<div align="center">

![Build status](https://github.com/ingonyama-zk/icicle/actions/workflows/main-build.yml/badge.svg)
![Discord server](https://img.shields.io/discord/1063033227788423299?label=Discord&logo=Discord&logoColor=%23&style=plastic)
![Follow us on twitter](https://img.shields.io/twitter/follow/Ingo_zk?style=social)

</div>

## Background

Zero Knowledge Proofs (ZKPs) are considered one of the greatest achievements of modern cryptography. Accordingly, ZKPs are expected to disrupt a number of industries and will usher in an era of trustless and privacy preserving services and infrastructure.

If we want ZK hardware today we have FPGAs or GPUs which are relatively inexpensive. However, the biggest selling point of GPUs is the software; we are referring specifically to CUDA, which simplifies writing code for Nvidia GPUs and taking advantage of their highly parallel architecture. Together with the widespread availability of these devices, if we can get GPUs to work on ZK workloads, then we have made a giant step towards accessible and efficient ZK provers.

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

## Build and usage

## Prerequisite dependencies
- A compatible Nvidia GPU and corresponding driver installed on your machine.
- CMake at least version 3.16.
- CUDA Toolkit version 12.0 or newer.
- GCC, version 9 or newer recommended.
- Optional: Golang or Rust should also be installed depending on the bindings you plan to use.

## Building 

1. Check here [CRV_PARAMS] if we support the curve required by your application (we currently don't support Pasta or Goldilocks).
2. If your curve isn't included in our [CRV_PARAMS], follow the instructions [here](#-Supporting-Additional-Curves).
3. Now you can build the ICICLE library using NVCC.

```sh
mkdir -p build
nvcc -o build/<binary_name> ./icicle/curves/index.cu -lib -arch=native
```

### Testing the CUDA code

We are using [googletest] library for testing. To build and run [the test suite](./icicle/README.md) for finite field and elliptic curve arithmetic, run from the `icicle` folder:

```sh
mkdir -p build
cmake -S . -B build
cmake --build build
cd build && ctest
```

NOTE: If you are using cmake versions < 3.24 add `-DCUDA_ARCH=<target_cumpute_arch>` to the command `cmake -S . -B build`

### Bindings
We support Golang and Rust bindings.

We offer extensive documentation on them, [Golang bindings][GOLANG_DOCS] and [Rust bindings][RUST_DOCS].

### Example Usage

Rust bindings - [fast-danksharding implementation][FDI] \
Golang bindings - [Gnark implementation][GNARKI]

# Supporting Additional Curves

Supporting additional curves can be done as follows:

Create a JSON file with the curve parameters. The curve is defined by the following parameters: 
- ``curve_name`` - e.g. ``bls12_381``.
- ``modulus_p`` - scalar field modulus (in decimal).
- ``bit_count_p`` - number of bits needed to represent `` modulus_p`` .
- ``limb_p`` - number of bytes needed to represent `` modulus_p``  (rounded).
- ``ntt_size`` - log of the maximal size subgroup of the scalar field.    
- ``modulus_q`` - base field modulus (in decimal).
- ``bit_count_q`` - number of bits needed to represent `` modulus_q`` .
- ``limb_q`` number of bytes needed to represent `` modulus_p``  (rounded).
- ``weierstrass_b`` - Weierstrass constant of the curve. 
- ``weierstrass_b_g2_re`` - Weierstrass real constant of the g2 curve. 
- ``weierstrass_b_g2_im`` - Weierstrass imaginary constant of the g2 curve. 
- ``gen_x`` - x-value of a generator element for the curve. 
- ``gen_y`` - y-value of a generator element for the curve.
- ``gen_x_re`` - real x-value of a generator element for the g2 curve. 
- ``gen_x_im`` - imaginary x-value of a generator element for the g2 curve. 
- ``gen_y_re`` - real y-value of a generator element for the g2 curve. 
- ``gen_y_im`` - imaginary y-value of a generator element for the g2 curve. 

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
    "weierstrass_b_g2_re":4,
    "weierstrass_b_g2_im":4,
    "gen_x" : 3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507,
    "gen_y" : 1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569,
    "gen_x_re" : 352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160,
    "gen_x_im" : 3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758,
    "gen_y_re" : 1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905,
    "gen_y_im" : 927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582
}
```

Save the parameters JSON file under the [``curve_parameters``][CRV_PARAMS] directory.

Then run the Python script ``new_curve_script.py `` from the root folder:

```
python3 ./curve_parameters/new_curve_script.py ./curve_parameters/bls12_381.json
```

The script does the following:
- Creates a folder in ``icicle/curves`` with the curve name, which contains all of the files needed for the supported operations in CUDA.
- Adds the curve's exported operations to ``icicle/curves/index.cu``. 
- Creates a file with the curve name in ``src/curves`` with the relevant objects for the curve. 
- Creates a test file with the curve name in ``src``. 
- You can now use our bindings with your new curves (please look at the bindings readme to make sure there are no binding specific extra steps).

## Contributions

Join our [Discord Server][DISCORD] and find us on the icicle channel. We will be happy to work together to support your use case and talk features, bugs and design.

### Development Contributions

If you are changing code, please make sure to change your [git hooks path][HOOKS_DOCS] to the repo's [hooks directory][HOOKS_PATH] by running the following command:

```sh
git config core.hooksPath ./scripts/hooks
```

This will ensure our custom hooks are run and will make it easier to follow our coding guidelines.

### Hall of Fame

- [Robik](https://github.com/robik75), for his on-going support and mentorship 

## License

ICICLE is distributed under the terms of the MIT License.

See [LICENSE-MIT][LMIT] for details.

<!-- Begin Links -->
[CRV_PARAMS]: ./curve_parameters/
[BLS12-381]: ./icicle/curves/bls12_381/supported_operations.cu
[BLS12-377]: ./icicle/curves/bls12_377/supported_operations.cu
[BN254]: ./icicle/curves/bn254/supported_operations.cu
[NVCC]: https://docs.nvidia.com/cuda/#installation-guides
[CRV_TEMPLATE]: ./icicle/curves/curve_template.cuh
[CRV_CONFIG]: ./icicle/curves/curve_config.cuh
[FDI]: https://github.com/ingonyama-zk/fast-danksharding
[GNARKI]: https://github.com/ingonyama-zk/gnark
[LMIT]: ./LICENSE
[DISCORD]: https://discord.gg/Y4SkbDf2Ff
[googletest]: https://github.com/google/googletest/
[HOOKS_DOCS]: https://git-scm.com/docs/githooks
[HOOKS_PATH]: ./scripts/hooks/
[RUST_DOCS]: ./src/READMEN.md
[GOLANG_DOCS]: ./goicicle/README.md

<!-- End Links -->
