# Supporting Additional Curves

We understand the need for ZK developers to use different curves, some common some more exotic. For this reason we designed ICICLE to allow developers to add any curve they desire.

## ICICLE Core

ICICLE core is very generic by design so all algorithms and primitives are designed to work based of configuration files [selected during compile](https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/curve_config.cuh) time. This is why we compile ICICLE Core per curve.

To add support a new curve you must create a new file under [`icicle/curves`](https://github.com/ingonyama-zk/icicle/tree/main/icicle/curves). The file should be named `<curve_name>_params.cuh`.

We also require some changes to [`curve_config.cuh`](https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/curve_config.cuh#L16-L29), we need to add a new curve id.

```
...

#define BN254     1
#define BLS12_381 2
#define BLS12_377 3
#define BW6_761   4
#define GRUMPKIN  5
#define <curve_name> 6

...
```

Make sure to modify the [rest of the file](https://github.com/ingonyama-zk/icicle/blob/4beda3a900eda961f39af3a496f8184c52bf3b41/icicle/curves/curve_config.cuh#L16-L29) accordingly.

Finally we must modify the [`make` file](https://github.com/ingonyama-zk/icicle/blob/main/icicle/CMakeLists.txt#L64) to make sure we can compile our new curve.

```
set(SUPPORTED_CURVES bn254;bls12_381;bls12_377;bw6_761;<curve_name>)
```

### Poseidon

If you want your curve to implement Poseidon hash function or a tree builder, you will need to pre-calculate the optimized parameters for it.  
Copy [constants_template.h](https://github.com/ingonyama-zk/icicle/blob/main/icicle/appUtils/poseidon/constants/constants_template.h) into `icicle/appUtils/poseidon/constants/<CURVE>_poseidon.h`. Run the [constants generation script](https://dev.ingonyama.com/icicle/primitives/poseidon#constants). The script will print the number of partial rounds and generate `constants.bin` file. Use `xxd -i constants.bin` to parse the file into C declarations. Copy the `unsigned char constants_bin[]` contents inside your new file. Repeat this process for arities 2, 4, 8 and 11.

After you've generated the constants, add your curve in this [SUPPORTED_CURVES_WITH_POSEIDON](https://github.com/ingonyama-zk/icicle/blob/main/icicle/CMakeLists.txt#L72) in the `CMakeLists.txt`.

## Bindings

In order to support a new curves in the binding libraries you first must support it in ICICLE core.

### Rust

Go to [rust wrappers folder](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-curves) and copy `icicle-curve-template` to a new folder named `icicle-<curve_name>`.

Find all the occurrences of `<CURVE>` placeholder inside the crate. (You can use `Ctrl+Shift+F` in VS Code or `grep -nr "<CURVE>"` in bash). You will then need to replace each occurrence with your new curve name.

#### Limbs

Go to [curve file](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-curves/icicle-curve-template/src/curve.rs) and set `SCALAR_LIMBS`, `BASE_LIMBS` and `G2_BASE_LIMBS` (if G2 is needed) to a minimum number of `u64` required to store a single scalar field / base field element respectively.  
e.g. for bn254, scalar field is 254 bit so `SCALAR_LIMBS` is set to 4.

#### Primitives

If your curve doesn't support any of the primitives (ntt/msm/poseidon/merkle tree/), or you simply don't won't to include it, just remove a corresponding module from `src` and then from `lib.rs`

#### G2

If your curve doesn't support G2 - remove all the code under `#[cfg(feature = "g2")]` and remove the feature from `Cargo.toml` and `build.rs`.

After this is done, add your new crate in the [global Cargo.toml](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/Cargo.toml).

### Golang

Golang is WIP in v1, coming soon. Please checkout a previous [release v0.1.0](https://github.com/ingonyama-zk/icicle/releases/tag/v0.1.0) for golang bindings.
