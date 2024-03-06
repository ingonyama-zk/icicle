# Supporting Additional Curves

We understand the need for ZK developers to use different curves, some common some more exotic. For this reason we designed ICICLE to allow developers to add any curve they desire.

## ICICLE Core

ICICLE core is very generic by design so all algorithms and primitives are designed to work based of configuration files [selected during compile](https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/curve_config.cuh) time. This is why we compile ICICLE Core per curve.

To add support for a new curve you must create a new file under [`icicle/curves`](https://github.com/ingonyama-zk/icicle/tree/main/icicle/curves). The file should be named `<curve_name>_params.cuh`.

### Adding curve_name_params.cuh

Start by copying `bn254_params.cuh` contents in your params file. Params should include:
 - **fq_config** - parameters of the Base field.
    - **limbs_count** - `ceil(field_byte_size / 4)`.
    - **modulus_bit_count** - bit-size of the modulus.
    - **num_of_reductions** - the number of times to reduce in reduce function. Use 2 if not sure.
    - **modulus** - modulus of the field.
    - **modulus_2** - modulus * 2.
    - **modulus_4** - modulus * 4. 
    - **neg_modulus** - negated modulus. 
    - **modulus_wide** - modulus represented as a double-sized integer.
    - **modulus_squared** - modulus**2 represented as a double-sized integer.
    - **modulus_squared_2** - 2 * modulus**2 represented as a double-sized integer.
    - **modulus_squared_4** - 4 * modulus**2 represented as a double-sized integer.
    - **m** - value used in multiplication. Can be computed as `2**(2*modulus_bit_count) // modulus`. 
    - **one** - multiplicative identity. 
    - **zero** - additive identity. 
    - **montgomery_r** - `2 ** M % modulus` where M is a closest (larger than) bitsize multiple of 32. E.g. 384 or 768 for bls and bw curves respectively
    - **montgomery_r_inv** - `2 ** (-M) % modulus`
 - **fp_config** - parameters of the Scalar field.
    Same as fq_config, but with additional arguments:
    - **omegas_count** - [two-adicity](https://cryptologie.net/article/559/whats-two-adicity/) of the field. And thus the maximum size of NTT.
    - **omegas** - an array of omegas for NTTs. An array of size `omegas_count`. The ith element is equal to `1.nth_root(2**(2**(omegas_count-i)))`.
    - **inv** - an array of inverses of powers of two in a field. Ith element is equal to `(2 ** (i+1)) ** -1`.
 - **G1 generators points** - affine coordinates of the generator point.
 - **G2 generators points** - affine coordinates of the extension generator. Remove these if `G2` is not supported.
 - **Weierstrass b value** - base field element equal to value of `b` in the curve equation.
 - **Weierstrass b value G2** - base field element equal to value of `b` for the extension. Remove this if `G2` is not supported.
 
 :::note

 All the params are not in Montgomery form.
 
 :::
 
 :::note

 To convert number values into `storage` type you can use the following python function

```python
import struct

def unpack(x, field_size):
    return ', '.join(["0x" + format(x, '08x') for x in struct.unpack('I' * (field_size) // 4, int(x).to_bytes(field_size, 'little'))])
```

:::

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
set(SUPPORTED_CURVES bn254;bls12_381;bls12_377;bw6_761;grumpkin;<curve_name>)
```

### Adding Poseidon support

If you want your curve to implement a Poseidon hash function or a tree builder, you will need to pre-calculate its optimized parameters.  
Copy [constants_template.h](https://github.com/ingonyama-zk/icicle/blob/main/icicle/appUtils/poseidon/constants/constants_template.h) into `icicle/appUtils/poseidon/constants/<CURVE>_poseidon.h`. Run the [constants generation script](https://dev.ingonyama.com/icicle/primitives/poseidon#constants). The script will print the number of partial rounds and generate a `constants.bin` file. Use `xxd -i constants.bin` to parse the file into C declarations. Copy the `unsigned char constants_bin[]` contents inside your new file. Repeat this process for arities 2, 4, 8 and 11.

After you've generated the constants, add your curve in this [SUPPORTED_CURVES_WITH_POSEIDON](https://github.com/ingonyama-zk/icicle/blob/main/icicle/CMakeLists.txt#L72) in the `CMakeLists.txt`.

## Bindings

In order to support a new curve in the binding libraries you first must support it in ICICLE core.

### Rust

Go to [rust curves folder](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-curves) and copy `icicle-curve-template` to a new folder named `icicle-<curve_name>`.

Find all the occurrences of `<CURVE>` placeholder inside the crate. (You can use `Ctrl+Shift+F` in VS Code or `grep -nr "<CURVE>"` in bash). You will then need to replace each occurrence with your new curve name.

#### Limbs

Go to your curve's `curve.rs` file and set `SCALAR_LIMBS`, `BASE_LIMBS` and `G2_BASE_LIMBS` (if G2 is needed) to a minimum number of `u64` required to store a single scalar field / base field element respectively.  
e.g. for bn254, scalar field is 254 bit so `SCALAR_LIMBS` is set to 4.

#### Primitives

If your curve doesn't support some of the primitives (ntt/msm/poseidon/merkle tree/), or you simply don't want to include it, just remove a corresponding module from `src` and then from `lib.rs`

#### G2

If your curve doesn't support G2 - remove all the code under `#[cfg(feature = "g2")]` and remove the feature from [Cargo.toml](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-curves/icicle-bn254/Cargo.toml#L29) and [build.rs](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-curves/icicle-bn254/build.rs#L15).

After this is done, add your new crate in the [global Cargo.toml](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/Cargo.toml).

### Golang

Golang is WIP in v1, coming soon. Please checkout a previous [release v0.1.0](https://github.com/ingonyama-zk/icicle/releases/tag/v0.1.0) for golang bindings.
