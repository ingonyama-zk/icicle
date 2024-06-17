# Poseidon

[Poseidon](https://eprint.iacr.org/2019/458.pdf) is a popular hash in the ZK ecosystem primarily because its optimized to work over large prime fields, a common setting for ZK proofs, thereby minimizing the number of multiplicative operations required.

Poseidon has also been specifically designed to be efficient when implemented within ZK circuits, Poseidon uses far less constraints compared to other hash functions like Keccak or SHA-256 in the context of ZK circuits.

Poseidon has been used in many popular ZK protocols such as Filecoin and [Plonk](https://drive.google.com/file/d/1bZZvKMQHaZGA4L9eZhupQLyGINkkFG_b/view?usp=drive_open).

Our implementation of Poseidon is implemented in accordance with the optimized [Filecoin version](https://spec.filecoin.io/algorithms/crypto/poseidon/).

Lets understand how Poseidon works.

## Initialization

Poseidon starts with the initialization of its internal state, which is composed of the input elements and some pre-generated constants. An initial round constant is added to each element of the internal state. Adding the round constants ensures the state is properly mixed from the beginning.

This is done to prevent collisions and to prevent certain cryptographic attacks by ensuring that the internal state is sufficiently mixed and unpredictable.

![Poseidon initialization of internal state added with pre-generated round constants](https://github.com/ingonyama-zk/icicle/assets/122266060/52257f5d-6097-47c4-8f17-7b6449b9d162)

## Applying full and partial rounds

To generate a secure hash output, the algorithm goes through a series of "full rounds" and "partial rounds" as well as transformations between these sets of rounds in the following order:

```First full rounds -> apply S-box and Round constants -> partial rounds -> Last full rounds -> Apply S-box```

### Full rounds

![Full round iterations consisting of S box operations, adding round constants, and a Full MDS matrix multiplication](https://github.com/ingonyama-zk/icicle/assets/122266060/e4ce0e98-b90b-4261-b83e-3cd8cce069cb)

**Uniform Application of S-box:** In full rounds, the S-box (a non-linear transformation) is applied uniformly to every element of the hash function's internal state. This ensures a high degree of mixing and diffusion, contributing to the hash function's security. The functions S-box involves raising each element of the state to a certain power denoted by `α` a member of the finite field defined by the prime `p`; `α` can be different depending on the implementation and user configuration.

**Linear Transformation:** After applying the S-box, a linear transformation is performed on the state. This involves multiplying the state by a MDS (Maximum Distance Separable) Matrix. which further diffuses the transformations applied by the S-box across the entire state.

**Addition of Round Constants:** Each element of the state is then modified by adding a unique round constant. These constants are different for each round and are precomputed as part of the hash function's initialization. The addition of round constants ensures that even minor changes to the input produce significant differences in the output.

### Partial Rounds

![Partial round iterations consisting of selective S box operation, adding a round constant and performing an MDS multiplication with a sparse matrix](https://github.com/ingonyama-zk/icicle/assets/122266060/e8c198b4-7aa4-4b4d-9ec4-604e39e07692)

**Selective Application of S-Box:** Partial rounds apply the S-box transformation to only one element of the internal state per round, rather than to all elements. This selective application significantly reduces the computational complexity of the hash function without compromising its security. The choice of which element to apply the S-box to can follow a specific pattern or be fixed, depending on the design of the hash function.

**Linear Transformation and Round Constants:** A linear transformation is performed and round constants are added. The linear transformation in partial rounds can be designed to be less computationally intensive (this is done by using a sparse matrix) than in full rounds, further optimizing the function's efficiency.

The user of Poseidon can often choose how many partial or full rounds he wishes to apply; more full rounds will increase security but degrade performance. The choice and balance is highly dependent on the use case.

## Using Poseidon

ICICLE Poseidon is implemented for GPU and parallelization is performed for each element of the state rather than for each state.
What that means is we calculate multiple hash-sums over multiple pre-images in parallel, rather than going block by block over the input vector.

So for Poseidon of arity 2 and input of size 1024 * 2, we would expect 1024 elements of output. Which means each block would be of size 2 and that would result in 1024 Poseidon hashes being performed.

### Supported Bindings

[`Rust`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-core/src/poseidon)

### Constants

Poseidon is extremely customizable and using different constants will produce different hashes, security levels and performance results.

We support pre-calculated and optimized constants for each of the [supported curves](../core#supported-curves-and-operations).The constants can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/icicle/include/poseidon/constants) and are labeled clearly per curve `<curve_name>_poseidon.h`.

If you wish to generate your own constants you can use our python script which can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/icicle/include/poseidon/constants/generate_parameters.py).

Prerequisites:

- Install python 3
- `pip install poseidon-hash`
- `pip install galois==0.3.7`
- `pip install numpy`

You will then need to modify the following values before running the script.

```python
# Modify these
arity = 11 # we support arity 2, 4, 8 and 11.
p = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001 # bls12-381
# p = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001 # bls12-377
# p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 # bn254
# p = 0x1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001 # bw6-761
prime_bit_len = 255
field_bytes = 32

...

# primitive_element = None
primitive_element = 7 # bls12-381
# primitive_element = 22 # bls12-377
# primitive_element = 5 # bn254
# primitive_element = 15 # bw6-761
```

We only support `alpha = 5` so if you want to use another alpha for S-box please reach out on discord or open a github issue.

### Rust API

This is the most basic way to use the Poseidon API.

```rust
let test_size = 1 << 10;
let arity = 2u32;
let ctx = get_default_device_context();
let constants = load_optimized_poseidon_constants::<F>(arity, &ctx).unwrap();
let config = PoseidonConfig::default();

let inputs = vec![F::one(); test_size * arity as usize];
let outputs = vec![F::zero(); test_size];
let mut input_slice = HostOrDeviceSlice::on_host(inputs);
let mut output_slice = HostOrDeviceSlice::on_host(outputs);

poseidon_hash_many::<F>(
    &mut input_slice,
    &mut output_slice,
    test_size as u32,
    arity as u32,
    &constants,
    &config,
)
.unwrap();
```

The `PoseidonConfig::default()` can be modified, by default the inputs and outputs are set to be on `Host` for example.

```rust
impl<'a> Default for PoseidonConfig<'a> {
    fn default() -> Self {
        let ctx = get_default_device_context();
        Self {
            ctx,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            input_is_a_state: false,
            aligned: false,
            loop_state: false,
            is_async: false,
        }
    }
}
```

In the example above `load_optimized_poseidon_constants::<F>(arity, &ctx).unwrap();` is used which will load the correct constants based on arity and curve. Its possible to [generate](#constants) your own constants and load them.

```rust
let ctx = get_default_device_context();
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    let constants_file = PathBuf::from(cargo_manifest_dir)
        .join("tests")
        .join(format!("{}_constants.bin", field_prefix));
    let mut constants_buf = vec![];
    File::open(constants_file)
        .unwrap()
        .read_to_end(&mut constants_buf)
        .unwrap();

    let mut custom_constants = vec![];
    for chunk in constants_buf.chunks(field_bytes) {
        custom_constants.push(F::from_bytes_le(chunk));
    }

    let custom_constants = create_optimized_poseidon_constants::<F>(
        arity as u32,
        &ctx,
        full_rounds_half,
        partial_rounds,
        &mut custom_constants,
    )
    .unwrap();
```

## The Tree Builder

The tree builder allows you to build Merkle trees using Poseidon.

You can define both the tree's `height` and its `arity`. The tree `height` determines the number of layers in the tree, including the root and the leaf layer. The `arity` determines how many children each internal node can have.

```rust
let height = 20;
let arity = 2;
let leaves = vec![F::one(); 1 << (height - 1)];
let mut digests = vec![F::zero(); merkle_tree_digests_len(height, arity)];

let mut leaves_slice = HostOrDeviceSlice::on_host(leaves);

let ctx = get_default_device_context();
let constants = load_optimized_poseidon_constants::<F>(arity, &ctx).unwrap()

let mut config = TreeBuilderConfig::default();
config.keep_rows = 1;
build_poseidon_merkle_tree::<F>(&mut leaves_slice, &mut digests, height, arity, &constants, &config).unwrap();

println!("Root: {:?}", digests[0..1][0]);
```

Similar to Poseidon, you can also configure the Tree Builder `TreeBuilderConfig::default()`

- `keep_rows`: The number of rows which will be written to output, 0 will write all rows.
- `are_inputs_on_device`: Have the inputs been loaded to device memory ?
- `is_async`: Should the TreeBuilder run asynchronously? `False` will block the current CPU thread. `True` will require you call `cudaStreamSynchronize` or `cudaDeviceSynchronize` to retrieve the result.

### Benchmarks

We ran the Poseidon tree builder on:

**CPU**: 12th Gen Intel(R) Core(TM) i9-12900K/

**GPU**: RTX 3090 Ti

**Tree height**: 30 (2^29 elements)

The benchmarks include copying data from and to the device.

| Rows to keep parameter      | Run time, Icicle | Supranational PC2
| ----------- | ----------- | -----------
| 10          | 9.4 seconds       |    13.6 seconds
| 20          | 9.5 seconds       |    13.6 seconds
| 29          | 13.7 seconds       |    13.6 seconds
