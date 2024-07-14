# Poseidon2

[Poseidon2](https://eprint.iacr.org/2023/323) is a recently released optimized version of Poseidon1. The two versions differ in two crucial points. First, Poseidon is a sponge hash function, while Poseidon2 can be either a sponge or a compression function depending on the use case. Secondly, Poseidon2 is instantiated by new and more efficient linear layers with respect to Poseidon. These changes decrease the number of multiplications in the linear layer by up to 90% and the number of constraints in Plonk circuits by up to 70%. This makes Poseidon2 currently the fastest arithmetization-oriented hash function without lookups.


## Using Poseidon2

ICICLE Poseidon2 is implemented for GPU and parallelization is performed for each state.
We calculate multiple hash-sums over multiple pre-images in parallel, rather than going block by block over the input vector.

So for Poseidon2 of width 16, input rate 8, output elements 8 and input of size 1024 * 8, we would expect 1024 * 8 elements of output. Which means each input block would be of size 8 and that would result in 1024 Poseidon2 hashes being performed.

### Supported Bindings

[`Rust`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-core/src/poseidon2)

### Constants

Poseidon2 is also extremely customizable and using different constants will produce different hashes, security levels and performance results.

We support pre-calculated constants for each of the [supported curves](../core#supported-curves-and-operations).The constants can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/icicle/include/poseidon2/constants) and are labeled clearly per curve `<curve_name>_poseidon2.h`.

One can also use his own set of constants as shown [here](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-fields/icicle-babybear/src/poseidon2/mod.rs#L290)

### Rust API

This is the most basic way to use the Poseidon2 API.

```rust
let test_size = 1 << 10;
let width = 16;
let rate = 8;
let ctx = get_default_device_context();
let poseidon = Poseidon2::load(width, rate, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();
let config = HashConfig::default();

let inputs = vec![F::one(); test_size * rate as usize];
let outputs = vec![F::zero(); test_size];
let mut input_slice = HostOrDeviceSlice::on_host(inputs);
let mut output_slice = HostOrDeviceSlice::on_host(outputs);

poseidon.hash_many::<F>(
    &mut input_slice,
    &mut output_slice,
    test_size as u32,
    rate as u32,
    8, // Output length
    &config,
)
.unwrap();
```

In the example above `Poseidon2::load(width, rate, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();` is used which will load the correct constants based on width and curve. Here, the default MDS matrices and diffusion are used. If you want to get a Plonky3 compliant version - set them to `MdsType::Plonky` and `DiffusionStrategy::Montgomery` respectively.

## The Tree Builder

Similar to Poseidon1, you can use Poseidon2 in a tree builder.

```rust
use icicle_bn254::tree::Bn254TreeBuilder;
use icicle_bn254::poseidon2::Poseidon2;

let mut config = TreeBuilderConfig::default();
let arity = 2;
config.arity = arity as u32;
let input_block_len = arity;
let leaves = vec![F::one(); (1 << height) * arity];
let mut digests = vec![F::zero(); merkle_tree_digests_len((height + 1) as u32, arity as u32, 1)];

let leaves_slice = HostSlice::from_slice(&leaves);
let digests_slice = HostSlice::from_mut_slice(&mut digests);

let ctx = device_context::DeviceContext::default();
let hash = Poseidon2::load(arity, arity, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();

let mut config = TreeBuilderConfig::default();
config.keep_rows = 5;
Bn254TreeBuilder::build_merkle_tree(
    leaves_slice,
    digests_slice,
    height,
    input_block_len,
    &hash,
    &hash,
    &config,
)
.unwrap();
```
