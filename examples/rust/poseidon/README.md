# ICICLE example: Poseidon hash in Rust

## Key-Takeaway

`ICICLE` provides Rust bindings to CUDA-accelerated C++ implementation of [Poseidon hash](https://github.com/ingonyama-zk/ingopedia/blob/9f602aae051100ee4c60791db5c6fa23d01e1f79/src/hashzk.md?plain=1#L30).

## Best Practices

In order to save time and setting up prerequisites manually, we recommend running this example in our [ZKContainer](../../ZKContainer.md).

## Usage

```rust
poseidon::poseidon_hash_many<F>(
    input: &mut HostOrDeviceSlice<F>, // a pointer to a vector of input data
    output: &mut HostOrDeviceSlice<F>, // a pointer to a vector of output data,
    number_of_states: u32, // number of input blocks of size `arity`
    arity: u32, // the arity of the hash function
    constants: &PoseidonConstants<F>, // Poseidon constants
    config: &PoseidonConfig, // config used to specify extra arguments of the Poseidon
) -> IcicleResult<()>
```

In this example we use the `BN254`, `BLS12377` and `BLS12381` fields.

## What's in this example

1. Load optimized Poseidon hash constants.
2. Generate custom Poseidon hash constants.
3. Configure Poseidon hash to use inputs and outputs on device
4. Execute Poseidon Hash on-device

Running the example:

```sh
cargo run --release
```

You can add the `--feature profile` flag to measure times of both ICICLE and arkworks.

> [!NOTE]
> The default size is 2^20. You can change this by passing the `--size <size>` option. To change the size to 2^23, run the example like this:

```sh
cargo run --release -- -s 23
```

## Benchmarks

TODO