# Using Poseidon2 API

Sanity checks with standard sage output from attached sage script. The sage script originally from [Horizen labs code](https://github.com/HorizenLabs/poseidon2/blob/055bde3f4782731ba5f5ce5888a440a94327eaf3/poseidon2_rust_params.sage#L1) has been modified to print in standard format.

::: info

Note that the digest element of the Poseidon2 hash api is the output state of Poseidon2 at index 1.

:::

Run the example with
```
cargo run --release
```

* The first example runs Poseidon2 API for the babybear and m31 fields for state sizes $t=2,3,4,8,12,16,20,24$ and prints the results. You can compare it with the results from the attached sage code.
* The second example builds a binary merkle tree with Poseidon2 $t=2$ using the Merkle tree builder API and verifies the path for an arbitrarily chosen leaf.