# Poseidon2

TODO update for V3

[Poseidon2](https://eprint.iacr.org/2023/323) is a recently released optimized version of Poseidon1. The two versions differ in two crucial points. First, Poseidon is a sponge hash function, while Poseidon2 can be either a sponge or a compression function depending on the use case. Secondly, Poseidon2 is instantiated by new and more efficient linear layers with respect to Poseidon. These changes decrease the number of multiplications in the linear layer by up to 90% and the number of constraints in Plonk circuits by up to 70%. This makes Poseidon2 currently the fastest arithmetization-oriented hash function without lookups.

## Using Poseidon2

(Danny pls update)

ICICLE Poseidon2 is implemented for GPU and parallelization is performed for each state.
We calculate multiple hash-sums over multiple pre-images in parallel, rather than going block by block over the input vector.

For example, for Poseidon2 of width 16, input rate 8, output elements 8 and input of size 1024 * 8, we would expect 1024 * 8 elements of output. Which means each input block would be of size 8, resulting in 1024 Poseidon2 hashes being performed.

### Supported Bindings

[`Rust`](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-core/src/poseidon2)

### Constants

Poseidon2 is also extremely customizable and using different constants will produce different hashes, security levels and performance results.

We support pre-calculated constants for each of the [supported curves](../libraries#supported-curves-and-operations). The constants can be found [here](https://github.com/ingonyama-zk/icicle/tree/main/icicle/include/poseidon2/constants) and are labeled clearly per curve `<curve_name>_poseidon2.h`.

You can also use your own set of constants as shown [here](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-fields/icicle-babybear/src/poseidon2/mod.rs#L290)

### Rust API

This is the most basic way to use the Poseidon2 API. See [examples/poseidon2](examples/rust/poseidon2)

```rust
let test_size = 4;
let poseidon = Poseidon2::new::<F>(test_size,None).unwrap();
let config = HashConfig::default();
let inputs = vec![F::one(); test_size];
let input_slice = HostSlice::from_slice(&inputs);
//digest is a single element
let out_init:F = F::zero();
let mut binding = [out_init];
let out_init_slice = HostSlice::from_mut_slice(&mut binding);

poseidon.hash(input_slice, &config, out_init_slice).unwrap();
println!("computed digest: {:?} ",out_init_slice.as_slice().to_vec()[0]);

```

## The Tree Builder

Similar to Poseidon1, you can use Poseidon2 in a tree builder. See [examples/poseidon2](examples/rust/poseidon2)

```rust
pub fn compute_binary_tree<F:FieldImpl>(
    mut test_vec: Vec<F>,
    leaf_size: u64,
    hasher: Hasher,
    compress: Hasher,
    mut tree_config: MerkleTreeConfig,
) -> MerkleTree
{
let tree_height: usize = test_vec.len().ilog2() as usize;
//just to be safe
tree_config.padding_policy = PaddingPolicy::ZeroPadding;
let layer_hashes: Vec<&Hasher> = std::iter::once(&hasher)
    .chain(std::iter::repeat(&compress).take(tree_height))
    .collect();
let vec_slice: &mut HostSlice<F> = HostSlice::from_mut_slice(&mut test_vec[..]);
let merkle_tree: MerkleTree = MerkleTree::new(&layer_hashes, leaf_size, 0).unwrap();

let _ = merkle_tree
    .build(vec_slice,&tree_config);
merkle_tree
}

//poseidon2 supports t=2,3,4,8,12,16,20,24. In this example we build a binary tree with Poseidon2 t=2.
let poseidon_state_size = 2; 
let leaf_size:u64 = 4;// each leaf is a 32 bit element 32/8 = 4 bytes

let mut test_vec = vec![F::from_u32(random::<u32>()); 1024* (poseidon_state_size as usize)];   
println!("Generated random vector of size {:?}", 1024* (poseidon_state_size as usize));
//to use later for merkle proof
let mut binding = test_vec.clone();
let test_vec_slice = HostSlice::from_mut_slice(&mut binding);
//define hash and compression functions (You can use different hashes here)
//note:"None" does not work with generics, use F= Fm31, Fbabybear etc
let hasher :Hasher = Poseidon2::new::<F>(poseidon_state_size.try_into().unwrap(),None).unwrap();
let compress: Hasher = Poseidon2::new::<F>((hasher.output_size()*2).try_into().unwrap(),None).unwrap();
//tree config
let tree_config = MerkleTreeConfig::default();
let merk_tree = compute_binary_tree(test_vec.clone(), leaf_size, hasher, compress,tree_config.clone());
println!("computed Merkle root {:?}", merk_tree.get_root::<F>().unwrap());

let random_test_index = rand::thread_rng().gen_range(0..1024*(leaf_size as usize));
print!("Generating proof for element {:?} at random test index {:?} ",test_vec[random_test_index], random_test_index);
let merkle_proof = merk_tree.get_proof::<F>(test_vec_slice, random_test_index.try_into().unwrap(), false, &tree_config).unwrap();

//actually should construct verifier tree :) 
assert!(merk_tree.verify(&merkle_proof).unwrap());
println!("\n Merkle proof verified successfully!");
``