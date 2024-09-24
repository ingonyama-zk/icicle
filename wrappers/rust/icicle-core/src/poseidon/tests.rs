use crate::{
    hash::{HashConfig, Hasher},
    merkle::{MerkleProof, MerkleTree, MerkleTreeConfig},
    poseidon::{create_poseidon_hasher, initialize_default_poseidon_constants, PoseidonHasher},
    traits::FieldImpl,
};
use icicle_runtime::memory::HostSlice;
use std::mem;

pub fn check_poseidon_hash<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    let batch = 1 << 10;
    let arity = 3;
    let mut inputs = vec![F::one(); batch * arity];
    let mut outputs = vec![F::zero(); arity];

    initialize_default_poseidon_constants::<F>().unwrap();
    let poseidon_hasher = create_poseidon_hasher::<F>(arity as u32).unwrap();

    poseidon_hasher
        .hash(
            HostSlice::from_slice(&mut inputs),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut outputs),
        )
        .unwrap();

    // TODO real test for both CPU and CUDA
}

pub fn check_poseidon_tree<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    let arity = 9;
    let nof_layers = 4;
    let num_elements = (1 << nof_layers) * arity;
    let mut leaves: Vec<F> = (0..num_elements)
        .map(|i| F::from_u32(i))
        .collect();

    let hasher = create_poseidon_hasher::<F>(arity as u32).unwrap();
    let layer_hashes: Vec<&Hasher> = (0..nof_layers)
        .map(|_| &hasher)
        .collect();
    let merkle_tree = MerkleTree::new(&layer_hashes[..], mem::size_of::<F>() as u64, 0).unwrap();
    merkle_tree
        .build(HostSlice::from_slice(&mut leaves), &MerkleTreeConfig::default())
        .unwrap();

    let leaf_idx_to_open = num_elements >> 1;
    let merkle_proof: MerkleProof = merkle_tree
        .get_proof(
            HostSlice::from_slice(&leaves),
            leaf_idx_to_open as u64,
            &MerkleTreeConfig::default(),
        )
        .unwrap();
    let root = merkle_proof.get_root::<F>();
    let path = merkle_proof.get_path::<F>();
    let (leaf, leaf_idx) = merkle_proof.get_leaf::<F>();
    println!("root = {:?}", root);
    println!("path = {:?}", path);
    println!("leaf = {:?}, leaf_idx = {}", leaf, leaf_idx);

    let verification_valid = merkle_tree
        .verify(&merkle_proof)
        .unwrap();
    assert_eq!(verification_valid, true);

    // TODO real test for both CPU and CUDA
}
