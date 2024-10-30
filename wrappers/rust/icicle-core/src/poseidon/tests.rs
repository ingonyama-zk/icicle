use crate::{
    hash::{HashConfig, Hasher},
    merkle::{MerkleProof, MerkleTree, MerkleTreeConfig},
    poseidon::{Poseidon, PoseidonHasher},
    test_utilities,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_runtime::{errors::eIcicleError, memory::HostSlice};
use std::mem;

pub fn check_poseidon_hash<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonHasher<F> + GenerateRandom<F>,
{
    let batch = 1 << 4;
    let domain_tag = F::Config::generate_random(1)[0];
    for t in [3, 5, 9, 12] {
        for domain_tag in [None, Some(&domain_tag)] {
            let inputs: Vec<F> = if domain_tag != None {
                F::Config::generate_random(batch * (t - 1))
            } else {
                F::Config::generate_random(batch * t)
            };
            let mut outputs_main = vec![F::zero(); batch];
            let mut outputs_ref = vec![F::zero(); batch];

            test_utilities::test_set_main_device();
            let poseidon_hasher_main = Poseidon::new::<F>(t as u32, domain_tag).unwrap();

            poseidon_hasher_main
                .hash(
                    HostSlice::from_slice(&inputs),
                    &HashConfig::default(),
                    HostSlice::from_mut_slice(&mut outputs_main),
                )
                .unwrap();

            test_utilities::test_set_ref_device();
            let poseidon_hasher_ref = Poseidon::new::<F>(t as u32, domain_tag).unwrap();

            poseidon_hasher_ref
                .hash(
                    HostSlice::from_slice(&inputs),
                    &HashConfig::default(),
                    HostSlice::from_mut_slice(&mut outputs_ref),
                )
                .unwrap();

            assert_eq!(outputs_main, outputs_ref);
        }
    }
}

pub fn check_poseidon_hash_sponge<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonHasher<F> + GenerateRandom<F>,
{
    for t in [3, 5, 9, 12] {
        let inputs: Vec<F> = F::Config::generate_random(t * 8 - 2);
        let mut outputs_main = vec![F::zero(); 1];
        let mut outputs_ref = vec![F::zero(); 1];

        test_utilities::test_set_main_device();
        let poseidon_hasher_main = Poseidon::new::<F>(t as u32, None /*domain_tag*/).unwrap();

        poseidon_hasher_main
            .hash(
                HostSlice::from_slice(&inputs),
                &HashConfig::default(),
                HostSlice::from_mut_slice(&mut outputs_main),
            )
            .unwrap();

        // Sponge poseidon is planned for v3.2. Not supported in v3.1
        test_utilities::test_set_ref_device();
        let poseidon_hasher_ref = Poseidon::new::<F>(t as u32, None /*domain_tag*/).unwrap();

        let err = poseidon_hasher_ref.hash(
            HostSlice::from_slice(&inputs),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut outputs_ref),
        );
        assert_eq!(err, Err(eIcicleError::InvalidArgument));
    }
}

pub fn check_poseidon_tree<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    let t = 9;
    let nof_layers = 4;
    let num_elements = 9_u32.pow(nof_layers);
    let mut leaves: Vec<F> = (0..num_elements)
        .map(|i| F::from_u32(i))
        .collect();

    let hasher = Poseidon::new::<F>(t as u32, None /*domain_tag*/).unwrap();
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
            true, /*=pruned */
            &MerkleTreeConfig::default(),
        )
        .unwrap();
    let _root = merkle_proof.get_root::<F>();
    let _path = merkle_proof.get_path::<F>();
    let (_leaf, _leaf_idx) = merkle_proof.get_leaf::<F>();

    let verification_valid = merkle_tree
        .verify(&merkle_proof)
        .unwrap();
    assert_eq!(verification_valid, true);
}
