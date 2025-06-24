use crate::{
    field::PrimeField,
    hash::{HashConfig, Hasher},
    merkle::{MerkleProof, MerkleTree, MerkleTreeConfig},
    poseidon2::{Poseidon2, Poseidon2Hasher},
    traits::GenerateRandom,
};
use icicle_runtime::{
    memory::{IntoIcicleSlice, IntoIcicleSliceMut},
    test_utilities,
};
use std::mem;

pub fn check_poseidon2_hash<F: PrimeField>()
where
    F: Poseidon2Hasher + GenerateRandom,
{
    let batch = 1 << 4;
    let domain_tag = F::generate_random(1)[0];
    for t in [2, 3, 4, 8, 12, 16, 20, 24] {
        let large_field = mem::size_of::<F>() > 4;
        let skip_case = large_field && t > 4; // In Poseidon2 there is no support for large fields when t > 4.
        if skip_case {
            continue;
        };
        for domain_tag in [None, Some(&domain_tag)] {
            let inputs: Vec<F> = if domain_tag != None {
                F::generate_random(batch * (t - 1))
            } else {
                F::generate_random(batch * t)
            };
            let mut outputs_main = vec![F::zero(); batch];
            let mut outputs_ref = vec![F::zero(); batch];

            test_utilities::test_set_main_device();
            let poseidon_hasher_main = Poseidon2::new::<F>(t as u32, domain_tag).unwrap();

            poseidon_hasher_main
                .hash(
                    inputs.into_slice(),
                    &HashConfig::default(),
                    outputs_main.into_slice_mut(),
                )
                .unwrap();

            test_utilities::test_set_ref_device();
            let poseidon_hasher_ref = Poseidon2::new::<F>(t as u32, domain_tag).unwrap();

            poseidon_hasher_ref
                .hash(
                    inputs.into_slice(),
                    &HashConfig::default(),
                    outputs_ref.into_slice_mut(),
                )
                .unwrap();

            assert_eq!(outputs_main, outputs_ref, "Domain tag {:?}, t {:?}", domain_tag, t);
        }
    }
}

// TODO uncomment once Poseidon2 sponge function is ready
pub fn check_poseidon2_hash_sponge<F: PrimeField>()
where
    F: Poseidon2Hasher + GenerateRandom,
{
    for t in [2, 3, 4, 8, 12, 16, 20, 24] {
        let large_field = mem::size_of::<F>() > 4;
        let skip_case = large_field && t > 4; // In Poseidon2 there is no support for large fields when t > 4.
        if skip_case {
            continue;
        };
        let inputs: Vec<F> = F::generate_random(t * 8 - 2);
        let mut outputs_main = vec![F::zero(); 1];
        let mut outputs_ref = vec![F::zero(); 1];

        test_utilities::test_set_main_device();
        let poseidon_hasher_main = Poseidon2::new::<F>(t as u32, None /*domain_tag*/).unwrap();

        let main_device_err = poseidon_hasher_main
            .hash(
                inputs.into_slice(),
                &HashConfig::default(),
                outputs_main.into_slice_mut(),
            )
            .unwrap();

        test_utilities::test_set_ref_device();
        let poseidon_hasher_ref = Poseidon2::new::<F>(t as u32, None /*domain_tag*/).unwrap();

        let ref_device_err = poseidon_hasher_ref
            .hash(
                inputs.into_slice(),
                &HashConfig::default(),
                outputs_ref.into_slice_mut(),
            )
            .unwrap();

        assert_eq!(main_device_err, ref_device_err);
    }
}

pub fn check_poseidon2_hash_multi_device<F: PrimeField>()
where
    F: Poseidon2Hasher + GenerateRandom,
{
    let t = 4; // t=4 is for Poseidon hash (t is the paper's terminology)
    let inputs: Vec<F> = F::generate_random(t);
    let mut outputs_main_0 = vec![F::zero(); 1];
    let mut outputs_main_1 = vec![F::zero(); 1];
    let mut outputs_ref = vec![F::zero(); 1];

    test_utilities::test_set_ref_device();
    let poseidon_hasher_ref = Poseidon2::new::<F>(t as u32, None /*domain_tag*/).unwrap();

    poseidon_hasher_ref
        .hash(
            inputs.into_slice(),
            &HashConfig::default(),
            outputs_ref.into_slice_mut(),
        )
        .unwrap();

    // initialize hasher on 2 devices
    test_utilities::test_set_main_device_with_id(0);
    let poseidon_hasher_main_dev_0 = Poseidon2::new::<F>(t as u32, None /*domain_tag*/).unwrap();
    test_utilities::test_set_main_device_with_id(1);
    let poseidon_hasher_main_dev_1 = Poseidon2::new::<F>(t as u32, None /*domain_tag*/).unwrap();

    // test device 1
    poseidon_hasher_main_dev_1
        .hash(
            inputs.into_slice(),
            &HashConfig::default(),
            outputs_main_1.into_slice_mut(),
        )
        .unwrap();
    assert_eq!(outputs_ref, outputs_main_1);

    // test device 0
    test_utilities::test_set_main_device_with_id(0);
    poseidon_hasher_main_dev_0
        .hash(
            inputs.into_slice(),
            &HashConfig::default(),
            outputs_main_0.into_slice_mut(),
        )
        .unwrap();
    assert_eq!(outputs_ref, outputs_main_0);
}

pub fn check_poseidon2_tree<F: PrimeField>()
where
    F: Poseidon2Hasher,
{
    let t = 4;
    let nof_layers = 4;
    let num_elements = 4_u32.pow(nof_layers);
    let leaves: Vec<F> = (0..num_elements)
        .map(|i| F::from_u32(i))
        .collect();

    let hasher = Poseidon2::new::<F>(t as u32, None /*domain_tag*/).unwrap();
    let layer_hashes: Vec<&Hasher> = (0..nof_layers)
        .map(|_| &hasher)
        .collect();
    let merkle_tree = MerkleTree::new(layer_hashes.as_slice(), mem::size_of::<F>() as u64, 0).unwrap();
    merkle_tree
        .build(leaves.into_slice(), &MerkleTreeConfig::default())
        .unwrap();

    let leaf_idx_to_open = num_elements >> 1;
    let merkle_proof: MerkleProof = merkle_tree
        .get_proof(
            leaves.into_slice(),
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
