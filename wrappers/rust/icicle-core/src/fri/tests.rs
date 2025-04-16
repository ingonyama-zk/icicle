use std::mem::ManuallyDrop;

use super::FriMerkleTree;
use crate::{
    fri::{
        fri_merkle_tree_prove, fri_merkle_tree_verify, fri_proof::FriProofTrait,
        fri_transcript_config::FriTranscriptConfig, FriConfig, FriProof,
    }, hash::Hasher, merkle::{MerkleProof, MerkleProofData}, traits::{FieldImpl, GenerateRandom, Serialization}
};
use icicle_runtime::{memory::DeviceVec, stream::IcicleStream};
use icicle_runtime::{memory::HostSlice, test_utilities};

pub fn check_fri<F: FieldImpl>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_hash: &Hasher,
) where
    <F as FieldImpl>::Config: FriMerkleTree<F> + GenerateRandom<F>,
{
    let check = || {
        const SIZE: u64 = 1 << 10;
        let fri_config = FriConfig::default();
        let scalars = F::Config::generate_random(SIZE as usize);

        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let merkle_tree_min_layer_to_store = 0;

        let fri_proof = fri_merkle_tree_prove::<F>(
            &fri_config,
            &transcript_config,
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            merkle_tree_min_layer_to_store,
        )
        .unwrap();
        let valid = fri_merkle_tree_verify(
            &fri_config,
            &transcript_config,
            &fri_proof,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
        )
        .unwrap();
        assert!(valid);
    };
    test_utilities::test_set_main_device();
    check();
    test_utilities::test_set_ref_device();
    check();
}

pub fn check_fri_on_device<F: FieldImpl>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_hash: &Hasher,
) where
    <F as FieldImpl>::Config: FriMerkleTree<F> + GenerateRandom<F>,
{
    let check = || {
        const SIZE: u64 = 1 << 10;
        let stream = IcicleStream::create().unwrap();
        let mut fri_config = FriConfig::default();
        fri_config.is_async = true;
        fri_config.stream_handle = *stream;
        let scalars = F::Config::generate_random(SIZE as usize);

        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let merkle_tree_min_layer_to_store = 0;

        let mut scalars_d: DeviceVec<_> = DeviceVec::<F>::device_malloc_async(SIZE as usize, &stream).unwrap();
        scalars_d
            .copy_from_host_async(HostSlice::from_slice(&scalars), &stream)
            .unwrap();

        let fri_proof = fri_merkle_tree_prove::<F>(
            &fri_config,
            &transcript_config,
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            merkle_tree_min_layer_to_store,
        )
        .unwrap();

        let query_proofs = fri_proof
            .get_query_proofs::<u8>()
            .unwrap();
        let final_poly = fri_proof
            .get_final_poly()
            .unwrap();
        let pow_nonce = fri_proof
            .get_pow_nonce()
            .unwrap();
        let fri_proof_copy = FriProof::<F>::create_with_arguments(query_proofs, final_poly, pow_nonce).unwrap();

        let valid = fri_merkle_tree_verify(
            &fri_config,
            &transcript_config,
            &fri_proof_copy,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
        )
        .unwrap();
        assert!(valid);
    };
    test_utilities::test_set_main_device();
    check();
    test_utilities::test_set_ref_device();
    check();
}

pub fn check_fri_proof_serialization<F: FieldImpl>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_hash: &Hasher,
) where
    <F as FieldImpl>::Config: FriMerkleTree<F> + GenerateRandom<F>,
{
    const SIZE: u64 = 1 << 10;
        let stream = IcicleStream::create().unwrap();
        let mut fri_config = FriConfig::default();
        fri_config.is_async = true;
        fri_config.stream_handle = *stream;
        let scalars = F::Config::generate_random(SIZE as usize);

        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let merkle_tree_min_layer_to_store = 0;

        let mut scalars_d: DeviceVec<_> = DeviceVec::<F>::device_malloc_async(SIZE as usize, &stream).unwrap();
        scalars_d
            .copy_from_host_async(HostSlice::from_slice(&scalars), &stream)
            .unwrap();

        let fri_proof = fri_merkle_tree_prove::<F>(
            &fri_config,
            &transcript_config,
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
        merkle_tree_min_layer_to_store,
    )
    .unwrap();

    let fri_proof_serialized = fri_proof.serialize().unwrap();
    let fri_proof_deserialized = FriProof::<F>::deserialize(&fri_proof_serialized).unwrap();

    fri_proof.serialize_to_file("fri_proof.bin").unwrap();
    let fri_proof_deserialized_from_file: FriProof<F> = FriProof::<F>::deserialize_from_file("fri_proof.bin").unwrap();
    let merkle_proofs = fri_proof.get_query_proofs::<u8>().unwrap();
    let merkle_proofs_deserialized_from_file = fri_proof_deserialized_from_file.get_query_proofs::<u8>().unwrap();
    let merkle_proofs_deserialized = fri_proof_deserialized.get_query_proofs::<u8>().unwrap();
    for i in 0..merkle_proofs.len() {
        for j in 0..merkle_proofs[i].len() {
            let pruned = merkle_proofs[i][j].is_pruned;
            let pruned_deserialized = merkle_proofs_deserialized[i][j].is_pruned;
            let pruned_deserialized_from_file = merkle_proofs_deserialized_from_file[i][j].is_pruned;
            assert_eq!(pruned, pruned_deserialized);
            assert_eq!(pruned, pruned_deserialized_from_file);
            let path = &merkle_proofs[i][j].path;
            let path_deserialized = &merkle_proofs_deserialized[i][j].path;
            let path_deserialized_from_file = &merkle_proofs_deserialized_from_file[i][j].path;
            assert_eq!(path, path_deserialized);
            assert_eq!(path, path_deserialized_from_file);
            let root = &merkle_proofs[i][j].root;
            let root_deserialized = &merkle_proofs_deserialized[i][j].root;
            let root_deserialized_from_file = &merkle_proofs_deserialized_from_file[i][j].root;
            assert_eq!(root, root_deserialized);
            assert_eq!(root, root_deserialized_from_file);
            let leaf = &merkle_proofs[i][j].leaf;
            let leaf_deserialized = &merkle_proofs_deserialized[i][j].leaf;
            let leaf_deserialized_from_file = &merkle_proofs_deserialized_from_file[i][j].leaf;
            assert_eq!(leaf, leaf_deserialized);
            assert_eq!(leaf, leaf_deserialized_from_file);
        }
    }
    assert_eq!(fri_proof.get_final_poly(), fri_proof_deserialized.get_final_poly());
    assert_eq!(fri_proof.get_final_poly(), fri_proof_deserialized_from_file.get_final_poly());
    assert_eq!(fri_proof.get_pow_nonce(), fri_proof_deserialized.get_pow_nonce());
    assert_eq!(fri_proof.get_pow_nonce(), fri_proof_deserialized_from_file.get_pow_nonce());
}