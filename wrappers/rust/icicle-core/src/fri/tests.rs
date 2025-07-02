use super::FriMerkleTree;
use crate::{
    fri::{
        fri_merkle_tree_prove, fri_merkle_tree_verify, fri_proof::FriProofOps,
        fri_transcript_config::FriTranscriptConfig, FriConfig, FriProof,
    },
    hash::Hasher,
    traits::{Arithmetic, GenerateRandom},
};
use crate::field::PrimeField;
use icicle_runtime::{memory::DeviceVec, stream::IcicleStream};
use icicle_runtime::{memory::HostSlice, test_utilities};

pub fn check_fri<F: PrimeField>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_hash: &Hasher,
) where
    F: FriMerkleTree<F> + GenerateRandom + Arithmetic,
{
    let check = || {
        const SIZE: u64 = 1 << 9;
        let fri_config = FriConfig::default();
        let scalars = F::generate_random(SIZE as usize);

        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let merkle_tree_min_layer_to_store = 0;

        let fri_proof = F::fri_merkle_tree_prove(
            &fri_config,
            &transcript_config,
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            merkle_tree_min_layer_to_store,
        )
        .unwrap();

        let query_proofs = fri_proof
            .get_query_proofs()
            .unwrap();
        let final_poly = fri_proof
            .get_final_poly()
            .unwrap();
        let pow_nonce = fri_proof
            .get_pow_nonce()
            .unwrap();
        let fri_proof_copy = FriProof::<F>::create_with_arguments(query_proofs, final_poly, pow_nonce).unwrap();

        let valid = F::fri_merkle_tree_verify(
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

pub fn check_fri_on_device<F: PrimeField>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_hash: &Hasher,
) where
    F: FriMerkleTree<F> + GenerateRandom + Arithmetic,
{
    let check = || {
        const SIZE: u64 = 1 << 10;
        let mut stream = IcicleStream::create().unwrap();
        let mut fri_config = FriConfig::default();
        fri_config.is_async = true;
        fri_config.stream_handle = *stream;
        let scalars = F::generate_random(SIZE as usize);

        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let merkle_tree_min_layer_to_store = 0;

        let mut scalars_d: DeviceVec<_> = DeviceVec::<F>::device_malloc_async(SIZE as usize, &stream).unwrap();
        scalars_d
            .copy_from_host_async(HostSlice::from_slice(&scalars), &stream)
            .unwrap();

        let fri_proof = fri_merkle_tree_prove::<F>(
            &fri_config,
            &transcript_config,
            &scalars_d,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            merkle_tree_min_layer_to_store,
        )
        .unwrap();
        stream
            .synchronize()
            .unwrap();

        let valid = fri_merkle_tree_verify(
            &fri_config,
            &transcript_config,
            &fri_proof,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
        )
        .unwrap();
        stream
            .synchronize()
            .unwrap();

        assert!(valid);
        stream
            .destroy()
            .unwrap();
    };
    test_utilities::test_set_main_device();
    check();
    test_utilities::test_set_ref_device();
    check();
}

pub fn check_fri_proof_serialization<F: PrimeField, S, D, T>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_hash: &Hasher,
    serialize: S,
    deserialize: D,
) where
    F: FriMerkleTree<F> + GenerateRandom + Arithmetic,
    S: Fn(&FriProof<F>) -> T,
    D: Fn(&T) -> FriProof<F>,
{
    const SIZE: u64 = 1 << 10;
    let fri_config = FriConfig::default();
    let scalars = F::generate_random(SIZE as usize);

    let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

    let merkle_tree_min_layer_to_store = 0;

    let mut scalars_d: DeviceVec<_> = DeviceVec::<F>::device_malloc(SIZE as usize).unwrap();
    scalars_d
        .copy_from_host(HostSlice::from_slice(&scalars))
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

    let fri_proof_serialized = serialize(&fri_proof);
    let fri_proof_deserialized = deserialize(&fri_proof_serialized);

    let merkle_proofs = fri_proof
        .get_query_proofs()
        .unwrap();
    let merkle_proofs_deserialized = fri_proof_deserialized
        .get_query_proofs()
        .unwrap();
    for i in 0..merkle_proofs.len() {
        for j in 0..merkle_proofs[i].len() {
            let pruned = merkle_proofs[i][j].is_pruned;
            let pruned_deserialized = merkle_proofs_deserialized[i][j].is_pruned;
            assert_eq!(pruned, pruned_deserialized);
            let path = &merkle_proofs[i][j].path;
            let path_deserialized = &merkle_proofs_deserialized[i][j].path;
            assert_eq!(path, path_deserialized);
            let root = &merkle_proofs[i][j].root;
            let root_deserialized = &merkle_proofs_deserialized[i][j].root;
            assert_eq!(root, root_deserialized);
            let leaf = &merkle_proofs[i][j].leaf;
            let leaf_deserialized = &merkle_proofs_deserialized[i][j].leaf;
            assert_eq!(leaf, leaf_deserialized);
        }
    }
    assert_eq!(fri_proof.get_final_poly(), fri_proof_deserialized.get_final_poly());
    assert_eq!(fri_proof.get_pow_nonce(), fri_proof_deserialized.get_pow_nonce());
}
