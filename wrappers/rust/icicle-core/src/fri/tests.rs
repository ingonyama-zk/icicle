use super::FriMerkleTree;
use crate::{
    fri::{
        fri_merkle_tree_prove, fri_merkle_tree_verify, fri_proof::FriProofTrait,
        fri_transcript_config::FriTranscriptConfig, FriConfig, FriProof,
    },
    hash::Hasher,
    traits::{FieldImpl, GenerateRandom},
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
            .get_query_proofs()
            .unwrap();
        let final_poly = fri_proof
            .get_final_poly()
            .unwrap();
        let pow_nonce = fri_proof
            .get_pow_nonce()
            .unwrap();

        let fri_proof_copy = FriProof::<F>::create_with_arguments(query_proofs, final_poly, pow_nonce);

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
