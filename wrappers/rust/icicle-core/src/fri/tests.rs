use super::FriMerkleTreeImpl;
use crate::{
    fri::{fri_merkle_tree_prove, fri_merkle_tree_verify, fri_proof::FriProofTrait, fri_transcript_config::FriTranscriptConfig, FriConfig, FriProof},
    hash::Hasher,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_runtime::{eIcicleError, memory::DeviceVec, stream::IcicleStream};
use icicle_runtime::{memory::HostSlice, test_utilities};

pub fn check_fri<F: FieldImpl>(hash_new: &dyn Fn(u64) -> Result<Hasher, eIcicleError>)
where
    <F as FieldImpl>::Config: FriMerkleTreeImpl<F> + GenerateRandom<F>,
{
    let check = || {
        const SIZE: u64 = 1 << 10;
        let fri_config = FriConfig::default();
        let scalars = F::Config::generate_random(SIZE as usize);

        let merkle_tree_leaves_hash = hash_new(std::mem::size_of::<F>() as u64).unwrap();
        let merkle_tree_compress_hash = hash_new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
        let transcript_hash = hash_new(0).unwrap();
        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let output_store_min_layer = 0;

        let fri_proof = fri_merkle_tree_prove::<F>(
            &fri_config,
            &transcript_config,
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            output_store_min_layer,
        )
        .unwrap();
        let valid = fri_merkle_tree_verify(
            &fri_config,
            &transcript_config,
            &fri_proof,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            output_store_min_layer,
        )
        .unwrap();
        assert!(valid);
    };
    test_utilities::test_set_main_device();
    check();
    test_utilities::test_set_ref_device();
    check();
}

pub fn check_fri_on_device<F: FieldImpl>(hash_new: &dyn Fn(u64) -> Result<Hasher, eIcicleError>)
where
    <F as FieldImpl>::Config: FriMerkleTreeImpl<F> + GenerateRandom<F>,
{
    let check = || {
        const SIZE: u64 = 1 << 10;
        let stream = IcicleStream::create().unwrap();
        let mut fri_config = FriConfig::default();
        fri_config.is_async = true;
        fri_config.stream_handle = *stream;
        let scalars = F::Config::generate_random(SIZE as usize);

        let merkle_tree_leaves_hash = hash_new(std::mem::size_of::<F>() as u64).unwrap();
        let merkle_tree_compress_hash = hash_new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
        let transcript_hash = hash_new(0).unwrap();
        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let output_store_min_layer = 0;

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
            output_store_min_layer,
        )
        .unwrap();

        let query_proofs = fri_proof.get_query_proofs().unwrap();
        let final_poly = fri_proof.get_final_poly().unwrap();
        let pow_nonce = fri_proof.get_pow_nonce().unwrap();

        let fri_proof_copy = FriProof::<F>::create_with_arguments(query_proofs, final_poly, pow_nonce);


        let valid = fri_merkle_tree_verify(
            &fri_config,
            &transcript_config,
            &fri_proof_copy,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            output_store_min_layer,
        )
        .unwrap();
        assert!(valid);
    };
    test_utilities::test_set_main_device();
    check();
    test_utilities::test_set_ref_device();
    check();
}