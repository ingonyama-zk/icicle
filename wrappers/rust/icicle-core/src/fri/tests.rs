use super::FriImpl;
use crate::fri::fri_proof::FriProofTrait;
use crate::{
    fri::{fri, fri_transcript_config::FriTranscriptConfig, verify_fri, FriConfig, FriProof},
    hash::Hasher,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_runtime::eIcicleError;
use icicle_runtime::{memory::HostSlice, test_utilities};

pub fn check_fri<F: FieldImpl>(hash_new: &dyn Fn(u64) -> Result<Hasher, eIcicleError>)
where
    <F as FieldImpl>::Config: FriImpl<F> + GenerateRandom<F>,
{
    let check = || {
        const SIZE: u64 = 1 << 10;
        let mut fri_proof = FriProof::<F>::new();
        let fri_config = FriConfig::default();
        let scalars = F::Config::generate_random(SIZE as usize);

        let merkle_tree_leaves_hash = hash_new(std::mem::size_of::<F>() as u64).unwrap();
        let merkle_tree_compress_hash = hash_new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
        let transcript_hash = hash_new(0).unwrap();
        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, F::one());

        let output_store_min_layer = 0;

        fri::<F>(
            &fri_config,
            &transcript_config,
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            output_store_min_layer,
            &mut fri_proof,
        )
        .unwrap();
        let valid = verify_fri(
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
