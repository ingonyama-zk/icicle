use icicle_runtime::{eIcicleError, memory::HostSlice};

use super::FriImpl;
use crate::fri::fri_proof::FriProofTrait;
use crate::{
    fri::{fri, fri_transcript_config::FriTranscriptConfig, verify_fri, FriConfig, FriProof},
    hash::Hasher,
    traits::{FieldImpl, GenerateRandom},
};

pub fn check_fri<'a, F: FieldImpl>(
    merkle_tree_leaves_hash: &Hasher,
    merkle_tree_compress_hash: &Hasher,
    transcript_config: &'a FriTranscriptConfig<'_, F>,
) where
    <F as FieldImpl>::Config: FriImpl<'a, F> + GenerateRandom<F>,
    eIcicleError: From<
        <<<F as FieldImpl>::Config as FriImpl<'a, F>>::FriTranscriptConfigFFI as TryFrom<
            &'a FriTranscriptConfig<'a, F>,
        >>::Error,
    >,
{
    const SIZE: u64 = 1 << 10;
    let mut fri_proof = FriProof::<F>::new();
    let fri_config = FriConfig::default();
    let scalars = F::Config::generate_random(SIZE as usize);

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
}
