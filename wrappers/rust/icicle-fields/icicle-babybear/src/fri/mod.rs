use icicle_core::impl_fri;

use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

impl_fri!("babybear", babybear_fri, ScalarField, ScalarCfg);
impl_fri!(
    "babybear_extension",
    babybear_extension_fri,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
mod tests {
    use icicle_core::{
        fri::{fri_proof::FriProofTrait, fri_transcript_config::FriTranscriptConfig, FriConfig, FriImpl},
        ntt, traits::FieldImpl,
    };
    use icicle_hash::keccak::Keccak256;
    use icicle_runtime::memory::HostSlice;

    use crate::{field::{ScalarCfg, ScalarField, SCALAR_LIMBS}, fri::babybear_fri::FriTranscriptConfigFFI};
    use icicle_core::traits::GenerateRandom;

    #[test]
    fn it_works() {
        const SIZE: u64 = 1 << 10;
        let ntt_init_config = ntt::NTTInitDomainConfig::default();
        ntt::initialize_domain(ntt::get_root_of_unity::<ScalarField>(SIZE), &ntt_init_config).unwrap();
        // let transcript_config = <ScalarCfg as FriImpl<ScalarField>>::FriTranscriptConfig::default();
        let mut fri_proof = <ScalarCfg as FriImpl<ScalarField>>::FriProof::new();
        let mut fri_config = FriConfig::default();
        fri_config.nof_queries = 5;
        fri_config.pow_bits = 8;

        let scalars = ScalarCfg::generate_random(SIZE as usize);

        let merkle_tree_leaves_hash = Keccak256::new(SCALAR_LIMBS as u64 * 4).unwrap();
        let merkle_tree_compress_hash = Keccak256::new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
        let output_store_min_layer = 0;

        let transcript_hash = Keccak256::new(0).unwrap();
        let transcript_config = FriTranscriptConfig::new_default_labels(&transcript_hash, ScalarField::one());

        ScalarCfg::get_fri_proof_mt(
            &fri_config,
            &FriTranscriptConfigFFI::try_from(&transcript_config).unwrap(),
            HostSlice::from_slice(&scalars),
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            output_store_min_layer,
            &mut fri_proof,
        )
        .unwrap();
        let valid = ScalarCfg::verify_fri_mt(
            &fri_config,
            &FriTranscriptConfigFFI::try_from(&transcript_config).unwrap(),
            &fri_proof,
            &merkle_tree_leaves_hash,
            &merkle_tree_compress_hash,
            output_store_min_layer,
        )
        .unwrap();
        assert!(valid);
    }
}
