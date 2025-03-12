use icicle_core::impl_fri;

use crate::field::{ExtensionField, ExtensionCfg, ScalarCfg, ScalarField};

impl_fri!("babybear", babybear_fri, ScalarField, ScalarCfg);
impl_fri!("babybear_extension", babybear_extension_fri, ExtensionField, ExtensionCfg);

#[cfg(test)]
mod tests {
    use icicle_core::fri::{fri_proof::FriProofTrait, FriConfig, FriImpl};
    use icicle_hash::keccak::Keccak256;
    use icicle_runtime::memory::HostSlice;

    use crate::field::{ScalarCfg, ScalarField, SCALAR_LIMBS};
    use icicle_core::traits::GenerateRandom;

    #[test]
    fn it_works() {
        let transcript_config = <ScalarCfg as FriImpl::<ScalarField>>::FriTranscriptConfig::default();
        let mut fri_proof = <ScalarCfg as FriImpl::<ScalarField>>::FriProof::new();
        let mut fri_config = FriConfig::default();
        fri_config.nof_queries = 5;
        fri_config.pow_bits = 8;

        let scalars = ScalarCfg::generate_random(1 << 10);

        let merkle_tree_leaves_hash = Keccak256::new(SCALAR_LIMBS as u64 * 4).unwrap();
        let merkle_tree_compress_hash = Keccak256::new(2 * merkle_tree_leaves_hash.output_size()).unwrap();
        let output_store_min_layer = 0;

        ScalarCfg::get_fri_proof_mt(&fri_config, &transcript_config, HostSlice::from_slice(&scalars), &merkle_tree_leaves_hash, &merkle_tree_compress_hash, output_store_min_layer, &mut fri_proof).unwrap();
    }
}