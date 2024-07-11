use icicle_cuda_runtime::memory::HostSlice;

use crate::{
    hash::SpongeHash,
    traits::FieldImpl,
    tree::{merkle_tree_digests_len, TreeBuilderConfig},
};

use super::FieldTreeBuilder;

pub fn check_build_field_merkle_tree<F, H, T>(
    height: usize,
    arity: usize,
    sponge: &H,
    compression: &H,
    _expected_root: F,
) where
    F: FieldImpl,
    H: SpongeHash<F, F>,
    T: FieldTreeBuilder<F, H, H>,
{
    let mut config = TreeBuilderConfig::default();
    config.arity = arity as u32;
    let input_block_len = arity;
    let leaves = vec![F::one(); (1 << height) * arity];
    let mut digests = vec![F::zero(); merkle_tree_digests_len((height + 1) as u32, arity as u32, 1)];

    let leaves_slice = HostSlice::from_slice(&leaves);
    let digests_slice = HostSlice::from_mut_slice(&mut digests);

    T::build_merkle_tree(
        leaves_slice,
        digests_slice,
        height,
        input_block_len,
        compression,
        sponge,
        &config,
    )
    .unwrap();
    println!("Root: {:?}", digests_slice[0]);
}
