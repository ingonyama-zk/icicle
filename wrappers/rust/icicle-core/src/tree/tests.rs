use icicle_cuda_runtime::memory::HostSlice;

use crate::{
    poseidon::{tests::init_poseidon, Poseidon},
    traits::FieldImpl,
    tree::{build_poseidon_merkle_tree, merkle_tree_digests_len, TreeBuilderConfig},
};

use super::TreeBuilder;

pub fn check_build_merkle_tree<F: FieldImpl>()
where
    <F as FieldImpl>::Config: TreeBuilder<F> + Poseidon<F>,
{
    let height = 20;
    let arity = 2;
    let keep_rows = 1;
    let mut leaves = vec![F::one(); 1 << (height - 1)];
    let mut digests = vec![F::zero(); merkle_tree_digests_len(height, arity)];

    let leaves_slice = HostSlice::from_mut_slice(&mut leaves);

    let constants = init_poseidon(arity as u32);

    let mut config = TreeBuilderConfig::default();
    config.keep_rows = keep_rows;
    build_poseidon_merkle_tree::<F>(leaves_slice, &mut digests, height, arity, &constants, &config).unwrap();

    println!("Root: {:?}", digests[0]);
}
