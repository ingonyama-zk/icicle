use icicle_cuda_runtime::memory::HostOrDeviceSlice;

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
    let leaves = vec![F::one(); 1 << (height - 1)];
    let mut digests = vec![F::zero(); merkle_tree_digests_len(height, arity)];

    let mut leaves_slice = HostOrDeviceSlice::on_host(leaves);

    let constants = init_poseidon(arity as u32);

    let mut config = TreeBuilderConfig::default();
    config.keep_rows = keep_rows;
    build_poseidon_merkle_tree::<F>(&mut leaves_slice, &mut digests, height, arity, &constants, &config).unwrap();

    println!("Root: {:?}", digests[0..1][0]);
}
