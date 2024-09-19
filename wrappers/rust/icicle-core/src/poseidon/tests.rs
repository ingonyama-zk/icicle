use crate::{
    hash::HashConfig,
    poseidon::{create_poseidon_hasher, initialize_default_poseidon_constants, PoseidonHasher},
    traits::FieldImpl,
};
use icicle_runtime::memory::HostSlice;

pub fn check_poseidon_hash<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonHasher<F>,
{
    let batch = 1 << 10;
    let arity = 3;
    let mut inputs = vec![F::one(); batch * arity];
    let mut outputs = vec![F::zero(); arity];

    initialize_default_poseidon_constants::<F>().unwrap();
    let poseidon_hasher = create_poseidon_hasher::<F>(arity as u32).unwrap();

    poseidon_hasher
        .hash(
            HostSlice::from_slice(&mut inputs),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut outputs),
        )
        .unwrap();

    // TODO real test
}
