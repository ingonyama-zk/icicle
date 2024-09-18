use crate::{
    hash::{HashConfig, Hasher},
    traits::FieldImpl,
};
use icicle_runtime::memory::HostSlice;

pub fn check_poseidon_hash<F: FieldImpl>(poseidon_hasher: &Hasher) {
    let test_size = 1 << 10;
    let arity = 3;
    let mut inputs = vec![F::one(); test_size * arity];
    let mut outputs = vec![F::zero(); test_size];

    poseidon_hasher
        .hash(
            HostSlice::from_slice(&mut inputs),
            &HashConfig::default(),
            HostSlice::from_mut_slice(&mut outputs),
        )
        .unwrap();

    // TODO real test
}
