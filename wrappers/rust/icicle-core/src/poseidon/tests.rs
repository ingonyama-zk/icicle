use crate::hash::SpongeHash;
use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};

use super::{Poseidon, PoseidonImpl};

pub fn init_poseidon<F: FieldImpl>(arity: usize) -> Poseidon<F>
where
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    let ctx = DeviceContext::default();
    Poseidon::load(arity, &ctx).unwrap()
}

pub fn _check_poseidon_hash_many<F: FieldImpl>(poseidon: Poseidon<F>)
where
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    let test_size = 1 << 10;
    let arity = poseidon.width - 1;
    let mut inputs = vec![F::one(); test_size * arity];
    let mut outputs = vec![F::zero(); test_size];

    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    let cfg = poseidon.default_config();
    poseidon
        .hash_many(input_slice, output_slice, test_size, arity, 1, &cfg)
        .unwrap();

    let a1 = output_slice[0];
    let a2 = output_slice[output_slice.len() - 1];

    assert_eq!(a1, a2);
}

pub fn check_poseidon_hash_many<F: FieldImpl>()
where
    <F as FieldImpl>::Config: PoseidonImpl<F>,
{
    for arity in [2, 4, 8, 11] {
        let constants = init_poseidon::<F>(arity);

        _check_poseidon_hash_many(constants);
    }
}
