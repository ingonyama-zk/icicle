use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::get_default_device_context;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use super::{load_optimized_poseidon_constants, poseidon_hash_many, Poseidon, PoseidonConfig, PoseidonConstants};

pub fn init_poseidon<'a, F: FieldImpl>(arity: u32) -> PoseidonConstants<'a, F>
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let ctx = get_default_device_context();

    load_optimized_poseidon_constants::<F>(arity, &ctx).unwrap()
}

pub fn check_poseidon_hash_many<F: FieldImpl>()
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let test_size = 1 << 10;
    let arity = 2u32;
    let inputs = vec![F::one(); test_size * arity as usize];
    let outputs = vec![F::zero(); test_size];

    let mut input_slice = HostOrDeviceSlice::on_host(inputs);
    let mut output_slice = HostOrDeviceSlice::on_host(outputs);

    let constants = init_poseidon(arity as u32);

    let config = PoseidonConfig::default();
    poseidon_hash_many::<F>(
        &mut input_slice,
        &mut output_slice,
        test_size as u32,
        arity as u32,
        &constants,
        &config,
    )
    .unwrap();

    println!(
        "first: {:?}, last: {:?}",
        output_slice[0..1][0],
        output_slice[output_slice.len() - 2..output_slice.len() - 1][0]
    );
}
