use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::get_default_device_context;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use super::{initialize_poseidon_constants, poseidon_hash_many, Poseidon, PoseidonConfig};

pub fn init_poseidon<F: FieldImpl>(arities: &[u32])
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let ctx = get_default_device_context();

    for arity in arities {
        initialize_poseidon_constants::<F>(*arity, &ctx).unwrap();
    }
}

pub fn check_poseidon_hash_many<F: FieldImpl>()
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let test_size = 1 << 10;
    let arity = 2;
    let inputs = vec![F::one(); test_size * arity];
    let outputs = vec![F::zero(); test_size];

    let mut input_slice = HostOrDeviceSlice::on_host(inputs);
    let mut output_slice = HostOrDeviceSlice::on_host(outputs);

    let config = PoseidonConfig::default();
    poseidon_hash_many::<F>(
        &mut input_slice,
        &mut output_slice,
        test_size as u32,
        arity as u32,
        &config,
    )
    .unwrap();

    println!(
        "first: {:?}, last: {:?}",
        output_slice[0..1][0],
        output_slice[output_slice.len() - 2..output_slice.len() - 1][0]
    );
}
