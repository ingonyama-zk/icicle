use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::get_default_device_context;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use std::io::Read;
use std::path::PathBuf;
use std::{env, fs::File};

use super::{
    create_optimized_poseidon_constants, load_optimized_poseidon_constants, poseidon_hash_many, Poseidon,
    PoseidonConfig, PoseidonConstants,
};

pub fn init_poseidon<'a, F: FieldImpl>(arity: u32) -> PoseidonConstants<'a, F>
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let ctx = get_default_device_context();

    load_optimized_poseidon_constants::<F>(arity, &ctx).unwrap()
}

pub fn _check_poseidon_hash_many<F: FieldImpl>(constants: PoseidonConstants<F>) -> (F, F)
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let test_size = 1 << 10;
    let arity = 2u32;
    let inputs = vec![F::one(); test_size * arity as usize];
    let outputs = vec![F::zero(); test_size];

    let mut input_slice = HostOrDeviceSlice::on_host(inputs);
    let mut output_slice = HostOrDeviceSlice::on_host(outputs);

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

    let a1 = output_slice[0..1][0];
    let a2 = output_slice[output_slice.len() - 2..output_slice.len() - 1][0];

    println!("first: {:?}, last: {:?}", a1, a2);
    assert_eq!(a1, a2);

    (a1, a2)
}

pub fn check_poseidon_hash_many<F: FieldImpl>()
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let arity = 2u32;
    let constants = init_poseidon::<F>(arity as u32);

    _check_poseidon_hash_many(constants);
}

pub fn check_poseidon_custom_config<F: FieldImpl>(field_bytes: usize, field_prefix: &str, partial_rounds: u32)
where
    <F as FieldImpl>::Config: Poseidon<F>,
{
    let arity = 2u32;
    let constants = init_poseidon::<F>(arity as u32);

    let full_rounds_half = 4;

    let ctx = get_default_device_context();
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    let constants_file = PathBuf::from(cargo_manifest_dir)
        .join("tests")
        .join(format!("{}_constants.bin", field_prefix));
    let mut constants_buf = vec![];
    File::open(constants_file)
        .unwrap()
        .read_to_end(&mut constants_buf)
        .unwrap();

    let mut custom_constants = vec![];
    for chunk in constants_buf.chunks(field_bytes) {
        custom_constants.push(F::from_bytes_le(chunk));
    }

    let custom_constants = create_optimized_poseidon_constants::<F>(
        arity as u32,
        &ctx,
        full_rounds_half,
        partial_rounds,
        &mut custom_constants,
    )
    .unwrap();

    let (a1, a2) = _check_poseidon_hash_many(constants);
    let (b1, b2) = _check_poseidon_hash_many(custom_constants);

    assert_eq!(a1, b1);
    assert_eq!(a2, b2);
}
