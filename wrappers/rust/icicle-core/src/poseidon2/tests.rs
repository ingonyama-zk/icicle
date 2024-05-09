use crate::poseidon2::{MdsType, PoseidonMode};
use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};

use super::{
    load_optimized_poseidon2_constants, poseidon_hash_many, DiffusionStrategy, Poseidon2, Poseidon2Config,
    Poseidon2Constants,
};

pub fn init_poseidon<'a, F: FieldImpl>(
    width: u32,
    mds_type: MdsType,
    diffusion: DiffusionStrategy,
) -> Poseidon2Constants<'a, F>
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    let ctx = DeviceContext::default();
    load_optimized_poseidon2_constants::<F>(width, mds_type, diffusion, &ctx).unwrap()
}

fn _check_poseidon_hash_many<F: FieldImpl>(width: u32, constants: Poseidon2Constants<F>) -> (F, F)
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    let test_size = 1 << 10;
    let mut inputs = vec![F::one(); test_size * width as usize];
    let mut outputs = vec![F::zero(); test_size];

    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    let config = Poseidon2Config::default();
    poseidon_hash_many::<F>(
        input_slice,
        output_slice,
        test_size as u32,
        width as u32,
        &constants,
        &config,
    )
    .unwrap();

    let a1 = output_slice[0];
    let a2 = output_slice[output_slice.len() - 2];

    assert_eq!(a1, a2);

    (a1, a2)
}

pub fn check_poseidon_hash_many<'a, F: FieldImpl + 'a>()
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    let widths = [2, 3, 4, 8, 12, 16, 20, 24];
    for width in widths {
        let constants = init_poseidon::<'a, F>(width as u32, MdsType::Default, DiffusionStrategy::Default);

        _check_poseidon_hash_many(width, constants);
    }
}

pub fn check_poseidon_kats<'a, F: FieldImpl>(width: usize, kats: &[F], constants: &Poseidon2Constants<'a, F>)
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    assert_eq!(width, kats.len());

    let batch_size = 1024;
    let mut input = vec![F::one(); width];
    let mut outputs = vec![F::zero(); width * batch_size];

    for i in 0..width {
        input[i] = F::from_u32(i as u32);
    }

    let mut inputs: Vec<F> = std::iter::repeat(input.clone())
        .take(batch_size)
        .flatten()
        .collect();

    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    let mut config = Poseidon2Config::default();
    config.mode = PoseidonMode::Permutation;
    poseidon_hash_many::<F>(
        input_slice,
        output_slice,
        batch_size as u32,
        width as u32,
        &constants,
        &config,
    )
    .unwrap();

    for (i, val) in output_slice
        .iter()
        .enumerate()
    {
        assert_eq!(*val, kats[i % width]);
    }
}
