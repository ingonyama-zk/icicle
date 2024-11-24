use crate::hash::SpongeHash;
use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};

use super::{DiffusionStrategy, MdsType, Poseidon2, Poseidon2Impl};

pub fn init_poseidon<F: FieldImpl>(width: usize, mds_type: MdsType, diffusion: DiffusionStrategy) -> Poseidon2<F>
where
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    let ctx = DeviceContext::default();
    Poseidon2::load(width, width, mds_type, diffusion, &ctx).unwrap()
}

fn _check_poseidon_hash_many<F: FieldImpl>(width: usize, poseidon: &Poseidon2<F>) -> (F, F)
where
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    let test_size = 1 << 10;
    let mut inputs = vec![F::one(); test_size * width];
    let mut outputs = vec![F::zero(); test_size];

    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    let cfg = poseidon.default_config();
    poseidon
        .hash_many(input_slice, output_slice, test_size, width, 1, &cfg)
        .unwrap();

    let a1 = output_slice[0];
    let a2 = output_slice[output_slice.len() - 2];

    assert_eq!(a1, a2);

    (a1, a2)
}

pub fn check_poseidon_hash_many<F: FieldImpl>()
where
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
{
    let widths = [2, 3, 4, 8, 12, 16, 20, 24];
    let ctx = DeviceContext::default();
    for width in widths {
        let poseidon = Poseidon2::<F>::load(width, width, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();

        _check_poseidon_hash_many(width, &poseidon);
    }
}

pub fn check_poseidon_kats<F: FieldImpl>(width: usize, kats: &[F], poseidon: &Poseidon2<F>)
where
    <F as FieldImpl>::Config: Poseidon2Impl<F>,
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

    let cfg = poseidon.default_config();

    poseidon
        .hash_many(input_slice, output_slice, batch_size, width, width, &cfg)
        .unwrap();

    for (i, val) in output_slice
        .iter()
        .enumerate()
    {
        assert_eq!(*val, kats[i % width]);
    }
}
