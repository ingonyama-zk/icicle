#![allow(unused_imports)]
use crate::test_utilities;
use crate::traits::GenerateRandom;
use crate::sumcheck::{
    sumcheck,
    FieldImpl, Sumcheck, SumcheckConfig,
};
use icicle_runtime::device::Device;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::{runtime, stream::IcicleStream};

#[test]
fn test_sumcheck_config() {
    let mut sumcheck_config = SumcheckConfig::default();
    sumcheck_config
        .ext
        .set_int("int_example", 5);

    assert_eq!(
        sumcheck_config
            .ext
            .get_int("int_example"),
        5
    );

    // just to test the stream can be set and used correctly
    let mut stream = IcicleStream::create().unwrap();
    sumcheck_config.stream_handle = *stream;

    stream
        .synchronize()
        .unwrap();
    stream
        .destroy()
        .unwrap();
}

pub fn check_sumcheck_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: Sumcheck<F, F> + GenerateRandom<F>,
{
    let test_size = 1 << 14;

    check_sumcheck_scalars_sumcheck::<F>(test_size);
}

pub fn check_sumcheck_scalars_sumcheck<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: Sumcheck<F, F> + GenerateRandom<F>,
{
    let evals = F::Config::generate_random(test_size);
    let cubic_polys = F::Config::generate_random(test_size);
    let transcript = F::Config::generate_random(test_size);
    let c = F::Config::generate_random(1);
    let num_rounds = 4;
    let nof_polys = 4;
    // let mut result_main = vec![F::zero(); test_size];
    // let mut result_ref = vec![F::zero(); test_size];

    let evals = HostSlice::from_slice(&evals);
    let cubic_polys = HostSlice::from_slice(&cubic_polys);
    let transcript = HostSlice::from_slice(&transcript);
    let c = HostSlice::from_slice(&c);

    // let result_main = HostSlice::from_mut_slice(&mut result_main);
    // let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    let mut stream = IcicleStream::create().unwrap();
    let mut cfg = SumcheckConfig::default();
    cfg.stream_handle = *stream;

    test_utilities::test_set_main_device();
    sumcheck::<F, F>(evals, cubic_polys, transcript, c, num_rounds, nof_polys, &stream).unwrap();

    stream
        .destroy()
        .unwrap();
}