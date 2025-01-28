#![allow(unused_imports)]
use crate::traits::GenerateRandom;
use crate::gate_ops::{GateOps, GateOpsConfig, FieldImpl};
use icicle_runtime::device::Device;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::{runtime, stream::IcicleStream, test_utilities};

#[test]
fn test_gate_ops_config() {
    let mut gate_ops_config = GateOpsConfig::default();
    gate_ops_config
        .ext
        .set_int("int_example", 5);

    assert_eq!(
        gate_ops_config
            .ext
            .get_int("int_example"),
        5
    );

    // just to test the stream can be set and used correctly
    let mut stream = IcicleStream::create().unwrap();
    gate_ops_config.stream_handle = *stream;

    stream
        .synchronize()
        .unwrap();
}

pub fn check_gate_ops_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let test_size = 1 << 14;

    check_gate_ops_evaluation::<F>(test_size);
}


pub fn check_gate_ops_evaluation<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let constants = F::Config::generate_random(test_size);
    let fixed = F::Config::generate_random(test_size);
    let advice = F::Config::generate_random(test_size);
    let instants = F::Config::generate_random(test_size);

    let mut result_main = vec![F::zero(); test_size];

    let _constants = HostSlice::from_slice(&constants);
    let _fixed = HostSlice::from_slice(&fixed);
    let _advice = HostSlice::from_slice(&advice);
    let _instants = HostSlice::from_slice(&instants);

    let _result = HostSlice::from_mut_slice(&mut result_main);

    let _cfg = GateOpsConfig::default();

    test_utilities::test_set_main_device();
    // call evaluation 
}