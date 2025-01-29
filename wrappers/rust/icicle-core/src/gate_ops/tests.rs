#![allow(unused_imports)]
use crate::traits::GenerateRandom;
use crate::gate_ops::{gate_evaluation, GateOps, GateOpsConfig, GateData, CalculationData, HornerData, FieldImpl};
use icicle_runtime::device::Device;
use icicle_runtime::memory::{DeviceVec, HostSlice, HostOrDeviceSlice};
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
    let instance = F::Config::generate_random(test_size);
    let challenges = F::Config::generate_random(test_size);
    let beta = F::Config::generate_random(1);
    let gamma = F::Config::generate_random(1);
    let theta = F::Config::generate_random(1);
    let y = F::Config::generate_random(1);
    let previous_value = F::Config::generate_random(1);

    let rotations: Vec<i32> = (0..test_size as i32).collect(); 
    let calculations: Vec<i32> = vec![0, 1, 2, 3];
    let i_value_types: Vec<i32> = vec![0, 1, 2, 3];
    let j_value_types: Vec<i32> = vec![1, 2, 3, 0];
    let i_value_indices: Vec<i32> = vec![0, 1, 2, 3];
    let j_value_indices: Vec<i32> = vec![3, 2, 1, 0];

    let horner_value_types: Vec<i32> = vec![0, 1];
    let i_horner_value_indices: Vec<i32> = vec![0, 1];
    let j_horner_value_indices: Vec<i32> = vec![1, 0];
    let horner_offsets: Vec<i32> = vec![0, 2];
    let horner_sizes: Vec<i32> = vec![2, 2];

    let gate_data = GateData::new(
        constants.as_ptr(),
        constants.len(),
        fixed.as_ptr(),
        fixed.len(), 
        advice.as_ptr(),
        advice.len(),
        instance.as_ptr(),
        instance.len(),
        rotations.as_ptr(),
        rotations.len(),
        challenges.as_ptr(),
        challenges.len(),
        beta.as_ptr(),
        gamma.as_ptr(),
        theta.as_ptr(),
        y.as_ptr(),
        previous_value.as_ptr(),
        test_size as i32,
        1,
        1,
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        i_value_types.as_ptr(),
        j_value_types.as_ptr(),
        i_value_indices.as_ptr(),
        j_value_indices.as_ptr(),
        calculations.len(),
        4,
    );

    let horner_data = HornerData::new(
        horner_value_types.as_ptr(),
        i_horner_value_indices.as_ptr(),
        j_horner_value_indices.as_ptr(),
        horner_offsets.as_ptr(),
        horner_sizes.as_ptr(),
    );

    let mut result = vec![F::zero(); test_size];
    let mut result = HostSlice::from_mut_slice(&mut result);

    test_utilities::test_set_main_device();
    let cfg = GateOpsConfig::default();

    gate_evaluation(
        &gate_data,
        &calc_data,
        &horner_data,
        result,
        &cfg,
    )
    .unwrap();
}
