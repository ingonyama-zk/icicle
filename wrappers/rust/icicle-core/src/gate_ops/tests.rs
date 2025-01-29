#![allow(unused_imports)]
use crate::traits::GenerateRandom;
use crate::gate_ops::{gate_evaluation, GateOps, GateOpsConfig, FieldImpl};
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

    let mut result = vec![F::zero(); test_size];

    let constants = HostSlice::from_slice(&constants);
    let fixed = HostSlice::from_slice(&fixed);
    let advice = HostSlice::from_slice(&advice);
    let instance = HostSlice::from_slice(&instance);
    let challenges = HostSlice::from_slice(&challenges);
    let beta =  HostSlice::from_slice(&beta);
    let gamma =  HostSlice::from_slice(&gamma);
    let theta =  HostSlice::from_slice(&theta);
    let y =  HostSlice::from_slice(&y);
    let previous_value =  HostSlice::from_slice(&previous_value);

    let result = HostSlice::from_mut_slice(&mut result);

    let cfg = GateOpsConfig::default();

    test_utilities::test_set_main_device();

    let evaluation_result = gate_evaluation(
        constants,
        fixed,
        advice,
        instance,
        challenges,
        &rotations,
        beta,
        gamma,
        theta,
        y,
        previous_value,
        &calculations,
        &i_value_types,
        &j_value_types,
        &i_value_indices,
        &j_value_indices,
        &horner_value_types,
        &i_horner_value_indices,
        &j_horner_value_indices,
        &horner_offsets,
        &horner_sizes,
        result,
        &cfg,
    );

   
    // for res in result.iter() {
    //     println!("Result: {:?}", res);
    // }
}
