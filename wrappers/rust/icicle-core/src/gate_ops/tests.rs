#![allow(unused_imports)]
use crate::traits::GenerateRandom;
use crate::gate_ops::{gate_evaluation, GateOps, GateOpsConfig, GateData, CalculationData, HornerData, FieldImpl};
use icicle_runtime::device::Device;
use icicle_runtime::memory::{DeviceVec, HostSlice, HostOrDeviceSlice};
use icicle_runtime::{runtime, stream::IcicleStream, test_utilities};
use rand::Rng;

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
    let test_size = 1 << 4;

    check_gate_ops_evaluation::<F>(test_size);
}


pub fn check_gate_ops_evaluation<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let mut rng = rand::thread_rng();

    let table_size = 1 << 5;
    let graph_size = 1 << 4;

    let constants = F::Config::generate_random(table_size);
    let fixed = F::Config::generate_random(table_size);
    let advice = F::Config::generate_random(table_size);
    let instance = F::Config::generate_random(table_size);
    let challenges = F::Config::generate_random(table_size);
    let beta = F::Config::generate_random(1);
    let gamma = F::Config::generate_random(1);
    let theta = F::Config::generate_random(1);
    let y = F::Config::generate_random(1);
    let previous_value = F::Config::generate_random(1);

    let fixed    = vec![fixed.as_ptr()];
    let advice   = vec![advice.as_ptr()];
    let instance = vec![instance.as_ptr()];

    let rotations: Vec<u32> = vec![0];
    let calculations: Vec<u32> = (0..graph_size).map(|_| rng.gen_range(0..8)).collect();
    let targets: Vec<u32> = Vec::from_iter(0..graph_size as u32);
    let value_types: Vec<u32> = (0..graph_size * 2).map(|_| rng.gen_range(0..8)).collect();
    let value_indices: Vec<u32> = (0..graph_size * 4).map(|_| rng.gen_range(0..8)).collect();

    let horner_value_types: Vec<u32> = vec![0, 1];
    let horner_value_indices: Vec<u32> = vec![0, 1];
    let horner_offsets: Vec<u32> = vec![0, 2];
    let horner_sizes: Vec<u32> = vec![2, 2];

    let gate_data = GateData::new(
        constants.as_ptr(),
        constants.len() as u32,
        fixed.as_ptr(),
        fixed.len() as u32,
        table_size as u32,
        advice.as_ptr(),
        advice.len() as u32,
        table_size as u32,
        instance.as_ptr(),
        instance.len() as u32,
        table_size as u32,
        rotations.as_ptr(),
        rotations.len() as u32,
        challenges.as_ptr(),
        challenges.len() as u32,
        beta.as_ptr(),
        gamma.as_ptr(),
        theta.as_ptr(),
        y.as_ptr(),
        previous_value.as_ptr(),
        test_size as u32,
        1,
        1,
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        targets.as_ptr(),
        value_types.as_ptr(),
        value_indices.as_ptr(),
        calculations.len() as u32,
        calculations.len() as u32,
    );

    let horner_data = HornerData::new(
        horner_value_types.as_ptr(),
        horner_value_indices.as_ptr(),
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

    println!("result: {:?}", result);
}
