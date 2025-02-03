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
    check_horner_evaluation::<F>(1);
    check_complex_gate_ops::<F>(1);
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
        1,
        table_size as u32,
        advice.as_ptr(),
        1,
        table_size as u32,
        instance.as_ptr(),
        1,
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
        horner_value_types.len() as u32
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

pub fn check_horner_evaluation<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let table_size = 8;

    let constants = vec![
        F::from_u32(5u32),
        F::from_u32(7u32),
        F::from_u32(100u32),
        F::from_u32(200u32),
    ];
    let fixed = vec![
        F::from_u32(11u32),
        F::from_u32(11u32),
        F::from_u32(11u32),
        F::from_u32(11u32),
    ];
    let advice = vec![
        F::from_u32(13u32),
        F::from_u32(13u32),
        F::from_u32(13u32),
        F::from_u32(13u32),
    ];

    let instance = vec![
        F::from_u32(13u32),
        F::from_u32(13u32),
        F::from_u32(13u32),
        F::from_u32(13u32),
    ];
    let challenges      = F::Config::generate_random(1);

    let beta            = F::Config::generate_random(1);
    let gamma           = F::Config::generate_random(1);
    let theta           = F::Config::generate_random(1);
    let y               = F::Config::generate_random(1);
    let previous_value  = F::Config::generate_random(1);

    let rotations: Vec<u32> = vec![0];
    let calculations: Vec<u32> = vec![6, 7];
    let targets: Vec<u32> = vec![0, 1];
    let value_types: Vec<u32> = vec![0, 0, 1, 1];
    let value_indices: Vec<u32> = vec![0, 0, 1, 0, 0, 0, 0, 0]; 

    // calc[0] = horner(constants[0], constants[1], [fixed[0][0], advice[0][0]]) -> target[0]
    // calc[1] = store(intermediates[0]) -> target[1]

    let horner_offsets: Vec<u32> = vec![0, 0];
    let horner_sizes:   Vec<u32> = vec![2, 0];
    let horner_value_types: Vec<u32> = vec![2, 3];
    let horner_value_indices: Vec<u32> = vec![0, 0, 0, 0];

    let gate_data = GateData::new(
        constants.as_ptr(),
        constants.len() as u32,
        fixed.as_ptr(),
        1,
        table_size as u32,
        advice.as_ptr(),
        1,
        table_size as u32,
        instance.as_ptr(),
        1,
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
        2,
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        targets.as_ptr(),
        value_types.as_ptr(),
        value_indices.as_ptr(),
        calculations.len() as u32,
        2,
    );

    let horner_data = HornerData::new(
        horner_value_types.as_ptr(),
        horner_value_indices.as_ptr(),
        horner_offsets.as_ptr(),
        horner_sizes.as_ptr(),
        horner_value_types.len() as u32,
    );

    let mut result = vec![F::zero(); test_size];
    let mut result_slice = HostSlice::from_mut_slice(&mut result);

    test_utilities::test_set_main_device();

    let cfg = GateOpsConfig::default();

    gate_evaluation(
        &gate_data,
        &calc_data,
        &horner_data,
        result_slice,
        &cfg,
    )
    .unwrap();

    let expected = F::from_u32(335u32);
    assert_eq!(result[0], expected, "Horner test did not match expected value");
}

pub fn check_complex_gate_ops<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let table_size = 8;

    let constants = vec![
        F::from_u32(5),
        F::from_u32(7),
        F::from_u32(3),
        F::from_u32(200),
    ];

    let fixed = vec![
        F::from_u32(11),
        F::from_u32(111),
        F::from_u32(111),
        F::from_u32(111),
        F::from_u32(111),
        F::from_u32(111),
        F::from_u32(111),
        F::from_u32(111),
    ];

    let advice = vec![
        F::from_u32(13),
        F::from_u32(113),
        F::from_u32(113),
        F::from_u32(113),
        F::from_u32(113),
        F::from_u32(113),
        F::from_u32(113),
        F::from_u32(113),
    ];

    let instance = vec![
        F::from_u32(17),
        F::from_u32(117),
        F::from_u32(117),
        F::from_u32(117),
        F::from_u32(117),
        F::from_u32(117),
        F::from_u32(117),
        F::from_u32(117),
    ];

    let challenges = vec![F::from_u32(19)];

    let beta           = vec![F::from_u32(101)];
    let gamma          = vec![F::from_u32(102)];
    let theta          = vec![F::from_u32(103)];
    let y              = vec![F::from_u32(104)];
    let previous_value = vec![F::from_u32(105)];

    let rotations = vec![0];

    let calculations = vec![0, 2, 3, 4, 7, 6];
    let targets = vec![0, 1, 2, 3, 4, 5];
    let value_types = vec![0, 0, 2, 3, 1, 1, 1, 1, 1, 1, 1, 0];
    let value_indices = vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 4, 0, 2, 0];

    let mut horner_offsets = vec![0; 6];
    let mut horner_sizes   = vec![0; 6];
    horner_offsets[5] = 0; 
    horner_sizes[5]   = 2;
    let horner_value_types   = vec![4, 5];
    let horner_value_indices = vec![0, 0, 0, 0];

    // Build the GateData:
    let gate_data = GateData::new(
        constants.as_ptr(),
        constants.len() as u32,
        fixed.as_ptr(),
        1,
        table_size as u32,
        advice.as_ptr(),
        1,
        table_size as u32,
        instance.as_ptr(),
        1,
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
        6,
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        targets.as_ptr(),
        value_types.as_ptr(),
        value_indices.as_ptr(),
        calculations.len() as u32,
        6,
    );

    let horner_data = HornerData::new(
        horner_value_types.as_ptr(),
        horner_value_indices.as_ptr(),
        horner_offsets.as_ptr(),
        horner_sizes.as_ptr(),
        horner_value_types.len() as u32,
    );

    let mut result = vec![F::zero(); test_size];
    let mut result_slice = HostSlice::from_mut_slice(&mut result);

    test_utilities::test_set_main_device();
    let cfg = GateOpsConfig::default();

    gate_evaluation(&gate_data, &calc_data, &horner_data, result_slice, &cfg)
        .expect("Gate evaluation failed");

    let expected = F::from_u32(1366);
    assert_eq!(result_slice[0], expected, "Final result mismatch!");
}