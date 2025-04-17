#![allow(unused_imports)]
use crate::traits::GenerateRandom;
use crate::gate_ops::{gate_evaluation, lookups_constraint, GateOps, GateOpsConfig, GateData, CalculationData, HornerData, LookupConfig, LookupData, FieldImpl};
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
    check_horner_evaluation::<F>(4096);
    // check_ezkl_gate_ops::<F>(2^25);
}

pub fn check_lookup_constraints_scalars<F: FieldImpl>()
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    check_sample_lookup_constraints::<F>(1);
}

pub fn check_sample_lookup_constraints<F: FieldImpl>(_test_size: usize)
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let table_values = vec![
        F::from_u32(2u32),
    ];

    let inputs_prods = vec![
        F::from_u32(2u32),
    ];

    let inputs_inv_sums = vec![
        F::from_u32(2u32),
    ];

    let phi_coset = vec![
        F::from_u32(2u32),
    ];

    let m_coset = vec![
        F::from_u32(2u32),
    ];

    let l0 = vec![
        F::from_u32(2u32),
    ];

    let l_last = vec![
        F::from_u32(2u32),
    ];

    let l_active_row = vec![
        F::from_u32(2u32),
    ];

    let previous_value = vec![
        F::from_u32(2u32),
    ];

    let y = vec![
        F::from_u32(2u32),
    ];

    let rot_scale = 1;
    let i_size = 1; 

    let lookup_data = LookupData::new(
        table_values.as_ptr(),
        table_values.len() as u32,
        table_values.as_ptr(),
        inputs_prods.len() as u32,
        inputs_inv_sums.as_ptr(),
        inputs_inv_sums.len() as u32,
        phi_coset.as_ptr(),
        phi_coset.len() as u32,
        m_coset.as_ptr(),
        m_coset.len() as u32,
        l0.as_ptr(),
        l0.len() as u32,
        l_last.as_ptr(),
        l_last.len() as u32,
        l_active_row.as_ptr(), 
        l_active_row.len() as u32,
        y.as_ptr(),
        previous_value.as_ptr(),
        previous_value.len() as u32,
        rot_scale as u32,
        i_size as u32,
    );

    let mut values = vec![ F::from_u32(0u32) ];
    let h_values = HostSlice::from_mut_slice(&mut values[..]);
    let mut cfg = LookupConfig::default();
    cfg.is_result_on_device = false;

    lookups_constraint(&lookup_data, h_values, &cfg).unwrap(); 

    println!("h_values: {:?}", h_values);
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

    let rotations: Vec<i32> = vec![0];
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
        fixed.as_ptr(),
        1,
        table_size as u32,
        advice.as_ptr(),
        1,
        table_size as u32,
        instance.as_ptr(),
        1,
        table_size as u32,
        challenges.as_ptr(),
        challenges.len() as u32,
        beta.as_ptr(),
        gamma.as_ptr(),
        theta.as_ptr(),
        y.as_ptr(),
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        targets.as_ptr(),
        value_types.as_ptr(),
        value_indices.as_ptr(),
        constants.as_ptr(),
        constants.len() as u32,
        rotations.as_ptr(),
        rotations.len() as u32,
        previous_value.as_ptr(),
        calculations.len() as u32,
        2,
        test_size as u32,
        1,
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
    let result_slice = HostSlice::from_mut_slice(&mut result);

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

    println!("result: {:?}", result);
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
        fixed.as_ptr(),
        1,
        table_size as u32,
        advice.as_ptr(),
        1,
        table_size as u32,
        instance.as_ptr(),
        1,
        table_size as u32,
        challenges.as_ptr(),
        challenges.len() as u32,
        beta.as_ptr(),
        gamma.as_ptr(),
        theta.as_ptr(),
        y.as_ptr(),
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        targets.as_ptr(),
        value_types.as_ptr(),
        value_indices.as_ptr(),
        constants.as_ptr(),
        constants.len() as u32,
        rotations.as_ptr(),
        rotations.len() as u32,
        previous_value.as_ptr(),
        calculations.len() as u32,
        6,
        test_size as u32,
        1,
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
    let result_slice = HostSlice::from_mut_slice(&mut result);

    test_utilities::test_set_main_device();
    let cfg = GateOpsConfig::default();

    gate_evaluation(&gate_data, &calc_data, &horner_data, result_slice, &cfg)
        .expect("Gate evaluation failed");

    let expected = F::from_u32(1366);
    assert_eq!(result_slice[0], expected, "Final result mismatch!");
}

pub fn check_ezkl_gate_ops<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: GateOps<F> + GenerateRandom<F>,
{
    let constants = vec![
        F::from_u32(0),
        F::from_u32(1),
        F::from_u32(2),
        F::from_u32(3),
    ];

    let fixed = vec![F::from_u32(1); 19 * 1048576];
    let advice = vec![F::from_u32(1); 6 * 1048576];
    let instance = vec![F::from_u32(1); 1 * 1048576];

    let challenges = vec![];

    let beta           = vec![F::from_u32(1)];
    let gamma          = vec![F::from_u32(1)];
    let theta          = vec![F::from_u32(1)];
    let y              = vec![F::from_u32(1)];
    let previous_value = vec![F::from_u32(0); test_size];

    let rotations = vec![0, -1];

    let calculations = vec![7, 1, 2, 1, 2, 7, 7, 7, 0, 1, 2, 7, 1, 2, 1, 2, 7, 7, 7, 0, 1, 2, 1, 2, 2, 2, 1, 2, 7, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 7, 1, 2, 7, 0, 0, 1, 2, 2, 1, 2, 7, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 6];
    let targets = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90];
    let value_types = vec![2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 9];
    let value_indices = vec![14, 0, 14, 0, 2, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 2, 0, 3, 0, 4, 0, 4, 0, 0, 0, 0, 0, 2, 0, 2, 0, 6, 0, 7, 0, 5, 0, 8, 0, 4, 0, 9, 0, 15, 0, 15, 0, 1, 0, 11, 0, 11, 0, 12, 0, 3, 0, 11, 0, 13, 0, 14, 0, 5, 0, 5, 0, 1, 0, 1, 0, 3, 0, 3, 0, 17, 0, 18, 0, 16, 0, 19, 0, 15, 0, 20, 0, 1, 0, 0, 0, 0, 0, 22, 0, 1, 0, 23, 0, 6, 0, 7, 0, 5, 0, 25, 0, 24, 0, 26, 0, 16, 0, 16, 0, 2, 0, 28, 0, 28, 0, 29, 0, 3, 0, 28, 0, 30, 0, 31, 0, 17, 0, 18, 0, 16, 0, 33, 0, 32, 0, 34, 0, 3, 0, 23, 0, 6, 0, 7, 0, 5, 0, 37, 0, 36, 0, 38, 0, 2, 0, 11, 0, 13, 0, 40, 0, 17, 0, 18, 0, 16, 0, 42, 0, 41, 0, 43, 0, 11, 0, 40, 0, 14, 0, 45, 0, 5, 0, 1, 0, 5, 0, 47, 0, 46, 0, 48, 0, 1, 0, 28, 0, 28, 0, 50, 0, 31, 0, 51, 0, 16, 0, 1, 0, 16, 0, 53, 0, 52, 0, 54, 0, 17, 0, 17, 0, 2, 0, 56, 0, 56, 0, 57, 0, 4, 1, 4, 1, 25, 0, 33, 0, 59, 0, 60, 0, 5, 0, 61, 0, 58, 0, 62, 0, 29, 0, 51, 0, 5, 0, 60, 0, 64, 0, 65, 0, 18, 0, 18, 0, 2, 0, 67, 0, 67, 0, 68, 0, 3, 0, 67, 0, 69, 0, 70, 0, 7, 0, 18, 0, 5, 0, 72, 0, 71, 0, 73, 0, 1, 0, 56, 0, 56, 0, 75, 0, 59, 0, 72, 0, 5, 0, 77, 0, 76, 0, 78, 0, 1, 0, 67, 0, 67, 0, 80, 0, 68, 0, 81, 0, 7, 0, 18, 0, 5, 0, 83, 0, 82, 0, 84, 0, 70, 0, 81, 0, 59, 0, 83, 0, 5, 0, 87, 0, 86, 0, 88, 0, 0, 0, 0, 0];

    let horner_value_types = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let horner_value_indices = vec![10, 0, 21, 0, 27, 0, 35, 0, 39, 0, 44, 0, 49, 0, 55, 0, 63, 0, 66, 0, 74, 0, 79, 0, 85, 0, 89, 0];
    let horner_offsets   = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let horner_sizes = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14];

    // println!("calculations: {:?}", calculations);
    // println!("targets: {:?}", targets);
    // println!("value_types: {:?}", value_types);
    // println!("value_indices: {:?}", value_indices);
    // println!("rotations: {:?}", rotations);
    // println!("horner_value_types: {:?}", horner_value_types);
    // println!("horner_value_indices: {:?}", horner_value_indices);
    // println!("horner_offsets: {:?}", horner_offsets);
    // println!("horner_sizes: {:?}", horner_sizes);

    // Build the GateData:
    let gate_data = GateData::new(
        fixed.as_ptr(),
        19,
        1048576,
        advice.as_ptr(),
        6,
        1048576,
        instance.as_ptr(),
        1,
        1048576,
        challenges.as_ptr(),
        0,
        beta.as_ptr(),
        gamma.as_ptr(),
        theta.as_ptr(),
        y.as_ptr(),
    );

    let calc_data = CalculationData::new(
        calculations.as_ptr(),
        targets.as_ptr(),
        value_types.as_ptr(),
        value_indices.as_ptr(),
        constants.as_ptr(),
        constants.len() as u32,
        rotations.as_ptr(),
        rotations.len() as u32,
        previous_value.as_ptr(),
        calculations.len() as u32,
        targets.len() as u32,
        test_size as u32,
        4,
        1048576,
    );

    let horner_data = HornerData::new(
        horner_value_types.as_ptr(),
        horner_value_indices.as_ptr(),
        horner_offsets.as_ptr(),
        horner_sizes.as_ptr(),
        horner_value_types.len() as u32,
    );

    println!("gate_data: {:?}", gate_data);
    println!("calc_data: {:?}", calc_data);
    println!("horner_data: {:?}", horner_data);

    let mut result = vec![F::zero(); test_size];
    let result_slice = HostSlice::from_mut_slice(&mut result);

    test_utilities::test_set_main_device();
    let cfg = GateOpsConfig::default();

    gate_evaluation(&gate_data, &calc_data, &horner_data, result_slice, &cfg)
        .expect("Gate evaluation failed");

    println!("result_slice: {:?}", result_slice);
}