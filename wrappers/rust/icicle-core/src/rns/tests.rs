use crate::{
    rns::{from_rns, to_rns, RnsConversion},
    traits::{FieldImpl, GenerateRandom},
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    test_utilities,
};

/// Tests the correctness of RNS conversion by:
/// 1. Converting `Zq` to `ZqRns` using both the reference and main device implementations.
/// 2. Ensuring both implementations produce identical results.
/// 3. Converting `ZqRns` back to `Zq` and verifying the output matches the original input.
pub fn check_rns_conversion<Zq: FieldImpl, ZqRns: FieldImpl>()
where
    <Zq as FieldImpl>::Config: RnsConversion<Zq, ZqRns> + GenerateRandom<Zq>,
{
    let size = 1 << 12;
    let batch_size = 3;
    let total_size = size * batch_size;

    let cfg = VecOpsConfig::default();
    let input_direct: Vec<Zq> = Zq::Config::generate_random(total_size);

    // Convert Zq -> ZqRns on reference device
    test_utilities::test_set_ref_device();
    let mut output_rns_ref = vec![ZqRns::zero(); total_size];
    to_rns(
        HostSlice::from_slice(&input_direct),
        &cfg,
        HostSlice::from_mut_slice(&mut output_rns_ref),
    )
    .unwrap();

    // Convert Zq -> ZqRns on main device
    test_utilities::test_set_main_device();
    let mut output_rns_main_d = DeviceVec::<ZqRns>::device_malloc(total_size).unwrap();
    let mut output_rns_main_h = vec![ZqRns::zero(); total_size];
    to_rns(HostSlice::from_slice(&input_direct), &cfg, &mut output_rns_main_d[..]).unwrap();

    // Ensure reference and main device implementations produce identical results
    output_rns_main_d
        .copy_to_host(HostSlice::from_mut_slice(&mut output_rns_main_h))
        .unwrap();
    assert_eq!(output_rns_ref, output_rns_main_h);

    // Convert ZqRns -> Zq on main device and compare with original input
    let mut output_direct = vec![Zq::zero(); total_size];
    assert_ne!(
        input_direct, output_direct,
        "Initial output buffer should not match input."
    );

    from_rns(
        &mut output_rns_main_d[..],
        &cfg,
        HostSlice::from_mut_slice(&mut output_direct),
    )
    .unwrap();

    assert_eq!(
        input_direct, output_direct,
        "Final conversion should match the original input."
    );
}
