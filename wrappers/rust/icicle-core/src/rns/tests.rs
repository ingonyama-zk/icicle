use crate::{
    rns::{from_rns, to_rns, RnsConversion},
    traits::{FieldImpl, GenerateRandom},
    vec_ops::{VecOps, VecOpsConfig},
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
    Zq::Config: RnsConversion<Zq, ZqRns> + GenerateRandom<Zq>,
{
    let size = 1 << 12;
    let batch_size = 3;
    let total_size = size * batch_size;

    let input_direct: Vec<Zq> = Zq::Config::generate_random(total_size);

    // Convert Zq -> ZqRns on reference device
    test_utilities::test_set_ref_device();
    let mut output_rns_ref = vec![ZqRns::zero(); total_size];
    to_rns(
        HostSlice::from_slice(&input_direct),
        HostSlice::from_mut_slice(&mut output_rns_ref),
        &VecOpsConfig::default(),
    )
    .unwrap();

    // Convert Zq -> ZqRns on main device
    test_utilities::test_set_main_device();
    let mut output_rns_main_d = DeviceVec::<ZqRns>::device_malloc(total_size).unwrap();
    let mut output_rns_main_h = vec![ZqRns::zero(); total_size];
    to_rns(
        HostSlice::from_slice(&input_direct),
        &mut output_rns_main_d[..],
        &VecOpsConfig::default(),
    )
    .unwrap();

    // Ensure reference and main device implementations produce identical results
    output_rns_main_d
        .copy_to_host(HostSlice::from_mut_slice(&mut output_rns_main_h))
        .unwrap();

    assert_eq!(output_rns_ref, output_rns_main_h);

    // Convert ZqRns -> Zq on main device and compare with original input
    let mut converted_back_from_rns = vec![Zq::zero(); total_size];
    assert_ne!(
        input_direct, converted_back_from_rns,
        "Initial output buffer should not match input."
    );

    from_rns(
        &mut output_rns_main_d[..],
        HostSlice::from_mut_slice(&mut converted_back_from_rns),
        &VecOpsConfig::default(),
    )
    .unwrap();

    assert_eq!(
        input_direct, converted_back_from_rns,
        "Final conversion should match the original input."
    );
}

/// This test verifies that arithmetic operations in the RNS representation
/// are consistent with direct computation in the original ring.
/// Specifically, we check that:
///   - `c = a * b` computed directly in `Zq`
///   - Matches `c_from_rns = from_rns(a_rns * b_rns)` computed in `ZqRns`
pub fn check_rns_arithmetic_consistency<Zq, ZqRns>()
where
    Zq: FieldImpl,
    Zq::Config: VecOps<Zq> + RnsConversion<Zq, ZqRns> + GenerateRandom<Zq>,
    ZqRns: FieldImpl,
    ZqRns::Config: VecOps<ZqRns>,
{
    // Set reference device for computations
    test_utilities::test_set_ref_device();

    let batch = 6;
    let size = batch * ((1 << 11) + 7);

    // Allocate inputs in Zq
    let a = Zq::Config::generate_random(size);
    let b = Zq::Config::generate_random(size);
    let mut c = vec![Zq::zero(); size];

    // Allocate corresponding RNS representations
    let mut a_rns = vec![ZqRns::zero(); size];
    let mut b_rns = vec![ZqRns::zero(); size];
    let mut c_rns = vec![ZqRns::zero(); size];

    // Compute in Zq: c = a * b
    Zq::Config::mul(
        HostSlice::from_slice(&a),
        HostSlice::from_slice(&b),
        HostSlice::from_mut_slice(&mut c),
        &VecOpsConfig::default(),
    )
    .unwrap();

    // Convert a, b to RNS representation (a_rns, b_rns)
    Zq::Config::to_rns(
        HostSlice::from_slice(&a),
        HostSlice::from_mut_slice(&mut a_rns),
        &VecOpsConfig::default(),
    )
    .unwrap();
    Zq::Config::to_rns(
        HostSlice::from_slice(&b),
        HostSlice::from_mut_slice(&mut b_rns),
        &VecOpsConfig::default(),
    )
    .unwrap();

    // Compute in RNS domain: c_rns = a_rns * b_rns
    ZqRns::Config::mul(
        HostSlice::from_slice(&a_rns),
        HostSlice::from_slice(&b_rns),
        HostSlice::from_mut_slice(&mut c_rns),
        &VecOpsConfig::default(),
    )
    .unwrap();

    // Convert c_rns back to Zq: c_from_rns = from_rns(c_rns)
    let mut c_from_rns = vec![Zq::zero(); size];
    Zq::Config::from_rns(
        HostSlice::from_slice(&c_rns),
        HostSlice::from_mut_slice(&mut c_from_rns),
        &VecOpsConfig::default(),
    )
    .unwrap();

    // Verify consistency: c (computed in Zq) == c_from_rns (computed via RNS)
    assert_eq!(c, c_from_rns, "Mismatch between direct and RNS computation results.");
}
