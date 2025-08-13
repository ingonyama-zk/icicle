use crate::{
    ring::IntegerRing,
    rns::{from_rns, to_rns, RnsConversion},
    traits::GenerateRandom,
    vec_ops::{VecOps, VecOpsConfig},
};
use icicle_runtime::{
    memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
    test_utilities,
};

/// Tests the correctness of RNS conversion by:
/// 1. Converting `Zq` to `ZqRns` using both the reference and main device implementations.
/// 2. Ensuring both implementations produce identical results.
/// 3. Converting `ZqRns` back to `Zq` and verifying the output matches the original input.
pub fn check_rns_conversion<Zq: IntegerRing, ZqRns: IntegerRing>()
where
    Zq: RnsConversion<Zq, ZqRns> + GenerateRandom,
{
    let size = 1 << 12;
    let batch_size = 3;
    let total_size = size * batch_size;

    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = batch_size as i32;
    let input_direct: Vec<Zq> = Zq::generate_random(total_size);

    // Convert Zq -> ZqRns on reference device
    test_utilities::test_set_ref_device();
    let mut output_rns_ref = vec![ZqRns::zero(); total_size];
    to_rns(input_direct.into_slice(), output_rns_ref.into_slice_mut(), &cfg).unwrap();

    // Convert Zq -> ZqRns on main device
    test_utilities::test_set_main_device();
    let mut output_rns_main_d = DeviceVec::<ZqRns>::malloc(total_size);
    to_rns(input_direct.into_slice(), output_rns_main_d.into_slice_mut(), &cfg).unwrap();

    // Ensure reference and main device implementations produce identical results
    let output_rns_main_h = output_rns_main_d.to_host_vec();

    assert_eq!(output_rns_ref, output_rns_main_h);

    // Convert ZqRns -> Zq on main device and compare with original input
    let mut converted_back_from_rns = vec![Zq::zero(); total_size];
    assert_ne!(
        input_direct, converted_back_from_rns,
        "Initial output buffer should not match input."
    );

    from_rns(
        output_rns_main_d.into_slice(),
        converted_back_from_rns.into_slice_mut(),
        &cfg,
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
    Zq: IntegerRing + VecOps<Zq> + RnsConversion<Zq, ZqRns> + GenerateRandom,
    ZqRns: IntegerRing + VecOps<ZqRns>,
{
    use crate::vec_ops::mul_scalars;

    // Set reference device for computations
    test_utilities::test_set_ref_device();

    let batch_size = 6;
    let size = batch_size * ((1 << 11) + 7);
    let mut cfg = VecOpsConfig::default();
    cfg.batch_size = batch_size as i32;

    // Allocate inputs in Zq
    let a = Zq::generate_random(size);
    let b = Zq::generate_random(size);
    let mut c = vec![Zq::zero(); size];

    // Allocate corresponding RNS representations
    let mut a_rns = vec![ZqRns::zero(); size];
    let mut b_rns = vec![ZqRns::zero(); size];
    let mut c_rns = vec![ZqRns::zero(); size];

    // Compute in Zq: c = a * b
    mul_scalars(a.into_slice(), b.into_slice(), c.into_slice_mut(), &cfg).unwrap();

    // Convert a, b to RNS representation (a_rns, b_rns)
    Zq::to_rns(a.into_slice(), a_rns.into_slice_mut(), &cfg).unwrap();
    Zq::to_rns(b.into_slice(), b_rns.into_slice_mut(), &cfg).unwrap();

    // Compute in RNS domain: c_rns = a_rns * b_rns
    mul_scalars(a_rns.into_slice(), b_rns.into_slice(), c_rns.into_slice_mut(), &cfg).unwrap();

    // Convert c_rns back to Zq: c_from_rns = from_rns(c_rns)
    let mut c_from_rns = vec![Zq::zero(); size];
    Zq::from_rns(c_rns.into_slice(), c_from_rns.into_slice_mut(), &cfg).unwrap();

    // Verify consistency: c (computed in Zq) == c_from_rns (computed via RNS)
    assert_eq!(c, c_from_rns, "Mismatch between direct and RNS computation results.");
}
