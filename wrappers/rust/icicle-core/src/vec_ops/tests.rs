#![allow(unused_imports)]
use crate::field::FieldArithmetic;
use crate::polynomial_ring::PolynomialRing;
use crate::program::{Instruction, PreDefinedProgram, Program, ReturningValueProgram};
use crate::symbol::Symbol;
use crate::traits::{Arithmetic, GenerateRandom};
use crate::vec_ops::poly_vecops::{polyvec_add, polyvec_mul, polyvec_mul_by_scalar, polyvec_sub, polyvec_sum_reduce};
use crate::vec_ops::{
    accumulate_scalars, add_scalars, bit_reverse, bit_reverse_inplace, div_scalars, execute_program, inv_scalars,
    mixed_mul_scalars, mul_scalars, product_scalars, scalar_add, scalar_mul, scalar_sub, slice, sub_scalars,
    sum_scalars, transpose_matrix, FieldImpl, MixedVecOps, VecOps, VecOpsConfig,
};
use icicle_runtime::device::Device;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::{runtime, stream::IcicleStream, test_utilities};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[test]
fn test_vec_ops_config() {
    let mut vec_ops_config = VecOpsConfig::default();
    vec_ops_config
        .ext
        .set_int("int_example", 5);

    assert_eq!(
        vec_ops_config
            .ext
            .get_int("int_example"),
        5
    );

    // just to test the stream can be set and used correctly
    let stream = IcicleStream::create().unwrap();
    vec_ops_config.stream_handle = *stream;

    stream
        .synchronize()
        .unwrap();
}

pub fn check_mixed_vec_ops_scalars<F: FieldImpl, T: FieldImpl>()
where
    <F as FieldImpl>::Config: MixedVecOps<F, T>,
    <T as FieldImpl>::Config: GenerateRandom<T>,
    <F as FieldImpl>::Config: GenerateRandom<F>,
{
    let test_size = 1 << 14;

    check_vec_ops_mixed_scalars_mul::<F, T>(test_size);
}

pub fn check_vec_ops_scalars_add<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let a_main = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::zero(); test_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    let cfg = VecOpsConfig::default();

    test_utilities::test_set_main_device();
    add_scalars(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    add_scalars(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_sub<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let a_main = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::zero(); test_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    let cfg = VecOpsConfig::default();

    test_utilities::test_set_main_device();
    sub_scalars(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    sub_scalars(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_mul<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let a_main = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::zero(); test_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    let cfg = VecOpsConfig::default();

    test_utilities::test_set_main_device();
    mul_scalars(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    mul_scalars(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_div<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let a_main = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::zero(); test_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    let cfg = VecOpsConfig::default();

    test_utilities::test_set_main_device();
    div_scalars(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    div_scalars(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_inv<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();

    let a = F::Config::generate_random(test_size);
    let mut inv = vec![F::zero(); test_size];
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::one(); test_size];
    let mut result = vec![F::one(); test_size];

    let a = HostSlice::from_slice(&a);
    let inv = HostSlice::from_mut_slice(&mut inv);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);
    let result = HostSlice::from_mut_slice(&mut result);

    test_utilities::test_set_main_device();
    inv_scalars(a, inv, &cfg).unwrap();
    mul_scalars(a, inv, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    inv_scalars(a, inv, &cfg).unwrap();
    mul_scalars(a, inv, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result.as_slice());
    assert_eq!(result_ref.as_slice(), result.as_slice());
}

pub fn check_vec_ops_scalars_sum<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let a_main = F::Config::generate_random(test_size * batch_size);
    let mut result_main = vec![F::zero(); batch_size];
    let mut result_ref = vec![F::zero(); batch_size];

    let a_main = HostSlice::from_slice(&a_main);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    test_utilities::test_set_main_device();
    sum_scalars(a_main, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    sum_scalars(a_main, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_product<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let a_main = F::Config::generate_random(test_size * batch_size);
    let mut result_main = vec![F::zero(); batch_size];
    let mut result_ref = vec![F::zero(); batch_size];

    let a_main = HostSlice::from_slice(&a_main);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    test_utilities::test_set_main_device();
    product_scalars(a_main, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    product_scalars(a_main, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_add_scalar<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let a_main = F::Config::generate_random(batch_size as usize);
    let b = F::Config::generate_random(test_size * batch_size as usize);
    let mut result_main = vec![F::zero(); test_size * batch_size as usize];
    let mut result_ref = vec![F::zero(); test_size * batch_size as usize];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    test_utilities::test_set_main_device();
    scalar_add(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    scalar_add(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_sub_scalar<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let a_main = F::Config::generate_random(batch_size);
    let b = F::Config::generate_random(test_size * batch_size);
    let mut result_main = vec![F::zero(); test_size * batch_size];
    let mut result_ref = vec![F::zero(); test_size * batch_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    test_utilities::test_set_main_device();
    scalar_sub(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    scalar_sub(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_mul_scalar<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let a_main = F::Config::generate_random(batch_size);
    let b = F::Config::generate_random(test_size * batch_size);
    let mut result_main = vec![F::zero(); test_size * batch_size];
    let mut result_ref = vec![F::zero(); test_size * batch_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    test_utilities::test_set_main_device();
    scalar_mul(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    scalar_mul(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

pub fn check_vec_ops_scalars_accumulate<F: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let mut a_main = F::Config::generate_random(test_size);
    let b = F::Config::generate_random(test_size);

    let mut a_clone = a_main.clone();

    let a_main_slice = HostSlice::from_mut_slice(&mut a_main);
    let b_slice = HostSlice::from_slice(&b);
    let a_clone_slice = HostSlice::from_mut_slice(&mut a_clone);

    let cfg = VecOpsConfig::default();

    test_utilities::test_set_main_device();
    accumulate_scalars(a_main_slice, b_slice, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    accumulate_scalars(a_clone_slice, b_slice, &cfg).unwrap();

    assert_eq!(a_clone_slice.as_slice(), a_main_slice.as_slice());
}

pub fn check_matrix_transpose<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let (r, c): (u32, u32) = (1u32 << 10, 1u32 << 4);
    let test_size = (r * c * batch_size) as usize;

    let input_matrix = F::Config::generate_random(test_size);
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::zero(); test_size];

    test_utilities::test_set_main_device();
    transpose_matrix(
        HostSlice::from_slice(&input_matrix),
        r,
        c,
        HostSlice::from_mut_slice(&mut result_main),
        &cfg,
    )
    .unwrap();

    test_utilities::test_set_ref_device();
    transpose_matrix(
        HostSlice::from_slice(&input_matrix),
        r,
        c,
        HostSlice::from_mut_slice(&mut result_ref),
        &cfg,
    )
    .unwrap();

    assert_eq!(result_main, result_ref);
}

pub fn check_slice<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    let cfg = VecOpsConfig::default();
    let batch_size = 3;

    let size_in: u64 = 1 << 10;
    let offset: u64 = 10;
    let stride: u64 = 3;
    let size_out: u64 = ((size_in - offset) / stride) - 1;

    let input_matrix = F::Config::generate_random(size_in as usize * batch_size);
    let mut result_main = vec![F::zero(); size_out as usize * batch_size];
    let mut result_ref = vec![F::zero(); size_out as usize * batch_size];

    test_utilities::test_set_main_device();
    slice(
        HostSlice::from_slice(&input_matrix),
        offset,
        stride,
        size_in,
        size_out,
        &cfg,
        HostSlice::from_mut_slice(&mut result_main),
    )
    .unwrap();

    test_utilities::test_set_ref_device();
    slice(
        HostSlice::from_slice(&input_matrix),
        offset,
        stride,
        size_in,
        size_out,
        &cfg,
        HostSlice::from_mut_slice(&mut result_ref),
    )
    .unwrap();

    assert_eq!(result_main, result_ref);
}

pub fn check_bit_reverse<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    test_utilities::test_set_main_device();

    const LOG_SIZE: u32 = 20;
    const TEST_SIZE: usize = 1 << LOG_SIZE;
    let input_vec = F::Config::generate_random(TEST_SIZE);
    let input = HostSlice::from_slice(&input_vec);
    let mut intermediate = DeviceVec::<F>::device_malloc(TEST_SIZE).unwrap();
    let cfg = VecOpsConfig::default();
    bit_reverse(input, &cfg, &mut intermediate[..]).unwrap();

    let mut intermediate_host = vec![F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut intermediate_host[..]))
        .unwrap();
    let index_reverser = |i: usize| i.reverse_bits() >> (usize::BITS - LOG_SIZE);
    intermediate_host
        .iter()
        .enumerate()
        .for_each(|(i, val)| assert_eq!(val, &input_vec[index_reverser(i)]));

    let mut result = vec![F::one(); TEST_SIZE];
    let result = HostSlice::from_mut_slice(&mut result);
    let cfg = VecOpsConfig::default();
    bit_reverse(&intermediate[..], &cfg, result).unwrap();
    assert_eq!(input.as_slice(), result.as_slice());
}

pub fn check_bit_reverse_inplace<F: FieldImpl>()
where
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
{
    test_utilities::test_set_main_device();

    const LOG_SIZE: u32 = 20;
    const TEST_SIZE: usize = 1 << LOG_SIZE;
    let input_vec = F::Config::generate_random(TEST_SIZE);
    let input = HostSlice::from_slice(&input_vec);
    let mut intermediate = DeviceVec::<F>::device_malloc(TEST_SIZE).unwrap();
    intermediate
        .copy_from_host(&input)
        .unwrap();
    let cfg = VecOpsConfig::default();
    bit_reverse_inplace(&mut intermediate[..], &cfg).unwrap();

    let mut intermediate_host = vec![F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut intermediate_host[..]))
        .unwrap();
    let index_reverser = |i: usize| i.reverse_bits() >> (usize::BITS - LOG_SIZE);
    intermediate_host
        .iter()
        .enumerate()
        .for_each(|(i, val)| assert_eq!(val, &input_vec[index_reverser(i)]));

    bit_reverse_inplace(&mut intermediate[..], &cfg).unwrap();
    let mut result_host = vec![F::one(); TEST_SIZE];
    intermediate
        .copy_to_host(HostSlice::from_mut_slice(&mut result_host[..]))
        .unwrap();
    assert_eq!(input.as_slice(), result_host.as_slice());
}

pub fn check_program<F, Prog>()
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F> + FieldArithmetic<F>,
    Prog: Program<F>,
{
    let example_lambda = |vars: &mut Vec<Prog::ProgSymbol>| {
        let a = vars[0]; // Shallow copies pointing to the same memory in the backend
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];

        vars[4] = d * (a * b - c) + F::from_u32(9);
        vars[5] = a * b - c.inverse();
        vars[6] += a * b - c.inverse();
        vars[3] = (vars[0] + vars[1]) * F::from_u32(2); // all variables can be both inputs and outputs
    };

    const TEST_SIZE: usize = 1 << 10;
    let a = F::Config::generate_random(TEST_SIZE);
    let b = F::Config::generate_random(TEST_SIZE);
    let c = F::Config::generate_random(TEST_SIZE);
    let eq = F::Config::generate_random(TEST_SIZE);
    let var4 = vec![F::zero(); TEST_SIZE];
    let var5 = vec![F::zero(); TEST_SIZE];
    let var6 = vec![F::zero(); TEST_SIZE];
    let a_slice = HostSlice::from_slice(&a);
    let b_slice = HostSlice::from_slice(&b);
    let c_slice = HostSlice::from_slice(&c);
    let eq_slice = HostSlice::from_slice(&eq);
    let var4_slice = HostSlice::from_slice(&var4);
    let var5_slice = HostSlice::from_slice(&var5);
    let var6_slice = HostSlice::from_slice(&var6);
    let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice, var5_slice, var6_slice];

    let program = Prog::new(example_lambda, 7).unwrap();

    let cfg = VecOpsConfig::default();
    execute_program(&mut parameters, &program, &cfg).expect("Program Failed");

    for i in 0..TEST_SIZE {
        let a = a[i];
        let b = b[i];
        let c = c[i];
        let eq = eq[i];
        let var3 = parameters[3][i];
        let var4 = parameters[4][i];
        let var5 = parameters[5][i];
        let var6 = parameters[6][i];
        assert_eq!(
            var3,
            <<F as FieldImpl>::Config as FieldArithmetic<F>>::mul(
                F::from_u32(2),
                <<F as FieldImpl>::Config as FieldArithmetic<F>>::add(a, b)
            )
        );
        assert_eq!(
            var4,
            <<F as FieldImpl>::Config as FieldArithmetic<F>>::add(
                F::from_u32(9),
                <<F as FieldImpl>::Config as FieldArithmetic<F>>::mul(
                    eq,
                    <<F as FieldImpl>::Config as FieldArithmetic<F>>::sub(
                        <<F as FieldImpl>::Config as FieldArithmetic<F>>::mul(a, b),
                        c
                    )
                )
            )
        );
        assert_eq!(
            var5,
            <<F as FieldImpl>::Config as FieldArithmetic<F>>::sub(
                <<F as FieldImpl>::Config as FieldArithmetic<F>>::mul(a, b),
                <<F as FieldImpl>::Config as FieldArithmetic<F>>::inv(c)
            )
        );
        assert_eq!(var6, var5);
    }
}

pub fn check_predefined_program<F, Prog>()
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F> + FieldArithmetic<F>,
    Prog: Program<F>,
{
    const TEST_SIZE: usize = 1 << 10;
    let a = F::Config::generate_random(TEST_SIZE);
    let b = F::Config::generate_random(TEST_SIZE);
    let c = F::Config::generate_random(TEST_SIZE);
    let eq = F::Config::generate_random(TEST_SIZE);
    let var4 = vec![F::zero(); TEST_SIZE];
    let a_slice = HostSlice::from_slice(&a);
    let b_slice = HostSlice::from_slice(&b);
    let c_slice = HostSlice::from_slice(&c);
    let eq_slice = HostSlice::from_slice(&eq);
    let var4_slice = HostSlice::from_slice(&var4);
    let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice];

    let program = Prog::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();

    let cfg = VecOpsConfig::default();
    execute_program(&mut parameters, &program, &cfg).expect("Program Failed");

    for i in 0..TEST_SIZE {
        let a = parameters[0][i];
        let b = parameters[1][i];
        let c = parameters[2][i];
        let eq = parameters[3][i];
        let var4 = parameters[4][i];
        assert_eq!(
            var4,
            <<F as FieldImpl>::Config as FieldArithmetic<F>>::mul(
                eq,
                <<F as FieldImpl>::Config as FieldArithmetic<F>>::sub(
                    <<F as FieldImpl>::Config as FieldArithmetic<F>>::mul(a, b),
                    c
                )
            )
        );
    }
}

pub fn check_vec_ops_mixed_scalars_mul<F: FieldImpl, T: FieldImpl>(test_size: usize)
where
    <F as FieldImpl>::Config: MixedVecOps<F, T> + GenerateRandom<F>,
    <T as FieldImpl>::Config: GenerateRandom<T>,
{
    let a_main = F::Config::generate_random(test_size);
    let b = T::Config::generate_random(test_size);
    let mut result_main = vec![F::zero(); test_size];
    let mut result_ref = vec![F::zero(); test_size];

    let a_main = HostSlice::from_slice(&a_main);
    let b = HostSlice::from_slice(&b);
    let result_main = HostSlice::from_mut_slice(&mut result_main);
    let result_ref = HostSlice::from_mut_slice(&mut result_ref);

    let cfg = VecOpsConfig::default();

    test_utilities::test_set_main_device();
    mixed_mul_scalars(a_main, b, result_main, &cfg).unwrap();

    test_utilities::test_set_ref_device();
    mixed_mul_scalars(a_main, b, result_ref, &cfg).unwrap();

    assert_eq!(result_main.as_slice(), result_ref.as_slice());
}

/// Tests `polyvec_add`, `polyvec_sub`, and `polyvec_mul` against manual computation
pub fn check_poly_vecops_add_sub_mul<P>()
where
    P: PolynomialRing + GenerateRandom<P> + PartialEq + core::fmt::Debug,
    P::Base: FieldImpl + Arithmetic,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    let size = 1 << 10;
    let a_vec = P::generate_random(size);
    let b_vec = P::generate_random(size);

    let cfg = VecOpsConfig::default();

    // Allocate buffers
    let mut add_result = vec![P::zero(); size];
    let mut sub_result = vec![P::zero(); size];
    let mut mul_result = vec![P::zero(); size];

    let mut expected_add = vec![P::zero(); size];
    let mut expected_sub = vec![P::zero(); size];
    let mut expected_mul = vec![P::zero(); size];

    // Run vectorized ops
    polyvec_add(
        HostSlice::from_slice(&a_vec),
        HostSlice::from_slice(&b_vec),
        HostSlice::from_mut_slice(&mut add_result),
        &cfg,
    )
    .expect("polyvec_add failed");

    polyvec_sub(
        HostSlice::from_slice(&a_vec),
        HostSlice::from_slice(&b_vec),
        HostSlice::from_mut_slice(&mut sub_result),
        &cfg,
    )
    .expect("polyvec_sub failed");

    polyvec_mul(
        HostSlice::from_slice(&a_vec),
        HostSlice::from_slice(&b_vec),
        HostSlice::from_mut_slice(&mut mul_result),
        &cfg,
    )
    .expect("polyvec_mul failed");

    // Manually compute expected values
    for i in 0..size {
        let a = a_vec[i].values();
        let b = b_vec[i].values();

        let add = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| *x + *y)
            .collect::<Vec<_>>();
        let sub = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| *x - *y)
            .collect::<Vec<_>>();
        let mul = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| *x * *y)
            .collect::<Vec<_>>();

        expected_add[i] = P::from_slice(&add);
        expected_sub[i] = P::from_slice(&sub);
        expected_mul[i] = P::from_slice(&mul);
    }

    // Assertions
    assert_eq!(add_result, expected_add, "polyvec_add mismatch");
    assert_eq!(sub_result, expected_sub, "polyvec_sub mismatch");
    assert_eq!(mul_result, expected_mul, "polyvec_mul mismatch");
}

/// Tests polyvec_mul_by_scalar against reference implementation
pub fn check_polyvec_mul_by_scalar<P>()
where
    P: PolynomialRing + GenerateRandom<P>,
    P::Base: FieldImpl + Arithmetic,
    <P::Base as FieldImpl>::Config: VecOps<P::Base> + GenerateRandom<P::Base>,
{
    let size = 1 << 10;
    let polyvec = P::generate_random(size);
    let scalarvec = <P::Base as FieldImpl>::Config::generate_random(size);

    let cfg = VecOpsConfig::default();

    let mut result = vec![P::zero(); size];
    let mut expected_result = vec![P::zero(); size];

    // Run the polyvec_mul_by_scalar vector op
    polyvec_mul_by_scalar(
        HostSlice::from_slice(&polyvec),
        HostSlice::from_slice(&scalarvec),
        HostSlice::from_mut_slice(&mut result),
        &cfg,
    )
    .expect("polyvec_mul_by_scalar failed");

    // Reference result (manual loop)
    for i in 0..size {
        let scalar = &scalarvec[i];
        let poly = polyvec[i].values();
        let product = poly
            .iter()
            .map(|c| *c * *scalar)
            .collect::<Vec<_>>();
        expected_result[i] = P::from_slice(&product);
    }

    // Check correctness
    assert_eq!(result, expected_result, "polyvec_mul_by_scalar mismatch");
}

/// Tests polyvec_sum_reduce by summing all polynomials manually and comparing
pub fn check_polyvec_sum_reduce<P>()
where
    P: PolynomialRing + GenerateRandom<P>,
    P::Base: FieldImpl + Arithmetic,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    let size = 1 << 10;
    let polyvec = P::generate_random(size);
    let cfg = VecOpsConfig::default();

    let mut result = vec![P::zero(); 1]; // single reduced polynomial
    let mut expected = vec![P::zero(); 1];

    // Run the vectorized reduction
    polyvec_sum_reduce(
        HostSlice::from_slice(&polyvec),
        HostSlice::from_mut_slice(&mut result),
        &cfg,
    )
    .expect("polyvec_sum_reduce failed");

    // Manually sum all polynomials coefficient-wise
    let mut acc = vec![P::Base::zero(); P::DEGREE];
    for poly in &polyvec {
        for (i, coeff) in poly
            .values()
            .iter()
            .enumerate()
        {
            acc[i] = acc[i] + *coeff;
        }
    }
    expected[0] = P::from_slice(&acc);

    // Assert result matches manual sum
    assert_eq!(result, expected, "polyvec_sum_reduce mismatch");
}
