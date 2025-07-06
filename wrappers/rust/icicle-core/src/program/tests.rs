use crate::program::{PreDefinedProgram, Program};
use crate::ring::IntegerRing;
use crate::symbol::Symbol;
use crate::traits::{Arithmetic, GenerateRandom, Invertible};
use crate::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::memory::HostSlice;

pub fn check_program<F, Prog>()
where
    F: IntegerRing + Invertible,
    F: VecOps<F> + GenerateRandom,
    Prog: Program<F>,
    Prog::ProgSymbol: Symbol<F> + Arithmetic + Invertible,
{
    let example_lambda = |vars: &mut Vec<Prog::ProgSymbol>| {
        let a = vars[0]; // Shallow copies pointing to the same memory in the backend
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];

        vars[4] = d * (a * b - c) + F::from(9);
        vars[5] = a * b - c.inv();
        vars[6] += a * b - c.inv();
        vars[3] = (vars[0] + vars[1]) * F::from(2); // all variables can be both inputs and outputs
    };

    const TEST_SIZE: usize = 1 << 10;
    let a = F::generate_random(TEST_SIZE);
    let b = F::generate_random(TEST_SIZE);
    let c = F::generate_random(TEST_SIZE);
    let eq = F::generate_random(TEST_SIZE);
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
    program
        .execute_program(&mut parameters, &cfg)
        .expect("Program Failed");

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
            F::from(2) * (a + b) // 2 * (a + b)
        );
        assert_eq!(
            var4,
            eq * (a * b - c) + F::from(9) // eq * (a * b - c) + 9
        );
        assert_eq!(
            var5,
            a * b - c.inv() // a * b - c.inv()
        );
        assert_eq!(var6, var5);
    }
}

pub fn check_predefined_program<F, Prog>()
where
    F: IntegerRing,
    F: VecOps<F> + GenerateRandom + Arithmetic,
    Prog: Program<F>,
{
    const TEST_SIZE: usize = 1 << 10;
    let a = F::generate_random(TEST_SIZE);
    let b = F::generate_random(TEST_SIZE);
    let c = F::generate_random(TEST_SIZE);
    let eq = F::generate_random(TEST_SIZE);
    let var4 = vec![F::zero(); TEST_SIZE];
    let a_slice = HostSlice::from_slice(&a);
    let b_slice = HostSlice::from_slice(&b);
    let c_slice = HostSlice::from_slice(&c);
    let eq_slice = HostSlice::from_slice(&eq);
    let var4_slice = HostSlice::from_slice(&var4);
    let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice];

    let program = Prog::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();

    let cfg = VecOpsConfig::default();
    program
        .execute_program(&mut parameters, &cfg)
        .expect("Program Failed");

    for i in 0..TEST_SIZE {
        let a = parameters[0][i];
        let b = parameters[1][i];
        let c = parameters[2][i];
        let eq = parameters[3][i];
        let var4 = parameters[4][i];
        assert_eq!(var4, eq * (a * b - c));
    }
}
