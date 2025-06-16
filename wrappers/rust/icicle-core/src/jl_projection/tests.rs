use crate::jl_projection::JLProjection;
use crate::traits::{Arithmetic, FieldImpl};
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::memory::HostSlice;
use rand::Rng;

pub fn check_jl_projection<T, F>()
where
    T: JLProjection<F>,
    F: FieldImpl + Arithmetic,
{
    let input_size = 1 << 10;
    let output_size = 256;
    let cfg = VecOpsConfig::default();

    let zero = F::zero();
    let one = F::one();
    let minus_one = zero - one;

    let input = vec![one; input_size];
    let mut output = vec![zero; output_size];
    let mut matrix = vec![zero; input_size * output_size];

    // Seed for matrix generation
    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    // Step 1: Run JL projection
    T::jl_projection(
        HostSlice::from_slice(&input),
        &seed,
        &cfg,
        HostSlice::from_mut_slice(&mut output),
    )
    .expect("JL projection failed");

    // Step 2: Get JL matrix rows
    T::get_jl_matrix_rows(
        &seed,
        input_size,
        0,
        output_size,
        &cfg,
        HostSlice::from_mut_slice(&mut matrix),
    );

    // Step 3: Check matrix elements are only in {0, 1, -1}
    for (i, &elem) in matrix
        .iter()
        .enumerate()
    {
        assert!(
            elem == F::zero() || elem == one || elem == minus_one,
            "matrix[{}] = {:?} not in {{0, ±1}}",
            i,
            elem
        );
    }

    // Step 4: Recompute output: since input = all 1s, row sum = dot(matrix_row, input)
    for row in 0..output_size {
        let mut acc = F::zero();
        for col in 0..input_size {
            acc = acc + matrix[row * input_size + col];
        }
        assert_eq!(
            output[row], acc,
            "JL projection mismatch at row {}: got {:?}, expected {:?}",
            row, output[row], acc
        );
    }
}

// TODO Yuval: test JL-projection conjugated matrix in Rq
