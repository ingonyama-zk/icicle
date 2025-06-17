use crate::jl_projection::{JLProjection, JLProjectionPolyRing};
use crate::polynomial_ring::PolynomialRing;
use crate::traits::{Arithmetic, FieldImpl};
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::memory::HostSlice;
use rand::Rng;

pub fn check_jl_projection<F>()
where
    F: FieldImpl + Arithmetic,
    F::Config: JLProjection<F>,
{
    let input_size = 1 << 10;
    let output_size = 256;
    let cfg = VecOpsConfig::default();

    let zero = F::zero();
    let one = F::one();
    let minus_one = zero - one;

    let input = vec![one; input_size];
    let mut output = vec![zero; output_size];
    let mut matrix = vec![one; input_size * output_size];

    // Seed for matrix generation
    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    // Step 1: Run JL projection
    F::Config::jl_projection(
        HostSlice::from_slice(&input),
        &seed,
        &cfg,
        HostSlice::from_mut_slice(&mut output),
    )
    .expect("JL projection failed");

    // Step 2: Get JL matrix rows
    F::Config::get_jl_matrix_rows(
        &seed,
        input_size,
        0,
        output_size,
        &cfg,
        HostSlice::from_mut_slice(&mut matrix),
    )
    .expect("JL-projection failed");

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

fn conjugate_poly<P: PolynomialRing>(poly: P) -> P
where
    P::Base: FieldImpl + Arithmetic,
{
    // negate and flip coeffs, except for coeff0
    let d = P::DEGREE;
    let coeffs = poly.values();
    let minus_one = P::Base::zero() - P::Base::one();
    let conjugated_coeffs: Vec<P::Base> = (0..d)
        .map(|i| {
            let j = d - i;
            if i == 0 {
                coeffs[0]
            } else {
                coeffs[j] * minus_one
            }
        })
        .collect();
    P::from_slice(&conjugated_coeffs)
}

pub fn check_jl_projection_polyring<Poly>()
where
    Poly: PolynomialRing + JLProjectionPolyRing<Poly>,
    Poly::Base: FieldImpl + Arithmetic,
    <Poly::Base as FieldImpl>::Config: JLProjection<Poly::Base>,
{
    let d = Poly::DEGREE;
    let num_rows = 10;
    let row_size = 13; // Number of polynomials per row
    let total_polys = num_rows * row_size;
    let total_scalars = total_polys * d;

    let cfg = VecOpsConfig::default();

    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    for conjugate in [false, true] {
        // Step 1: Generate raw scalar JL matrix rows
        let mut scalar_data = vec![Poly::Base::zero(); total_scalars];
        <Poly::Base as FieldImpl>::Config::get_jl_matrix_rows(
            &seed,
            row_size * d,
            0,
            num_rows,
            &cfg,
            HostSlice::from_mut_slice(&mut scalar_data),
        )
        .expect("scalar JL matrix gen failed");

        // Step 2: Convert into Poly instances (with optional conjugation)
        let expected: Vec<Poly> = (0..total_polys)
            .map(|i| {
                let offset = i * d;
                let coeffs = &scalar_data[offset..offset + d];
                let poly = Poly::from_slice(coeffs);
                if conjugate {
                    conjugate_poly(poly)
                } else {
                    poly
                }
            })
            .collect();

        // Step 3: Get polyring JL rows from API
        let mut actual = vec![Poly::zero(); total_polys];
        Poly::get_jl_matrix_rows_as_polyring(
            &seed,
            row_size,
            0,
            num_rows,
            conjugate,
            &cfg,
            HostSlice::from_mut_slice(&mut actual),
        )
        .expect("polyring JL matrix gen failed");

        // Step 4: Compare
        for i in 0..total_polys {
            assert_eq!(
                actual[i], expected[i],
                "Mismatch at index {} (conjugate = {}):\nExpected: {:?}\nActual: {:?}",
                i, conjugate, expected[i], actual[i]
            );
        }
    }
}
