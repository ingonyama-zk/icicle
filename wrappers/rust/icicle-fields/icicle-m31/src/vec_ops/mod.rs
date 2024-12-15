use crate::field::{
    ComplexExtensionCfg, ComplexExtensionField, ExtensionCfg, QuarticExtensionField, ScalarCfg, ScalarField,
};

use icicle_core::error::IcicleResult;
use icicle_core::impl_vec_ops_field;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::vec_ops::{BitReverseConfig, VecOps, VecOpsConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("m31", m31, ScalarField, ScalarCfg);
impl_vec_ops_field!("m31_q_extension", m31_q_extension, QuarticExtensionField, ExtensionCfg);
impl_vec_ops_field!(
    "m31_c_extension",
    m31_c_extension,
    ComplexExtensionField,
    ComplexExtensionCfg
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ComplexExtensionField, QuarticExtensionField, ScalarField};
    use icicle_core::impl_vec_add_tests;
    use icicle_core::ntt::FieldImpl;
    use icicle_core::vec_ops::{fold_scalars, tests::*, VecOpsConfig};
    use icicle_cuda_runtime::memory::HostSlice;

    impl_vec_add_tests!(ScalarField);
    mod complex_extension {
        use super::*;
        impl_vec_add_tests!(ComplexExtensionField);
    }
    mod extension {
        use super::*;
        impl_vec_add_tests!(QuarticExtensionField);
    }

    #[test]
    fn m31_fold_test() {
        // use icicle_core::traits::GenerateRandom;
        // let test_size = 1 << 14;

        // let mut a = ScalarCfg::generate_random(test_size);
        // let mut a_clone = a.clone();
        // let b = ScalarCfg::generate_random(test_size);
        // let ones = vec![ScalarField::one(); test_size];
        // let mut result = vec![ScalarField::zero(); test_size];
        // let mut result2 = vec![ScalarField::zero(); test_size];
        // let mut result3 = vec![ScalarField::zero(); test_size];
        // let a = HostSlice::from_mut_slice(&mut a);
        // let a_clone = HostSlice::from_mut_slice(&mut a_clone);
        // let b = HostSlice::from_slice(&b);
        // let ones = HostSlice::from_slice(&ones);
        // let result = HostSlice::from_mut_slice(&mut result);
        // let result2 = HostSlice::from_mut_slice(&mut result2);
        // let result3 = HostSlice::from_mut_slice(&mut result3);

        //--- on host ---
        let cfg = VecOpsConfig::default();

        // Example input: power-of-two values and appropriate folding factors
        let values = vec![
            ScalarField::from_u32(1),
            ScalarField::from_u32(2),
            ScalarField::from_u32(3),
            ScalarField::from_u32(4),
            ScalarField::from_u32(5),
            ScalarField::from_u32(6),
            ScalarField::from_u32(7),
            ScalarField::from_u32(8),
        ];
        let folding_factors = vec![
            QuarticExtensionField::from_u32(2),
            QuarticExtensionField::from_u32(3),
            QuarticExtensionField::from_u32(4),
        ];

        let a = HostSlice::from_slice(&values);
        let b = HostSlice::from_slice(&folding_factors);
        let mut result = vec![QuarticExtensionField::zero()];
        let res = HostSlice::from_mut_slice(&mut result);

        fold_scalars(a, b, res, &cfg).unwrap();

        let expected = QuarticExtensionField::from_u32(358);
        let result = result[0];
        assert_eq!(result, expected, "Result for simple folding is incorrect");

        // Set the desired length for folding_factors
        let folding_factors_length = 20; // Example length
        let values_length = 1 << folding_factors_length; // 2^folding_factors_length

        // Initialize the `values` vector

        use rayon::iter::IntoParallelIterator;
        use rayon::prelude::*;

        let values: Vec<_> = (1..=values_length)
            .into_par_iter()
            .map(|i| ScalarField::from_u32(i as u32))
            .collect();

        // Initialize the `folding_factors` vector
        // let mut folding_factors = Vec::with_capacity(folding_factors_length);
        // for i in 2..(2 + folding_factors_length) {
        //     folding_factors.push(QuarticExtensionField::from_u32(i as u32));
        // }

        let mut folding_factors: Vec<_> = (2..(2 + folding_factors_length))
            .into_par_iter()
            .map(|i| QuarticExtensionField::from_u32(i as u32))
            .collect();

        let a = HostSlice::from_slice(&values);
        let b = HostSlice::from_slice(&mut folding_factors);
        let mut result = vec![QuarticExtensionField::zero()];
        let res = HostSlice::from_mut_slice(&mut result);

        let time = std::time::Instant::now();
        fold_scalars(a, b, res, &cfg).unwrap();
        let elapsed = time.elapsed();
        println!("Elapsed time for 2^{}: {:?}", folding_factors_length, elapsed);

        let expected = QuarticExtensionField::from_u32(223550878);
        assert_eq!(result[0], expected, "Result for large folding is incorrect");
    }
}
