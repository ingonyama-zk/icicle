use icicle_runtime::{memory::HostSlice, test_utilities};

use crate::{
    ecntt::*,
    ntt::{NTTConfig, NTTDir, Ordering},
};

pub fn check_ecntt<P: Projective>()
where
    P::ScalarField: ECNTT<P, P::ScalarField>,
{
    let test_sizes = [1 << 4, 1 << 9, 1 << 11];
    for test_size in test_sizes {
        for dir in [NTTDir::kForward, NTTDir::kInverse] {
            let config: NTTConfig<P::ScalarField> = NTTConfig::default();

            let points = P::generate_random(test_size);
            let mut ecntt_result = vec![P::zero(); test_size];
            let mut ecntt_result_ref = vec![P::zero(); test_size];

            // compare main to ref device
            test_utilities::test_set_main_device();
            ecntt(
                HostSlice::from_slice(&points),
                dir,
                &config,
                HostSlice::from_mut_slice(&mut ecntt_result),
            )
            .unwrap();

            test_utilities::test_set_ref_device();
            ecntt(
                HostSlice::from_slice(&points),
                dir,
                &config,
                HostSlice::from_mut_slice(&mut ecntt_result_ref),
            )
            .unwrap();

            assert_eq!(ecntt_result, ecntt_result_ref);

            // inverse ECNTT and assert that we get the points back
            let inv_dir = match dir {
                NTTDir::kForward => NTTDir::kInverse,
                NTTDir::kInverse => NTTDir::kForward,
            };

            test_utilities::test_set_main_device();
            ecntt_inplace(HostSlice::from_mut_slice(&mut ecntt_result), inv_dir, &config).unwrap();
            assert_eq!(ecntt_result, points);
        }
    }
}

pub fn check_ecntt_batch<P: Projective>()
where
    P::ScalarField: ECNTT<P, P::ScalarField>,
{
    test_utilities::test_set_main_device();

    let test_sizes = [1 << 4, 1 << 9];
    let batch_sizes = [1, 1 << 4, 21];
    for test_size in test_sizes {
        let mut config: NTTConfig<P::ScalarField> = NTTConfig::default();
        for batch_size in batch_sizes {
            let slice = &P::generate_random(test_size * batch_size);
            let points = HostSlice::from_slice(slice);

            for is_inverse in [NTTDir::kInverse, NTTDir::kForward] {
                config.ordering = Ordering::kNN;
                let mut slice = vec![P::zero(); batch_size * test_size];
                let batch_ntt_result = HostSlice::from_mut_slice(&mut slice);
                config.batch_size = batch_size as i32;
                ecntt(points, is_inverse, &config, batch_ntt_result).unwrap();

                config.batch_size = 1;
                let mut slice = vec![P::zero(); test_size];
                let one_ntt_result = HostSlice::from_mut_slice(&mut slice);
                for i in 0..batch_size {
                    ecntt(
                        HostSlice::from_slice(&points[i * test_size..(i + 1) * test_size].as_slice()),
                        is_inverse,
                        &config,
                        one_ntt_result,
                    )
                    .unwrap();
                    assert_eq!(
                        batch_ntt_result[i * test_size..(i + 1) * test_size].as_slice(),
                        one_ntt_result.as_slice()
                    );
                }
            }
        }
    }
}
