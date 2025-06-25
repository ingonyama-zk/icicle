use crate::random_sampling::{random_sampling, RandomSampling};
use crate::traits::{Arithmetic, FieldImpl};
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{memory::HostSlice, test_utilities};
use rand::Rng;

pub fn check_random_sampling<F>()
where
    F: FieldImpl + Arithmetic,
    F::Config: RandomSampling<F>,
{
    let output_size = 1 << 10;
    let cfg = VecOpsConfig::default();

    let mut output_a = vec![F::zero(); output_size];
    let mut output_b = vec![F::zero(); output_size];

    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    test_utilities::test_set_main_device();
    random_sampling(
        output_size,
        false,
        &seed,
        &cfg,
        HostSlice::from_mut_slice(&mut output_a),
    )
    .expect("random sampling failed");
    test_utilities::test_set_ref_device();
    random_sampling(
        output_size,
        false,
        &seed,
        &cfg,
        HostSlice::from_mut_slice(&mut output_b),
    )
    .expect("random sampling failed");

    for i in 0..output_size {
        assert_eq!(output_a[i], output_b[i], "random sampling mismatch at index {}", i);
    }
}
