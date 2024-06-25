use crate::ntt::{NttAlgorithm, CUDA_NTT_ALGORITHM, CUDA_NTT_FAST_TWIDDLES_MODE};
use icicle_runtime::{device::Device, memory::HostSlice, runtime};
// use icicle_runtime::errors::eIcicleError;
// use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    ntt::{
        get_root_of_unity, initialize_domain, ntt, ntt_inplace, release_domain, NTTConfig, NTTDir, NTTDomain,
        NTTInitDomainConfig, NTT,
    },
    traits::{FieldImpl, GenerateRandom},
    // vec_ops::{transpose_matrix, VecOps},
};

pub fn init_domain<F: FieldImpl>(max_size: u64, fast_twiddles_mode: bool)
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    let config = NTTInitDomainConfig::default();
    // TODO Yuval : need a safer way to set configurations for backend
    config
        .ext
        .set_bool(CUDA_NTT_FAST_TWIDDLES_MODE, fast_twiddles_mode);
    let rou = get_root_of_unity::<F>(max_size);
    initialize_domain(rou, &config).unwrap();
}

pub fn rel_domain<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    release_domain::<F>().unwrap();
}

// TODO Yuval : fix or remove
// pub fn reverse_bit_order(n: u32, order: u32) -> u32 {
//     fn is_power_of_two(n: u32) -> bool {
//         n != 0 && n & (n - 1) == 0
//     }
//     assert!(is_power_of_two(order));
//     let mask = order - 1;
//     let binary = format!("{:0width$b}", n, width = (32 - mask.leading_zeros()) as usize);
//     let reversed = binary
//         .chars()
//         .rev()
//         .collect::<String>();
//     u32::from_str_radix(&reversed, 2).unwrap()
// }

// pub fn list_to_reverse_bit_order<T: Copy>(l: &[T]) -> Vec<T> {
//     l.iter()
//         .enumerate()
//         .map(|(i, _)| l[reverse_bit_order(i as u32, l.len() as u32) as usize])
//         .collect()
// }

// This test is comparing main and reference devices (typically CUDA and CPU) for NTT and inplace-INTT
pub fn check_ntt<F: FieldImpl>(main_device: &Device, ref_device: &Device)
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 17];
    for test_size in test_sizes {
        let scalars: Vec<F> = F::Config::generate_random(test_size);
        let mut ntt_result_main = vec![F::zero(); test_size];
        let mut ntt_result_ref = vec![F::zero(); test_size];

        let config: NTTConfig<F> = NTTConfig::default();
        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
            config
                .ext
                .set_int(CUDA_NTT_ALGORITHM, alg as i32);

            // compute NTT on main and reference devices and compare
            runtime::set_device(main_device).unwrap();
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result_main),
            )
            .unwrap();

            runtime::set_device(ref_device).unwrap();
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result_ref),
            )
            .unwrap();

            assert_eq!(ntt_result_main, ntt_result_ref);

            // compute INTT on main and reference devices, inplace, and compare

            runtime::set_device(main_device).unwrap();
            ntt_inplace(
                HostSlice::from_mut_slice(&mut ntt_result_main),
                NTTDir::kForward,
                &config,
            )
            .unwrap();

            runtime::set_device(ref_device).unwrap();
            ntt_inplace(
                HostSlice::from_mut_slice(&mut ntt_result_ref),
                NTTDir::kForward,
                &config,
            )
            .unwrap();

            assert_eq!(ntt_result_main, ntt_result_ref);
        }
    }
}
