use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use icicle_cuda_runtime::device_context::get_default_device_context;

use crate::{
    field::FieldConfig,
    ntt::Ordering,
    traits::{ArkConvertible, FieldImpl, GenerateRandom},
};

use super::NTT;

pub fn reverse_bit_order(n: u32, order: u32) -> u32 {
    fn is_power_of_two(n: u32) -> bool {
        n != 0 && n & (n - 1) == 0
    }
    assert!(is_power_of_two(order));
    let mask = order - 1;
    let binary = format!("{:0width$b}", n, width = (32 - mask.leading_zeros()) as usize);
    let reversed = binary
        .chars()
        .rev()
        .collect::<String>();
    u32::from_str_radix(&reversed, 2).unwrap()
}

pub fn list_to_reverse_bit_order<T: Copy>(l: &[T]) -> Vec<T> {
    l.iter()
        .enumerate()
        .map(|(i, _)| l[reverse_bit_order(i as u32, l.len() as u32) as usize])
        .collect()
}

pub fn check_ntt<F: FieldImpl + ArkConvertible, Fc: FieldConfig + NTT<F> + GenerateRandom<F>>()
where
    F::ArkEquivalent: FftField,
{
    let test_size = 1 << 16;
    let ctx = get_default_device_context();
    // two roughly analogous calls for icicle and arkworks. one difference is that icicle call creates
    // domain for all NTTs of size <= `test_size`. also for icicle domain is a hidden static object
    Fc::initialize_domain(
        F::from_ark(F::ArkEquivalent::get_root_of_unity(test_size as u64).unwrap()),
        &ctx,
    )
    .unwrap();
    let ark_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

    let scalars: Vec<F> = Fc::generate_random(test_size);

    let config = Fc::get_default_ntt_config();
    let mut ntt_result = vec![F::zero(); test_size];
    Fc::ntt(&scalars, false, &config, &mut ntt_result).unwrap();
    assert_ne!(ntt_result, scalars);

    let ark_scalars = scalars
        .iter()
        .map(|v| v.to_ark())
        .collect::<Vec<F::ArkEquivalent>>();
    let mut ark_ntt_result = ark_scalars.clone();
    ark_domain.fft_in_place(&mut ark_ntt_result);
    assert_ne!(ark_ntt_result, ark_scalars);

    let ntt_result_as_ark = ntt_result
        .iter()
        .map(|p| p.to_ark())
        .collect::<Vec<F::ArkEquivalent>>();
    assert_eq!(ark_ntt_result, ntt_result_as_ark);

    let mut intt_result = vec![F::zero(); test_size];
    Fc::ntt(&ntt_result, true, &config, &mut intt_result).unwrap();

    assert_eq!(intt_result, scalars);
    // check that ntt_result wasn't mutated by the latest `ntt` call
    assert_eq!(ntt_result_as_ark[1], ntt_result[1].to_ark());
}

pub fn check_ntt_coset_from_subgroup<F: FieldImpl + ArkConvertible, Fc: FieldConfig + NTT<F> + GenerateRandom<F>>()
where
    F::ArkEquivalent: FftField,
{
    let test_size = 1 << 16;
    let small_size = test_size >> 1;
    let test_size_rou = F::ArkEquivalent::get_root_of_unity(test_size as u64).unwrap();
    let ctx = get_default_device_context();
    // two roughly analogous calls for icicle and arkworks. one difference is that icicle call creates
    // domain for all NTTs of size <= `test_size`. also for icicle domain is a hidden static object
    Fc::initialize_domain(F::from_ark(test_size_rou), &ctx).unwrap();
    let ark_small_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(small_size)
        .unwrap()
        .get_coset(test_size_rou)
        .unwrap();
    let ark_large_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

    let mut scalars: Vec<F> = Fc::generate_random(small_size);

    let mut config = Fc::get_default_ntt_config();
    config.ordering = Ordering::kNR;
    let mut ntt_result = vec![F::zero(); test_size];
    Fc::ntt(&scalars, false, &config, &mut ntt_result[..small_size]).unwrap();
    assert_ne!(ntt_result[..small_size], scalars);
    config.coset_gen = F::from_ark(test_size_rou);
    Fc::ntt(&scalars, false, &config, &mut ntt_result[small_size..]).unwrap();
    let mut ntt_large_result = vec![F::zero(); test_size];
    // back to non-coset NTT
    config.coset_gen = F::one();
    scalars.resize(test_size, F::zero());
    Fc::ntt(&scalars, false, &config, &mut ntt_large_result).unwrap();
    assert_eq!(ntt_result, ntt_large_result);

    let mut ark_scalars = scalars
        .iter()
        .map(|v| v.to_ark())
        .collect::<Vec<F::ArkEquivalent>>();
    let mut ark_large_scalars = ark_scalars.clone();
    ark_small_domain.fft_in_place(&mut ark_scalars);
    let ntt_result_as_ark = ntt_result
        .iter()
        .map(|p| p.to_ark())
        .collect::<Vec<F::ArkEquivalent>>();
    assert_eq!(
        ark_scalars[..small_size],
        list_to_reverse_bit_order(&ntt_result_as_ark[small_size..])
    );
    ark_large_domain.fft_in_place(&mut ark_large_scalars);
    assert_eq!(ark_large_scalars, list_to_reverse_bit_order(&ntt_result_as_ark));

    config.coset_gen = F::from_ark(test_size_rou);
    config.ordering = Ordering::kRN;
    let mut intt_result = vec![F::zero(); small_size];
    Fc::ntt(&ntt_result[small_size..], true, &config, &mut intt_result).unwrap();
    assert_eq!(intt_result, scalars[..small_size]);

    ark_small_domain.ifft_in_place(&mut ark_scalars);
    let intt_result_as_ark = intt_result
        .iter()
        .map(|p| p.to_ark())
        .collect::<Vec<F::ArkEquivalent>>();
    assert_eq!(ark_scalars[..small_size], intt_result_as_ark);
}
