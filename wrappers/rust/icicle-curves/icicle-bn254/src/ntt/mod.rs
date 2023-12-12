use crate::curve::ScalarField;

use icicle_core::ntt::NTTConfig;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

extern "C" {
    #[link_name = "bn254NTTCuda"]
    fn ntt_cuda<'a>(
        input: *const ScalarField,
        size: usize,
        is_inverse: bool,
        config: &NTTConfig<'a, ScalarField>,
        output: *mut ScalarField,
    ) -> CudaError;

    #[link_name = "bn254DefaultNTTConfig"]
    fn default_ntt_config() -> NTTConfig<'static, ScalarField>;

    #[link_name = "bn254InitializeDomain"]
    fn initialize_ntt_domain(primitive_root: ScalarField, ctx: &DeviceContext) -> CudaError;
}

pub fn get_default_ntt_config() -> NTTConfig<'static, ScalarField> {
    unsafe { default_ntt_config() }
}

pub fn initialize_domain(primitive_root: ScalarField, ctx: &DeviceContext) -> CudaResult<()> {
    unsafe { initialize_ntt_domain(primitive_root, ctx).wrap() }
}

pub fn ntt(
    input: &[ScalarField],
    is_inverse: bool,
    cfg: &NTTConfig<ScalarField>,
    output: &mut [ScalarField],
) -> CudaResult<()> {
    if input.len() != output.len() {
        return Err(CudaError::cudaErrorInvalidValue);
    }

    unsafe {
        ntt_cuda(
            input as *const _ as *const ScalarField,
            input.len(),
            is_inverse,
            cfg,
            output as *mut _ as *mut ScalarField,
        )
        .wrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::traits::ArkConvertible;
    use icicle_core::ntt::Ordering;
    use icicle_cuda_runtime::device_context::get_default_device_context;

    use crate::curve::generate_random_scalars;
    use crate::ntt::{get_default_ntt_config, initialize_domain, ntt, ScalarField};

    use ark_bn254::Fr;
    use ark_ff::FftField;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    fn reverse_bit_order(n: u32, order: u32) -> u32 {
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

    fn list_to_reverse_bit_order<T: Copy>(l: &[T]) -> Vec<T> {
        l.iter()
            .enumerate()
            .map(|(i, _)| l[reverse_bit_order(i as u32, l.len() as u32) as usize])
            .collect()
    }

    #[test]
    fn test_ntt() {
        let test_size = 1 << 16;
        let ctx = get_default_device_context();
        // two roughly analogous calls for icicle and arkworks. one difference is that icicle call creates
        // domain for all NTTs of size <= `test_size`. also for icicle domain is a hidden static object
        initialize_domain(
            ScalarField::from_ark(Fr::get_root_of_unity(test_size as u64).unwrap()),
            &ctx,
        ).unwrap();
        let ark_domain = GeneralEvaluationDomain::<Fr>::new(test_size).unwrap();

        let scalars: Vec<ScalarField> = generate_random_scalars(test_size);

        let config = get_default_ntt_config();
        let mut ntt_result = vec![ScalarField::zero(); test_size];
        ntt(&scalars, false, &config, &mut ntt_result).unwrap();
        assert_ne!(ntt_result, scalars);

        let ark_scalars = scalars
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<Fr>>();
        let mut ark_ntt_result = ark_scalars.clone();
        ark_domain.fft_in_place(&mut ark_ntt_result);
        assert_ne!(ark_ntt_result, ark_scalars);

        let ntt_result_as_ark = ntt_result
            .iter()
            .map(|p| p.to_ark())
            .collect::<Vec<Fr>>();
        assert_eq!(ark_ntt_result, ntt_result_as_ark);

        let mut intt_result = vec![ScalarField::zero(); test_size];
        ntt(&ntt_result, true, &config, &mut intt_result).unwrap();

        assert_eq!(intt_result, scalars);
        // check that ntt_result wasn't mutated by the latest `ntt` call
        assert_eq!(ntt_result_as_ark[1], ntt_result[1].to_ark());
    }

    #[test]
    fn test_ntt_coset_from_subgroup() {
        let test_size = 1 << 16;
        let small_size = test_size >> 1;
        let test_size_rou = Fr::get_root_of_unity(test_size as u64).unwrap();
        let ctx = get_default_device_context();
        // two roughly analogous calls for icicle and arkworks. one difference is that icicle call creates
        // domain for all NTTs of size <= `test_size`. also for icicle domain is a hidden static object
        initialize_domain(ScalarField::from_ark(test_size_rou), &ctx).unwrap();
        let ark_small_domain = GeneralEvaluationDomain::<Fr>::new(small_size).unwrap().get_coset(test_size_rou).unwrap();
        let ark_large_domain = GeneralEvaluationDomain::<Fr>::new(test_size).unwrap();

        let mut scalars: Vec<ScalarField> = generate_random_scalars(small_size);

        let mut config = get_default_ntt_config();
        config.ordering = Ordering::kNR;
        let mut ntt_result = vec![ScalarField::zero(); test_size];
        ntt(&scalars, false, &config, &mut ntt_result[..small_size]).unwrap();
        assert_ne!(ntt_result[..small_size], scalars);
        config.coset_gen = ScalarField::from_ark(test_size_rou);
        ntt(&scalars, false, &config, &mut ntt_result[small_size..]).unwrap();
        let mut ntt_large_result = vec![ScalarField::zero(); test_size];
        // back to non-coset NTT
        config.coset_gen = ScalarField::one();
        scalars.resize(test_size, ScalarField::zero());
        ntt(&scalars, false, &config, &mut ntt_large_result).unwrap();
        assert_eq!(ntt_result, ntt_large_result);

        let mut ark_scalars = scalars
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<Fr>>();
        let mut ark_large_scalars = ark_scalars.clone();
        ark_small_domain.fft_in_place(&mut ark_scalars);
        let ntt_result_as_ark = ntt_result
            .iter()
            .map(|p| p.to_ark())
            .collect::<Vec<Fr>>();
        assert_eq!(ark_scalars[..small_size], list_to_reverse_bit_order(&ntt_result_as_ark[small_size..]));
        ark_large_domain.fft_in_place(&mut ark_large_scalars);
        assert_eq!(ark_large_scalars, list_to_reverse_bit_order(&ntt_result_as_ark));

        config.coset_gen = ScalarField::from_ark(test_size_rou);
        config.ordering = Ordering::kRN;
        let mut intt_result = vec![ScalarField::zero(); small_size];
        ntt(&ntt_result[small_size..], true, &config, &mut intt_result).unwrap();
        assert_eq!(intt_result, scalars[..small_size]);

        ark_small_domain.ifft_in_place(&mut ark_scalars);
        let intt_result_as_ark = intt_result
            .iter()
            .map(|p| p.to_ark())
            .collect::<Vec<Fr>>();
        assert_eq!(ark_scalars[..small_size], intt_result_as_ark);
    }
}
