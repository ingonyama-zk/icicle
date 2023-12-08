use crate::curve::ScalarField;

use icicle_core::ntt::{Ordering, NTTConfig};
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};
use icicle_cuda_runtime::device_context::DeviceContext;

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
    use icicle_cuda_runtime::device_context::get_default_device_context;

    use crate::curve::generate_random_scalars;
    use crate::ntt::{ScalarField, ntt, get_default_ntt_config, initialize_domain};

    use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
    use ark_ff::FftField;
    use ark_bn254::Fr;

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
        let test_size = 1 << 2;

        let ctx = get_default_device_context();
        // two roughly analogous calls for icicle and arkworks. one difference is that icicle call creates
        // domain for all NTTs of size <= `test_size`. also for icicle domain is a hidden static object
        initialize_domain(ScalarField::from_ark(Fr::get_root_of_unity(test_size as u64).unwrap()), &ctx);
        let ark_domain = GeneralEvaluationDomain::<Fr>::new(test_size).unwrap();

        let scalars: Vec<ScalarField> = generate_random_scalars(test_size);
    
        let mut config = get_default_ntt_config();
        let mut ntt_result = vec![ScalarField::zero(); test_size];
        ntt(&scalars, false, &config, &mut ntt_result);

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

        // let mut ark_intt_result = ark_ntt_result;

        // ark_domain.ifft_in_place(&mut ark_intt_result);
        // assert_eq!(ark_intt_result, ark_scalars_batch);

        // // check that ntt output is different from input
        // assert_ne!(ntt_result, scalars_batch);

        // // do intt
        // let mut intt_result = ntt_result;

        // ntt_wip(&mut intt_result, true, false, Ordering::kNN, false, batches);

        // assert!(ark_intt_result == ark_scalars_batch);
        // assert!(intt_result == scalars_batch);

        // let mut ntt_intt_result = intt_result;
        // ntt_wip(&mut ntt_intt_result, false, false, Ordering::kNR, false, batches);
        // assert!(ntt_intt_result != scalars_batch);
        // ntt_wip(&mut ntt_intt_result, true, false, Ordering::kRN, false, batches);
        // assert!(ntt_intt_result == scalars_batch);

        // let mut ntt_intt_result = list_to_reverse_bit_order(&ntt_intt_result);
        // ntt_wip(&mut ntt_intt_result, false, false, Ordering::kRR, false, batches);
        // assert!(ntt_intt_result != scalars_batch);
        // ntt_wip(&mut ntt_intt_result, true, false, Ordering::kRN, false, batches);
        // assert!(ntt_intt_result == scalars_batch);

        // ////
        // let size = ntt_intt_result.len() / batches;

        // let mut config = get_ntt_config_with_input(&mut ntt_intt_result, size, batches);

        // ntt_internal(&mut config);

        // //host
        // let mut ntt_result = scalars_batch.clone();
        // ntt_wip(&mut ntt_result, false, false, Ordering::kNR, false, batches);

        // // let mut buff1 = DeviceBuffer::from_slice(&scalars_batch[..]).unwrap();
        // // let dev_ptr1 = buff1
        // //     .as_device_ptr()
        // //     .as_raw_mut();

        // // let buff_len = buff1.len();

        // // std::mem::forget(buff1);

        // // let buff_from_dev_ptr = unsafe { DeviceBuffer::from_raw_parts(DevicePointer::wrap(dev_ptr1), buff_len) };
        // // let mut from_device = vec![ScalarField::zero(); scalars_batch.len()];
        // // buff_from_dev_ptr
        // //     .copy_to(&mut from_device)
        // //     .unwrap();

        // // assert_eq!(from_device, scalars_batch);

        // // host - device - device - host
        // let mut ntt_intt_result = scalars_batch.clone();

        // let mut config = get_ntt_config_with_input(&mut ntt_intt_result, size, batches);

        // config.is_input_on_device = false;
        // config.is_output_on_device = true;
        // // config.is_preserving_twiddles = true; // TODO: same as in get_ntt_config
        // config.ordering = Ordering::kNR;

        // ntt_internal(&mut config); //twiddles are preserved after first call

        // // config.is_preserving_twiddles = true;        //TODO: same as in get_ntt_config
        // config.is_inverse = true;
        // config.is_input_on_device = false;
        // config.is_output_on_device = true;
        // config.ordering = Ordering::kNR;

        // ntt_internal(&mut config); //inv_twiddles are preserved after first call

        // let ntt_intt_result = &mut scalars_batch.clone()[..];
        // let raw_scalars_batch_copy = ntt_intt_result as *mut _ as *mut ScalarField;

        // let config_inout2: &mut [ScalarField] =
        //     unsafe { std::slice::from_raw_parts_mut(raw_scalars_batch_copy, config.size as usize) };
        // assert_eq!(config_inout2, scalars_batch);

        // config.is_preserving_twiddles = true; //TODO: same as in get_ntt_config

        // config.inout = raw_scalars_batch_copy;

        // config.is_inverse = false;
        // config.is_input_on_device = false;
        // config.is_output_on_device = true;
        // config.ordering = Ordering::kNR;

        // ntt_internal(&mut config);

        // config.is_inverse = true;
        // config.is_input_on_device = true;
        // config.is_output_on_device = false;
        // config.ordering = Ordering::kRN;

        // ntt_internal(&mut config);

        // let result_from_device: &mut [ScalarField] =
        //     unsafe { std::slice::from_raw_parts_mut(config.inout, scalars_batch.len()) };

        // assert_eq!(result_from_device, &scalars_batch);
    }

//     #[test]
//     fn test_batch_ntt() {
//         //NTT
//         let test_size = 1 << 11;
//         let batches = 2;

//         let full_test_size = test_size * batches;
//         let scalars_batch: Vec<ScalarField> = generate_random_scalars(full_test_size);

//         let mut scalar_vec_of_vec: Vec<Vec<ScalarField>> = Vec::new();

//         for i in 0..batches {
//             scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
//         }

//         let mut ntt_result = scalars_batch.clone();

//         // do batch ntt
//         ntt_wip(&mut ntt_result, false, false, Ordering::kNN, false, batches);

//         let mut ntt_result_vec_of_vec = Vec::new();

//         // do ntt for every chunk
//         for i in 0..batches {
//             ntt_result_vec_of_vec.push(scalar_vec_of_vec[i].clone());

//             ntt_wip(&mut ntt_result_vec_of_vec[i], false, false, Ordering::kNN, false, 1);
//         }

//         // check that the ntt of each vec of scalars is equal to the ntt of the specific batch
//         for i in 0..batches {
//             assert_eq!(ntt_result_vec_of_vec[i], ntt_result[i * test_size..(i + 1) * test_size]);
//         }

//         // check that ntt output is different from input
//         assert_ne!(ntt_result, scalars_batch);

//         let mut intt_result = ntt_result.clone();

//         // do batch intt
//         // intt_batch(&mut intt_result, test_size, 0);
//         ntt_wip(&mut intt_result, true, false, Ordering::kNN, false, batches);

//         let mut intt_result_vec_of_vec = Vec::new();

//         // do intt for every chunk
//         for i in 0..batches {
//             intt_result_vec_of_vec.push(ntt_result_vec_of_vec[i].clone());
//             // intt(&mut intt_result_vec_of_vec[i], 0);
//             ntt_wip(&mut intt_result_vec_of_vec[i], true, false, Ordering::kNN, false, 1);
//         }

//         // check that the intt of each vec of scalars is equal to the intt of the specific batch
//         for i in 0..batches {
//             assert_eq!(
//                 intt_result_vec_of_vec[i],
//                 intt_result[i * test_size..(i + 1) * test_size]
//             );
//         }

//         assert_eq!(intt_result, scalars_batch);
//     }
}
