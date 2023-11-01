mod config;
pub mod domain;

use std::any::TypeId;

use crate::{cuda::*, curve::*};

use self::config::*;

extern "C" {
    #[link_name = "NTTDefaultContextCuda"]
    fn ntt_cuda(config: *mut NTTConfig) -> cudaError_t;
}

pub(crate) fn ntt_wip(
    inout: &mut [ScalarField],
    is_inverse: bool,
    is_input_on_device: bool,
    ordering: Ordering,
    is_output_on_device: bool,
    batch_size: usize,
) {
    let mut batch_size = batch_size;
    if batch_size == 0 {
        batch_size = 1;
    }

    let size = inout.len() / batch_size;

    let mut config = get_ntt_default_config::<ScalarField, ScalarField>(size);

    config.inout = inout as *mut _ as *mut ScalarField;
    config.is_inverse = is_inverse;
    config.is_input_on_device = is_input_on_device;
    config.is_output_on_device = is_output_on_device;
    config.ordering = ordering;
    config.batch_size = batch_size as i32;

    ntt_internal(&mut config);
}

pub(self) fn ntt_internal<TConfig: 'static>(config: *mut TConfig) -> u32 {
    let result_code;
    let typeid = TypeId::of::<TConfig>();
    if typeid == TypeId::of::<NTTConfig>() {
        result_code = unsafe { ntt_cuda(config as _) };
    } else {
        result_code = 0; //TODO: unsafe { ecntt_cuda(config as _) };
    }

    if result_code != 0 {
        println!("_result_code = {}", result_code);
    }

    return result_code;
}

pub(self) fn ecntt_internal(config: *mut ECNTTConfig) -> u32 {
    let result_code = 0; //TODO: unsafe { ecntt_cuda(config) };
    if result_code != 0 {
        println!("_result_code = {}", result_code);
    }

    return result_code;
}

#[cfg(test)]
pub(crate) mod tests {

    // use ark_bn254::{Fr, G1Projective};
    use ark_bls12_381::{Fr, G1Projective};
    use ark_ff::PrimeField;
    use ark_poly::EvaluationDomain;
    use ark_poly::GeneralEvaluationDomain;
    use ark_std::UniformRand;
    use rand::RngCore;
    use rustacuda::prelude::CopyDestination;
    use rustacuda::prelude::DeviceBuffer;
    use rustacuda_core::DevicePointer;

    use crate::ntt::domain::NTTDomain;
    use crate::{curve::*, ntt::*, utils::get_rng};

    pub fn generate_random_points(count: usize, mut rng: Box<dyn RngCore>) -> Vec<PointAffineNoInfinity> {
        (0..count)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)).to_xy_strip_z())
            .collect()
    }

    #[allow(dead_code)]
    pub fn generate_random_points_proj(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Point> {
        (0..count)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)))
            .collect()
    }

    pub fn generate_random_scalars(count: usize, mut rng: Box<dyn RngCore>) -> Vec<ScalarField> {
        (0..count)
            .map(|_| ScalarField::from_ark(Fr::rand(&mut rng).into_repr()))
            .collect()
    }

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

    #[test]
    fn test_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 11;
        let batches = 1;

        let full_test_size = test_size * batches;
        let scalars_batch: Vec<ScalarField> = generate_random_scalars(full_test_size, get_rng(seed));

        // let scalars_batch: Vec<ScalarField> = (0..full_test_size)
        //     .into_iter()
        //     .map(|x| {
        //         // if x % 1 == 0 {
        //         if x % 2 == 0 {
        //             ScalarField::one()
        //         } else {
        //             ScalarField::zero()
        //         }
        //     })
        //     .collect();

        let mut ntt_result = scalars_batch.clone();

        let ark_domain = GeneralEvaluationDomain::<Fr>::new(test_size).unwrap();
        let mut domain = NTTDomain::new_for_default_context(test_size);

        let ark_scalars_batch = scalars_batch
            .clone()
            .iter()
            .map(|v| Fr::new(v.to_ark()))
            .collect::<Vec<Fr>>();
        let mut ark_ntt_result = ark_scalars_batch.clone();

        ark_domain.fft_in_place(&mut ark_ntt_result);

        assert_ne!(ark_ntt_result, ark_scalars_batch);

        // do ntt
        // ntt_wip(&mut ntt_result, false, false, Ordering::kNN, false, batches);
        domain.ntt(&mut ntt_result); //single ntt
        let ntt_result_as_ark = ntt_result
            .iter()
            .map(|p| Fr::new(p.to_ark()))
            .collect::<Vec<Fr>>();

        assert_ne!(ntt_result, scalars_batch);
        assert_eq!(ark_ntt_result, ntt_result_as_ark);

        let mut ark_intt_result = ark_ntt_result;

        ark_domain.ifft_in_place(&mut ark_intt_result);
        assert_eq!(ark_intt_result, ark_scalars_batch);

        // check that ntt output is different from input
        assert_ne!(ntt_result, scalars_batch);

        // do intt
        let mut intt_result = ntt_result;

        ntt_wip(&mut intt_result, true, false, Ordering::kNN, false, batches);

        assert!(ark_intt_result == ark_scalars_batch);
        assert!(intt_result == scalars_batch);

        let mut ntt_intt_result = intt_result;
        ntt_wip(&mut ntt_intt_result, false, false, Ordering::kNR, false, batches);
        assert!(ntt_intt_result != scalars_batch);
        ntt_wip(&mut ntt_intt_result, true, false, Ordering::kRN, false, batches);
        assert!(ntt_intt_result == scalars_batch);

        let mut ntt_intt_result = list_to_reverse_bit_order(&ntt_intt_result);
        ntt_wip(&mut ntt_intt_result, false, false, Ordering::kRR, false, batches);
        assert!(ntt_intt_result != scalars_batch);
        ntt_wip(&mut ntt_intt_result, true, false, Ordering::kRN, false, batches);
        assert!(ntt_intt_result == scalars_batch);

        ////
        let size = ntt_intt_result.len() / batches;

        let mut config = get_ntt_config_with_input(&mut ntt_intt_result, size, batches);

        ntt_internal(&mut config);

        //host
        let mut ntt_result = scalars_batch.clone();
        ntt_wip(&mut ntt_result, false, false, Ordering::kNR, false, batches);

        let mut buff1 = DeviceBuffer::from_slice(&scalars_batch[..]).unwrap();
        let dev_ptr1 = buff1
            .as_device_ptr()
            .as_raw_mut();

        let buff_len = buff1.len();

        std::mem::forget(buff1);

        let buff_from_dev_ptr = unsafe { DeviceBuffer::from_raw_parts(DevicePointer::wrap(dev_ptr1), buff_len) };
        let mut from_device = vec![ScalarField::zero(); scalars_batch.len()];
        buff_from_dev_ptr
            .copy_to(&mut from_device)
            .unwrap();

        assert_eq!(from_device, scalars_batch);

        // host - device - device - host
        let mut ntt_intt_result = scalars_batch.clone();

        let mut config = get_ntt_config_with_input(&mut ntt_intt_result, size, batches);

        config.is_input_on_device = false;
        config.is_output_on_device = true;
        // config.is_preserving_twiddles = true; // TODO: same as in get_ntt_config
        config.ordering = Ordering::kNR;

        ntt_internal(&mut config); //twiddles are preserved after first call

        // config.is_preserving_twiddles = true;        //TODO: same as in get_ntt_config
        config.is_inverse = true;
        config.is_input_on_device = false;
        config.is_output_on_device = true;
        config.ordering = Ordering::kNR;

        ntt_internal(&mut config); //inv_twiddles are preserved after first call

        println!("\ntwiddles should be initialized here\n");

        let ntt_intt_result = &mut scalars_batch.clone()[..];
        let raw_scalars_batch_copy = ntt_intt_result as *mut _ as *mut ScalarField;

        let config_inout2: &mut [ScalarField] =
            unsafe { std::slice::from_raw_parts_mut(raw_scalars_batch_copy, config.size as usize) };
        assert_eq!(config_inout2, scalars_batch);

        config.is_preserving_twiddles = true; //TODO: same as in get_ntt_config

        config.inout = raw_scalars_batch_copy;

        config.is_inverse = false;
        config.is_input_on_device = false;
        config.is_output_on_device = true;
        config.ordering = Ordering::kNR;

        ntt_internal(&mut config);

        config.is_inverse = true;
        config.is_input_on_device = true;
        config.is_output_on_device = false;
        config.ordering = Ordering::kRN;

        ntt_internal(&mut config);

        let result_from_device: &mut [ScalarField] =
            unsafe { std::slice::from_raw_parts_mut(config.inout, scalars_batch.len()) };

        assert_eq!(result_from_device, &scalars_batch);
    }

    #[test]
    fn test_batch_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 11;
        let batches = 2;

        let full_test_size = test_size * batches;
        let scalars_batch: Vec<ScalarField> = generate_random_scalars(full_test_size, get_rng(seed));

        let mut scalar_vec_of_vec: Vec<Vec<ScalarField>> = Vec::new();

        for i in 0..batches {
            scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result = scalars_batch.clone();

        // do batch ntt
        ntt_wip(&mut ntt_result, false, false, Ordering::kNN, false, batches);

        let mut ntt_result_vec_of_vec = Vec::new();

        // do ntt for every chunk
        for i in 0..batches {
            ntt_result_vec_of_vec.push(scalar_vec_of_vec[i].clone());

            ntt_wip(&mut ntt_result_vec_of_vec[i], false, false, Ordering::kNN, false, 1);
        }

        // check that the ntt of each vec of scalars is equal to the ntt of the specific batch
        for i in 0..batches {
            assert_eq!(ntt_result_vec_of_vec[i], ntt_result[i * test_size..(i + 1) * test_size]);
        }

        // check that ntt output is different from input
        assert_ne!(ntt_result, scalars_batch);

        let mut intt_result = ntt_result.clone();

        // do batch intt
        // intt_batch(&mut intt_result, test_size, 0);
        ntt_wip(&mut intt_result, true, false, Ordering::kNN, false, batches);

        let mut intt_result_vec_of_vec = Vec::new();

        // do intt for every chunk
        for i in 0..batches {
            intt_result_vec_of_vec.push(ntt_result_vec_of_vec[i].clone());
            // intt(&mut intt_result_vec_of_vec[i], 0);
            ntt_wip(&mut intt_result_vec_of_vec[i], true, false, Ordering::kNN, false, 1);
        }

        // check that the intt of each vec of scalars is equal to the intt of the specific batch
        for i in 0..batches {
            assert_eq!(
                intt_result_vec_of_vec[i],
                intt_result[i * test_size..(i + 1) * test_size]
            );
        }

        assert_eq!(intt_result, scalars_batch);
    }
}
