use std::ffi::{c_int, c_uint};

// use rustacuda::prelude::DeviceBuffer;
// use rustacuda_core::DevicePointer;

use crate::{curve::*, msm::DeviceContext};

/**
 * @enum Ordering
 * How to order inputs and outputs of the NTT:
 * - kNN: inputs and outputs are natural-order (example of natural ordering: \f$ \{a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7\} \f$).
 * - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: \f$ \{a_0, a_4, a_2, a_6, a_1, a_5, a_3, a_7\} \f$).
 * - kRN: inputs are bit-reversed-order and outputs are natural-order.
 * - kRR: inputs and outputs are bit-reversed-order.
 */
#[allow(non_camel_case_types)]
#[repr(C)]
pub enum Ordering {
    kNN,
    kNR,
    kRN,
    kRR,
}

/**
 * @enum Decimation
 * Decimation of the NTT algorithm:
 * - kDIT: decimation in time.
 * - kDIF: decimation in frequency.
 */
#[allow(non_camel_case_types)]
#[repr(C)]
enum Decimation {
    kDIT,
    kDIF,
}

/**
 * @enum Butterfly
 * [Butterfly](https://en.wikipedia.org/wiki/Butterfly_diagram) used in the NTT algorithm (i.e. what happens to each pair of inputs on every iteration):
 * - kCooleyTukey: Cooley-Tukey butterfly.
 * - kGentlemanSande: Gentleman-Sande butterfly.
 */
#[allow(non_camel_case_types)]
#[repr(C)]
enum Butterfly {
    kCooleyTukey,
    kGentlemanSande,
}

/**
 * @struct NTTConfig
 * Struct that encodes NTT parameters to be passed into the [ntt](@ref ntt) function.
 */
#[repr(C)]
pub(crate) struct NTTConfigCuda<E, S> {
    inout: *mut E,
    /**< Input that's mutated in-place by this function. Length of this array needs to be \f$ size \cdot config.batch_size \f$.
     *   Note that if inputs are in Montgomery form, the outputs will be as well and vice-verse: non-Montgomery inputs produce non-Montgomety outputs.*/
    is_input_on_device: bool,
    /**< True if inputs/outputs are on device and false if they're on host. Default value: false. */
    is_inverse: bool,
    /**< True if true . Default value: false. */
    ordering: Ordering,
    /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`. */
    decimation: Decimation,
    /**< Decimation of the algorithm, see [Decimation](@ref Decimation). Default value: `Decimation::kDIT`.
     *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
     *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
     *   `Decimation::kDIT` and if ordering is `Ordering::kNR` — to `Decimation::kDIF`. */
    butterfly: Butterfly,
    /**< Butterfly used by the NTT. See [Butterfly](@ref Butterfly). Default value: `Butterfly::kCooleyTukey`.
     *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
     *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
     *   `Butterfly::kCooleyTukey` and if ordering is `Ordering::kNR` — to `Butterfly::kGentlemanSande`. */
    is_coset: bool,
    /**< If false, NTT is computed on a subfield given by [twiddles](@ref twiddles). If true, NTT is computed
     *   on a coset of [twiddles](@ref twiddles) given by [the coset generator](@ref coset_gen), so:
     *   \f$ \{coset\_gen\cdot\omega^0, coset\_gen\cdot\omega^1, \dots, coset\_gen\cdot\omega^{n-1}\} \f$. Default value: false. */
    coset_gen: *const S,
    /**< The field element that generates a coset if [is_coset](@ref is_coset) is true.
     *   Otherwise should be set to `nullptr`. Default value: `nullptr`. */
    twiddles: *const S,
    /**< "Twiddle factors", (or "domain", or "roots of unity") on which the NTT is evaluated.
     *   This pointer is expected to live on device. The order is as follows:
     *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle factors
     *   are generated online using the default generator (TODO: link to twiddle gen here) and function
     *   [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    size: c_int,
    /**< NTT size \f$ n \f$. If a batch of NTTs (which all need to have the same size) is computed, this is the size of 1 NTT. */
    batch_size: c_int,
    /**< The number of NTTs to compute. Default value: 1. */
    is_preserving_tweedles: bool,
    /**< If true, twiddle factors are preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    is_output_on_device: bool,
    /**< If true, output is preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    ctx: DeviceContext, /*< Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext). */
}

#[allow(non_camel_case_types)]
pub type cudaError_t = c_uint;

pub(crate) type ECNTTConfig = NTTConfigCuda<Point, ScalarField>;
pub(crate) type NTTConfig = NTTConfigCuda<ScalarField, ScalarField>;

extern "C" {
    #[link_name = "NTTDefaultContextCuda"]
    fn ntt_cuda(config: *mut NTTConfig) -> cudaError_t;
}

pub fn ntt(
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

    let size = (inout.len() / batch_size) as i32;

    // println!("size: {}", size);

    let mut config = NTTConfig {
        inout: inout as *mut _ as *mut ScalarField,
        is_input_on_device,
        is_inverse,
        ordering,
        decimation: Decimation::kDIF,
        butterfly: Butterfly::kCooleyTukey,
        is_coset: false,
        coset_gen: &[ScalarField::zero()] as _, //TODO: ?
        twiddles: 0 as *const ScalarField,      //TODO: ?,
        size,
        batch_size: batch_size as i32,
        is_preserving_tweedles: true,
        is_output_on_device,
        ctx: DeviceContext {
            device_id: 0,
            stream: 0,
            mempool: 0,
        },
    };
    // println!("hehe");
    let result_code = unsafe { ntt_cuda(&mut config) };
    // println!("_result_code = {}", result_code);
}

// pub fn ntt_device(inout: &mut DeviceBuffer<ScalarField>, twiddles: &mut DeviceBuffer<ScalarField>, is_inverse: bool) {
//     let count = twiddles.len();
//     if count != inout.len() {
//         todo!("variable length")
//     }

//     let _result_code = unsafe {
//         ntt_device_cuda(
//             inout.as_device_ptr(),
//             twiddles.as_device_ptr(),
//             inout.len(),
//             is_inverse, // get_default_msm_config(points.len()),
//         )
//     };
// }

// pub fn ecntt(inout: &mut [PointAffineNoInfinity], twiddles: &[ScalarField], is_inverse: bool) {
//     let count = inout.len();
//     if count != twiddles.len() {
//         todo!("variable length")
//     }

//     let _result_code = unsafe {
//         ecntt_cuda(
//             inout as *mut _ as *mut PointAffineNoInfinity,
//             inout as *const _ as *const ScalarField,
//             inout.len(),
//             is_inverse, // get_default_msm_config(points.len()),
//         )
//     };
// }

#[cfg(test)]
pub(crate) mod tests {
    use std::vec;

    // use ark_bn254::{Fr, G1Projective};
    use ark_bls12_381::{Fr, G1Projective};
    use ark_ff::PrimeField;
    use ark_poly::EvaluationDomain;
    use ark_poly::GeneralEvaluationDomain;
    use ark_std::UniformRand;
    use rand::RngCore;
    use rustacuda::prelude::CopyDestination;

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

    #[test]
    fn test_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 8;
        let batches = 1;

        let full_test_size = test_size * batches;
        let scalars_batch: Vec<ScalarField> = generate_random_scalars(full_test_size, get_rng(seed));

        // let scalars_batch: Vec<ScalarField> = (0..full_test_size)
        //     .into_iter()
        //     .map(|x| {
        //         if x % 2 == 0 {
        //             ScalarField::one()
        //         } else {
        //             ScalarField::zero()
        //         }
        //     })
        //     .collect();

        let mut scalar_vec_of_vec: Vec<Vec<ScalarField>> = Vec::new();

        for i in 0..batches {
            scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result = scalars_batch.clone();

        let ark_domain = GeneralEvaluationDomain::<Fr>::new(test_size).unwrap();

        let ark_scalars_batch = scalars_batch
            .clone()
            .iter()
            .map(|v| Fr::new(v.to_ark()))
            .collect::<Vec<Fr>>();
        let mut ark_ntt_result = ark_scalars_batch.clone();

        ark_domain.fft_in_place(&mut ark_ntt_result);

        assert_ne!(ark_ntt_result, ark_scalars_batch);

        // do batch ntt
        ntt(&mut ntt_result, false, false, Ordering::kNN, false, batches);

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

        ntt(&mut intt_result, true, false, Ordering::kNN, false, batches);

        assert!(ark_intt_result == ark_scalars_batch);
        assert!(intt_result == scalars_batch);
    }

    #[test]
    fn test_ntt_batch() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 8;
        let batches = 2;

        let full_test_size = test_size * batches;
        let scalars_batch: Vec<ScalarField> = generate_random_scalars(full_test_size, get_rng(seed));

        let mut scalar_vec_of_vec: Vec<Vec<ScalarField>> = Vec::new();

        for i in 0..batches {
            scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result = scalars_batch.clone();

        // do batch ntt
        ntt(&mut ntt_result, false, false, Ordering::kNN, false, batches);

        let mut ntt_result_vec_of_vec = Vec::new();

        // do ntt for every chunk
        for i in 0..batches {
            ntt_result_vec_of_vec.push(scalar_vec_of_vec[i].clone());

            ntt(&mut ntt_result_vec_of_vec[i], false, false, Ordering::kNN, false, 1);
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
        ntt(&mut intt_result, true, false, Ordering::kNN, false, batches);

        let mut intt_result_vec_of_vec = Vec::new();

        // do intt for every chunk
        for i in 0..batches {
            intt_result_vec_of_vec.push(ntt_result_vec_of_vec[i].clone());
            // intt(&mut intt_result_vec_of_vec[i], 0);
            ntt(&mut intt_result_vec_of_vec[i], true, false, Ordering::kNN, false, 1);
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
