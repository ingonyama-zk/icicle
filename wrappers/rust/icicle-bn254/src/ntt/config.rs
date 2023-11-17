use std::os::raw::c_int;

use icicle_core::ntt::{Ordering, Decimation, Butterfly, NTTConfigCuda};
use icicle_cuda_runtime::device_context::{get_default_device_context, DeviceContext};
use crate::curve::*;

pub(super) type ECNTTConfig<'a> = NTTConfigCuda<'a, G1Projective, ScalarField>;
pub(super) type NTTConfig<'a> = NTTConfigCuda<'a, ScalarField, ScalarField>;

pub(super) fn get_ntt_config<E, S>(size: usize, ctx: DeviceContext) -> NTTConfigCuda<E, S> {
    //TODO: implement on CUDA side

    NTTConfigCuda::<E, S> {
        inout: 0 as _, // inout as *mut _ as *mut ScalarField,
        is_input_on_device: false,
        is_inverse: false,
        ordering: Ordering::kNN,
        decimation: Decimation::kDIF,
        butterfly: Butterfly::kCooleyTukey,
        is_coset: false,
        coset_gen: 0 as _,    //TODO: ?
        twiddles: 0 as _,     //TODO: ?,
        inv_twiddles: 0 as _, //TODO: ?,
        size: size as i32,
        batch_size: 0 as i32,
        is_preserving_twiddles: true,
        is_output_on_device: false,
        ctx,
    }
}

pub(super) fn get_ntt_default_config<E, S>(size: usize) -> NTTConfigCuda<'static, E, S> {
    //TODO: implement on CUDA side
    let ctx = get_default_device_context();

    // let root_of_unity = S::default(); //TODO: implement on CUDA side

    let config = get_ntt_config(size, ctx);

    config
}

pub(super) fn get_ntt_config_with_input(ntt_intt_result: &mut [ScalarField], size: usize, batches: usize) -> NTTConfig {
    NTTConfig {
        inout: ntt_intt_result as *mut _ as *mut ScalarField,
        is_input_on_device: false,
        is_inverse: false,
        ordering: Ordering::kNN,
        decimation: Decimation::kDIF,
        butterfly: Butterfly::kCooleyTukey,
        is_coset: false,
        coset_gen: &[ScalarField::zero()] as _, //TODO: ?
        twiddles: 0 as *const ScalarField,      //TODO: ?,
        inv_twiddles: 0 as *const ScalarField,  //TODO: ?,
        size: size as _,
        batch_size: batches as i32,
        is_preserving_twiddles: true,
        is_output_on_device: true,
        ctx: get_default_device_context(),
    }
}
