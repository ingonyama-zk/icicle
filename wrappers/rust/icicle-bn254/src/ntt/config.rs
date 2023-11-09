use icicle_cuda_runtime::{get_default_device_context, DeviceContext, DevicePointer};
use std::os::raw::c_int;

use crate::curve::*;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    kNN,
    kNR,
    kRN,
    kRR,
}

impl Default for Ordering {
    fn default() -> Ordering {
        Ordering::kNN
    }
}

/**
 * @enum Decimation
 * Decimation of the NTT algorithm:
 * - kDIT: decimation in time.
 * - kDIF: decimation in frequency.
 */
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decimation {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Butterfly {
    kCooleyTukey,
    kGentlemanSande,
}

/**
 * @struct NTTConfig
 * Struct that encodes NTT parameters to be passed into the [ntt](@ref ntt) function.
 */
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NTTConfigCuda<E, S> {
    pub(super) inout: *mut E,
    /**< Input that's mutated in-place by this function. Length of this array needs to be \f$ size \cdot config.batch_size \f$.
     *   Note that if inputs are in Montgomery form, the outputs will be as well and vice-verse: non-Montgomery inputs produce non-Montgomety outputs.*/
    pub(super) is_input_on_device: bool,
    /**< True if inputs/outputs are on device and false if they're on host. Default value: false. */
    pub(super) is_inverse: bool,
    /**< True if true . Default value: false. */
    pub(super) ordering: Ordering,
    /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`. */
    pub(super) decimation: Decimation,
    /**< Decimation of the algorithm, see [Decimation](@ref Decimation). Default value: `Decimation::kDIT`.
     *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
     *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
     *   `Decimation::kDIT` and if ordering is `Ordering::kNR` — to `Decimation::kDIF`. */
    pub(super) butterfly: Butterfly,
    /**< Butterfly used by the NTT. See [Butterfly](@ref Butterfly). Default value: `Butterfly::kCooleyTukey`.
     *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
     *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
     *   `Butterfly::kCooleyTukey` and if ordering is `Ordering::kNR` — to `Butterfly::kGentlemanSande`. */
    pub(super) is_coset: bool,
    /**< If false, NTT is computed on a subfield given by [twiddles](@ref twiddles). If true, NTT is computed
     *   on a coset of [twiddles](@ref twiddles) given by [the coset generator](@ref coset_gen), so:
     *   \f$ \{coset\_gen\cdot\omega^0, coset\_gen\cdot\omega^1, \dots, coset\_gen\cdot\omega^{n-1}\} \f$. Default value: false. */
    pub(super) coset_gen: *const S,
    /**< The field element that generates a coset if [is_coset](@ref is_coset) is true.
     *   Otherwise should be set to `nullptr`. Default value: `nullptr`. */
    pub(super) twiddles: *const S,
    /**< "Twiddle factors", (or "domain", or "roots of unity") on which the NTT is evaluated.
     *   This pointer is expected to live on device. The order is as follows:
     *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle factors
     *   are generated online using the default generator (TODO: link to twiddle gen here) and function
     *   [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    pub(super) inv_twiddles: *const S,
    /**< "Inverse twiddle factors", (or "domain", or "roots of unity") on which the iNTT is evaluated.
     *   This pointer is expected to live on device. The order is as follows:
     *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle factors
     *   are generated online using the default generator (TODO: link to twiddle gen here) and function
     *   [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    pub(super) size: c_int,
    /**< NTT size \f$ n \f$. If a batch of NTTs (which all need to have the same size) is computed, this is the size of 1 NTT. */
    pub(super) batch_size: c_int,
    /**< The number of NTTs to compute. Default value: 1. */
    pub(super) is_preserving_twiddles: bool,
    /**< If true, twiddle factors are preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    pub(super) is_output_on_device: bool,
    /**< If true, output is preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    pub(super) ctx: DeviceContext, /*< Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext). */
}

pub(super) type ECNTTConfig = NTTConfigCuda<G1Projective, ScalarField>;
pub(super) type NTTConfig = NTTConfigCuda<ScalarField, ScalarField>;

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

pub(super) fn get_ntt_default_config<E, S>(size: usize) -> NTTConfigCuda<E, S> {
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
        ctx: DeviceContext {
            device_id: 0,
            stream: 0,
            mempool: 0,
        },
    }
}
