use icicle_cuda_runtime::device_context::DeviceContext;
use std::os::raw::c_int;

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
#[derive(Debug)]
pub struct NTTConfigCuda<'a, E, S> {
    pub inout: *mut E,
    /**< Input that's mutated in-place by this function. Length of this array needs to be \f$ size \cdot config.batch_size \f$.
    *   Note that if inputs are in Montgomery form, the outputs will be as well and vice-verse: non-Montgomery inputs produce non-Montgomety outputs.*/
    pub is_input_on_device: bool,
    /**< True if inputs/outputs are on device and false if they're on host. Default value: false. */
    pub is_inverse: bool,
    /**< True if true . Default value: false. */
    pub ordering: Ordering,
    /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`. */
    pub decimation: Decimation,
    /**< Decimation of the algorithm, see [Decimation](@ref Decimation). Default value: `Decimation::kDIT`.
    *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
    *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
    *   `Decimation::kDIT` and if ordering is `Ordering::kNR` — to `Decimation::kDIF`. */
    pub butterfly: Butterfly,
    /**< Butterfly used by the NTT. See [Butterfly](@ref Butterfly). Default value: `Butterfly::kCooleyTukey`.
    *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
    *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
    *   `Butterfly::kCooleyTukey` and if ordering is `Ordering::kNR` — to `Butterfly::kGentlemanSande`. */
    pub is_coset: bool,
    /**< If false, NTT is computed on a subfield given by [twiddles](@ref twiddles). If true, NTT is computed
    *   on a coset of [twiddles](@ref twiddles) given by [the coset generator](@ref coset_gen), so:
    *   \f$ \{coset\_gen\cdot\omega^0, coset\_gen\cdot\omega^1, \dots, coset\_gen\cdot\omega^{n-1}\} \f$. Default value: false. */
    pub coset_gen: *const S,
    /**< The field element that generates a coset if [is_coset](@ref is_coset) is true.
    *   Otherwise should be set to `nullptr`. Default value: `nullptr`. */
    pub twiddles: *const S,
    /**< "Twiddle factors", (or "domain", or "roots of unity") on which the NTT is evaluated.
    *   This pointer is expected to live on device. The order is as follows:
    *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle factors
    *   are generated online using the default generator (TODO: link to twiddle gen here) and function
    *   [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    pub inv_twiddles: *const S,
    /**< "Inverse twiddle factors", (or "domain", or "roots of unity") on which the iNTT is evaluated.
    *   This pointer is expected to live on device. The order is as follows:
    *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle factors
    *   are generated online using the default generator (TODO: link to twiddle gen here) and function
    *   [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    pub size: c_int,
    /**< NTT size \f$ n \f$. If a batch of NTTs (which all need to have the same size) is computed, this is the size of 1 NTT. */
    pub batch_size: c_int,
    /**< The number of NTTs to compute. Default value: 1. */
    pub is_preserving_twiddles: bool,
    /**< If true, twiddle factors are preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    pub is_output_on_device: bool,
    /**< If true, output is preserved on device for subsequent use in config and not freed after calculation. Default value: false. */
    pub ctx: DeviceContext<'a>, /*< Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext). */
}


