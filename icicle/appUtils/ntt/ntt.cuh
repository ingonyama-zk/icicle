#pragma once
#ifndef NTT_H
#define NTT_H

#include <cuda_runtime.h>

#include "../../curves/curve_config.cuh"
#include "../../utils/device_context.cuh"
#include "../../utils/error_handler.cuh"
#include "../../utils/sharedmem.cuh"
#include "../../utils/utils_kernels.cuh"

/**
 * @namespace ntt
 * Number Theoretic Transform, or NTT is a version of [fast Fourier
 * transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) where instead of real or complex numbers, inputs and
 * outputs belong to certain finite groups or fields. NTT computes the values of a polynomial \f$ p(x) = p_0 + p_1 \cdot
 * x + \dots + p_{n-1} \cdot x^{n-1} \f$ on special subfields called "roots of unity", or "twiddle factors": \f[ NTT(p)
 * = \{ p(\omega^0), p(\omega^1), \dots, p(\omega^{n-1}) \} \f] Inverse NTT, or iNTT solves the inverse problem of
 * computing coefficients of \f$ p(x) \f$ from evaluations \f$ \{ p(\omega^0), p(\omega^1), \dots, p(\omega^{n-1}) \}
 * \f$. If not specified otherwise, \f$ n \f$ is a power of 2.
 */
namespace ntt {

  /**
   * @enum Ordering
   * How to order inputs and outputs of the NTT:
   * - kNN: inputs and outputs are natural-order (example of natural ordering: \f$ \{a_0, a_1, a_2, a_3, a_4, a_5, a_6,
   * a_7\} \f$).
   * - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: \f$ \{a_0,
   * a_4, a_2, a_6, a_1, a_5, a_3, a_7\} \f$).
   * - kRN: inputs are bit-reversed-order and outputs are natural-order.
   * - kRR: inputs and outputs are bit-reversed-order.
   */
  enum class Ordering { kNN, kNR, kRN, kRR };

  /**
   * @enum Decimation
   * Decimation of the NTT algorithm:
   * - kDIT: decimation in time.
   * - kDIF: decimation in frequency.
   */
  enum class Decimation { kDIT, kDIF };

  /**
   * @enum Butterfly
   * [Butterfly](https://en.wikipedia.org/wiki/Butterfly_diagram) used in the NTT algorithm (i.e. what happens to each
   * pair of inputs on every iteration):
   * - kCooleyTukey: Cooley-Tukey butterfly.
   * - kGentlemanSande: Gentleman-Sande butterfly.
   */
  enum class Butterfly { kCooleyTukey, kGentlemanSande };

  /**
   * @struct NTTConfig
   * Struct that encodes NTT parameters to be passed into the [ntt](@ref ntt) function.
   */
  template <typename E, typename S>
  struct NTTConfig {
    E* inout; /**< Input that's mutated in-place by this function. Length of this array needs to be \f$ size \cdot
               * config.batch_size \f$. Note that if inputs are in Montgomery form, the outputs will be as well and
               * vice-verse: non-Montgomery inputs produce non-Montgomety outputs.*/
    bool are_inputs_on_device; /**< True if inputs/outputs are on device and false if they're on host. Default value:
                                  false. */
    bool is_inverse;           /**< True if true . Default value: false. */
    Ordering
      ordering; /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`. */
    Decimation
      decimation; /**< Decimation of the algorithm, see [Decimation](@ref Decimation). Default value:
                   * `Decimation::kDIT`.
                   *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
                   *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
                   *   `Decimation::kDIT` and if ordering is `Ordering::kNR` — to `Decimation::kDIF`. */
    Butterfly
      butterfly;     /**< Butterfly used by the NTT. See [Butterfly](@ref Butterfly). Default value:
                      * `Butterfly::kCooleyTukey`.
                      *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
                      *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to
                      *   `Butterfly::kCooleyTukey` and if ordering is `Ordering::kNR` — to `Butterfly::kGentlemanSande`. */
    bool is_coset;   /**< If false, NTT is computed on a subfield given by [twiddles](@ref twiddles). If true, NTT is
                      * computed   on a coset of [twiddles](@ref twiddles) given by [the coset generator](@ref coset_gen),
                      * so:   \f$ \{coset\_gen\cdot\omega^0, coset\_gen\cdot\omega^1, \dots, coset\_gen\cdot\omega^{n-1}\}
                      * \f$. Default value: false. */
    S* coset_gen;    /**< The field element that generates a coset if [is_coset](@ref is_coset) is true.
                      *   Otherwise should be set to `nullptr`. Default value: `nullptr`. */
    S* twiddles;     /**< "Twiddle factors", (or "domain", or "roots of unity") on which the NTT is evaluated.
                      *   This pointer is expected to live on device. The order is as follows:
                      *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle
                      * factors     are generated online using the default generator (TODO: link to twiddle gen here) and
                      * function     [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    S* inv_twiddles; /**< "Inverse twiddle factors", (or "domain", or "roots of unity") on which the iNTT is evaluated.
                      *   This pointer is expected to live on device. The order is as follows:
                      *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `nullptr`, twiddle
                      * factors are generated online using the default generator (TODO: link to twiddle gen here) and
                      * function [GenerateTwiddleFactors](@ref GenerateTwiddleFactors). Default value: `nullptr`. */
    int size; /**< NTT size \f$ n \f$. If a batch of NTTs (which all need to have the same size) is computed, this is
                  the size of 1 NTT. */
    int batch_size;              /**< The number of NTTs to compute. Default value: 1. */
    bool is_preserving_twiddles; /**< If true, twiddle factors are preserved on device for subsequent use in config and
                                    not freed after calculation. Default value: false. */
    bool is_output_on_device;    /**< If true, output is preserved on device for subsequent use in config and not freed
                                    after calculation. Default value: false. */
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. See
                                          [DeviceContext](@ref device_context::DeviceContext). */
  };

  /**
   * A function that computes NTT or iNTT in-place.
   * @param config [NTTConfig](@ref NTTConfig) used in this NTT.
   * @tparam E The type of inputs and outputs (i.e. coefficients \f$ \{p_i\} \f$ and values \f$ p(x) \f$). Must be a
   * group.
   * @tparam S The type of "twiddle factors" \f$ \{ \omega^i \} \f$. Must be a field. Often (but not always) `S=E`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E, typename S>
  cudaError_t NTT(NTTConfig<E, S>* config);

  /**
   * Generates twiddles \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$ from root of unity \f$ \omega \f$ and
   * stores them on device.
   * @param d_twiddles Input empty array on device to which twiddles are to be written.
   * @param n_twiddles Number of twiddle \f$ n \f$ factors to generate.
   * @param omega Root of unity \f$ \omega \f$.
   * @param ctx Details related to the device such as its id and stream id. See [DeviceContext](@ref
   * device_context::DeviceContext).
   * @tparam S The type of twiddle factors \f$ \{ \omega^i \} \f$.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename S>
  cudaError_t GenerateTwiddleFactors(S* d_twiddles, int n_twiddles, S omega, device_context::DeviceContext ctx);

} // namespace ntt

#endif
