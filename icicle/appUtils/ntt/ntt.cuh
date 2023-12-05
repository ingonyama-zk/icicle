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
 * x + \dots + p_{n-1} \cdot x^{n-1} \f$ on special subfields called "roots of unity", or "twiddle factors" (optionally
 * shifted by an additional element called "coset generator"): \f[ NTT(p) = \{ p(\omega^0), p(\omega^1), \dots,
 * p(\omega^{n-1}) \} \f] Inverse NTT, or iNTT solves the inverse problem of computing coefficients of \f$ p(x) \f$
 * given evaluations \f$ \{ p(\omega^0), p(\omega^1), \dots, p(\omega^{n-1}) \} \f$. If not specified otherwise,
 * \f$ n \f$ is a power of 2.
 */
namespace ntt {

  /**
   * @enum Ordering
   * How to order inputs and outputs of the NTT. If needed, use this field to specify decimation: decimation in time
   * (DIT) corresponds to `Ordering::kRN` while decimation in frequency (DIF) to `Ordering::kNR`. Also, to specify
   * butterfly to be used, select `Ordering::kRN` for Cooley-Tukey and `Ordering::kNR` for Gentleman-Sande. There's
   * no implication that a certain decimation or butterfly will actually be used under the hood, this is just for
   * compatibility with codebases that use "decimation" and "butterfly" to denote ordering of inputs and outputs.
   *
   * Ordering options are:
   * - kNN: inputs and outputs are natural-order (example of natural ordering: \f$ \{a_0, a_1, a_2, a_3, a_4, a_5, a_6,
   * a_7\} \f$).
   * - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: \f$ \{a_0,
   * a_4, a_2, a_6, a_1, a_5, a_3, a_7\} \f$).
   * - kRN: inputs are bit-reversed-order and outputs are natural-order.
   * - kRR: inputs and outputs are bit-reversed-order.
   */
  enum class Ordering { kNN, kNR, kRN, kRR };

  /**
   * @struct Domain
   * Struct containing information about the domain on which (i)NTT is evaluated: twiddle factors and coset generator.
   * Twiddle factors are private, static and can only be set using [GenerateDomain](@ref GenerateDomain) function.
   * The internal representation of twiddles is prone to change in accordance with changing [NTT](@ref NTT) algorithm.
   * @tparam S The type of "twiddle factors" \f$ \{ \omega^i \} \f$ and coset generator. Must be a field.
   */
  template <typename S>
  struct Domain {
    S coset_gen; /**< Scalar element that specifies a coset to be used in (i)NTT. Default value: `S::one()`. */
  private:
    static int max_size;
    static int log_max_size;
    static S* twiddles;
    static S* inv_twiddles;
  }

  /**
   * Generate [Domain](@ref Domain) struct that supports all NTTs of sizes under a certain threshold.
   * @param primitive_root Primitive root in field `S` of order \f$ 2^{log\_size} \f$.
   * @param log_size Binary logarithm of order of `primitive_root`. Should be the smallest value that's large enough
   * to support any NTT you might want to perform.
   * @return [Domain](@ref Domain) with appropriate twiddle factors and default coset generator (`S::one()`).
   */
  template <typename S>
  cudaError_t Domain GenerateDomain(S primitive_root, int log_size);

  /**
   * @struct NTTConfig
   * Struct that encodes NTT parameters to be passed into the [NTT](@ref NTT) function.
   */
  struct NTTConfig {
    Ordering ordering;          /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
                                 *   `Ordering::kNN`. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    int batch_size;             /**< The number of NTTs to compute. Default value: 1. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async;              /**< Whether to run the NTT asyncronously. If set to `true`, the NTT function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                 *   function will block the current CPU thread. */
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream. */
  };

  /**
   * A function that returns the default value of [NTTConfig](@ref NTTConfig) for the [NTT](@ref NTT) function.
   * @return Default value of [NTTConfig](@ref NTTConfig).
   */
  extern "C" NTTConfig DefaultNTTConfig();

  /**
   * A function that computes NTT or iNTT in-place.
   * @param inout Input that's mutated in-place by this function. Length of this array needs to be \f$ size \cdot
   * config.batch\_size \f$. Note that if inputs are in Montgomery form, the outputs will be as well and vice-versa:
   * non-Montgomery inputs produce non-Montgomety outputs.
   * @param size NTT size. If a batch of NTTs (which all need to have the same size) is computed, this is the size
   * of 1 NTT, so it must equal the size of `inout` divided by `config.batch_size`.
   * @param is_inverse True for inverse NTT and false for direct NTT. Default value: false.
   * @param domain [Domain](@ref Domain) on which NTT is evaluated.
   * @param config [NTTConfig](@ref NTTConfig) used in this NTT.
   * @tparam E The type of inputs and outputs (i.e. coefficients \f$ \{p_i\} \f$ and values \f$ p(x) \f$). Must be a
   * group.
   * @tparam S The type of "twiddle factors" \f$ \{ \omega^i \} \f$. Must be a field. Often (but not always) `S=E`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename E, typename S>
  cudaError_t NTT(E* inout, int size, bool is_inverse, Domain<S> domain, NTTConfig<S>* config);

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
