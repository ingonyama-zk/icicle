#pragma once
#ifndef NTT_H
#define NTT_H

#include <cuda_runtime.h>

#include "curves/curve_config.cuh"
#include "utils/device_context.cuh"
#include "utils/error_handler.cuh"
#include "utils/sharedmem.cuh"
#include "utils/utils_kernels.cuh"
#include "utils/utils.h"

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
   * Generate a domain that supports all NTTs of sizes under a certain threshold. Note that the this function might
   * be expensive, so if possible it should be called before all time-critical operations.
   * It's assumed that during program execution only the coset generator might change, but twiddles stay fixed, so
   * they are initialized at the first call of this function and don't change afterwards.
   * @param primitive_root Primitive root in field `S` of order \f$ 2^s \f$. This should be the smallest power-of-2
   * order that's large enough to support any NTT you might want to perform.
   * @param ctx Details related to the device such as its id and stream id.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename S>
  cudaError_t InitDomain(S primitive_root, device_context::DeviceContext& ctx);

  /**
   * @enum NTTDir
   * Whether to perform normal forward NTT, or inverse NTT (iNTT). Mathematically, forward NTT computes polynomial
   * evaluations from coefficients while inverse NTT computes coefficients from evaluations.
   */
  enum class NTTDir { kForward, kInverse };

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
   * @struct NTTConfig
   * Struct that encodes NTT parameters to be passed into the [NTT](@ref NTT) function.
   */
  template <typename S>
  struct NTTConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream. */
    S coset_gen;                       /**< Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()`
                                        *   (corresponding to no coset being used). */
    int batch_size;                    /**< The number of NTTs to compute. Default value: 1. */
    Ordering ordering;          /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
                                 *   `Ordering::kNN`. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async;              /**< Whether to run the NTT asynchronously. If set to `true`, the NTT function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                 *   function will block the current CPU thread. */
  };

  /**
   * A function that returns the default value of [NTTConfig](@ref NTTConfig) for the [NTT](@ref NTT) function.
   * @return Default value of [NTTConfig](@ref NTTConfig).
   */
  template <typename S>
  NTTConfig<S> DefaultNTTConfig();

  /**
   * A function that computes NTT or iNTT in-place. It's necessary to call [InitDomain](@ref InitDomain) with an
   * appropriate primitive root before calling this function (only one call to `InitDomain` should suffice for all
   * NTTs).
   * @param input Input of the NTT. Length of this array needs to be \f$ size \cdot config.batch\_size \f$. Note
   * that if inputs are in Montgomery form, the outputs will be as well and vice-versa: non-Montgomery inputs produce
   * non-Montgomety outputs.
   * @param size NTT size. If a batch of NTTs (which all need to have the same size) is computed, this is the size
   * of 1 NTT, so it must equal the size of `inout` divided by `config.batch_size`.
   * @param dir Whether to compute forward or inverse NTT.
   * @param config [NTTConfig](@ref NTTConfig) used in this NTT.
   * @param output Buffer for the output of the NTT. Should be of the same size as `input`.
   * @tparam E The type of inputs and outputs (i.e. coefficients \f$ \{p_i\} \f$ and values \f$ p(x) \f$). Must be a
   * group.
   * @tparam S The type of "twiddle factors" \f$ \{ \omega^i \} \f$. Must be a field. Often (but not always) `S=E`.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename S, typename E>
  cudaError_t NTT(E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output);

} // namespace ntt

#endif