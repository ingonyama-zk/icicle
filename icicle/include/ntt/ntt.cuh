#pragma once
#ifndef NTT_H
#define NTT_H

#include <cuda_runtime.h>

#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"
#include "gpu-utils/sharedmem.cuh"
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
   * @param fast_twiddles_mode A mode where more memory is allocated for twiddle factors in exchange for faster compute.
   * In this mode need additional 4N memory when N is the largest NTT size to be supported (which is derived by the
   * primitive_root).
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  template <typename S, typename R = S>
  cudaError_t init_domain(
    R primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode = false);

  /**
   * Releases and deallocates resources associated with the domain initialized for performing NTTs.
   * This function should be called to clean up resources once they are no longer needed.
   * It's important to note that after calling this function, any operation that relies on the released domain will
   * fail unless init_domain is called again to reinitialize the resources. Therefore, ensure that release_domain is
   * only called when the operations requiring the NTT domain are completely finished and the domain is no longer
   * needed.
   * Also note that it is releasing the domain associated to the specific device.
   * @param ctx Details related to the device context such as its id and stream id.
   * @return `cudaSuccess` if the resource release was successful, indicating that the domain and its associated
   * resources have been properly deallocated. Returns an error code otherwise, indicating failure to release
   * the resources. The error code can be used to diagnose the problem.
   * */
  template <typename S>
  cudaError_t release_domain(device_context::DeviceContext& ctx);

  /* Returns the basic root of unity Wn
   * @param logn log size of the required root.
   * @return Wn root of unity
   */
  template <typename S>
  S get_root_of_unity(uint64_t max_size);

  /* Returns the basic root of unity Wn corresponding to the basic root used to initialize the domain.
   * This function can be called only after InitializeDomain()!
   * Useful when computing NTT on cosets. In that case we must use the root W_2n that is between W_n and W_n+1.
   * @param logn log size of the required root.
   * @param ctx Details related to the device such as its id and stream id.
   * @return Wn root of unity corresponding to logn and the basic root used for initDomain(root)
   */
  template <typename S>
  S get_root_of_unity_from_domain(uint64_t logn, device_context::DeviceContext& ctx);

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
   *
   * Mixed-Radix NTT: digit-reversal is a generalization of bit-reversal where the latter is a special case with 1b
   * digits. Mixed-radix NTTs of different sizes would generate different reordering of inputs/outputs. Having said
   * that, for a given size N it is guaranteed that every two mixed-radix NTTs of size N would have the same
   * digit-reversal pattern. The following orderings kNM and kMN are conceptually like kNR and kRN but for
   * mixed-digit-reordering. Note that for the cases '(1) NTT, (2) elementwise ops and (3) INTT' kNM and kMN are most
   * efficient.
   * Note: kNR, kRN, kRR refer to the radix-2 NTT reversal pattern. Those cases are supported by mixed-radix NTT with
   * reduced efficiency compared to kNM and kMN.
   * - kNM: inputs are natural-order and outputs are digit-reversed-order (=mixed).
   * - kMN: inputs are digit-reversed-order (=mixed) and outputs are natural-order.
   */
  enum class Ordering { kNN, kNR, kRN, kRR, kNM, kMN };

  /**
   * @enum NttAlgorithm
   * Which NTT algorithm to use. options are:
   * - Auto: implementation selects automatically based on heuristic. This value is a good default for most cases.
   * - Radix2: explicitly select radix-2 NTT algorithm
   * - MixedRadix: explicitly select mixed-radix NTT algorithm
   */
  enum class NttAlgorithm { Auto, Radix2, MixedRadix };

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
    bool columns_batch;                /**< True if the batches are the columns of an input matrix
                                       (they are strided in memory with a stride of ntt size) Default value: false.  */
    Ordering ordering;          /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
                                 *   `Ordering::kNN`. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async;              /**< Whether to run the NTT asynchronously. If set to `true`, the NTT function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                 *   function will block the current CPU thread. */
    NttAlgorithm ntt_algorithm; /**< Explicitly select the NTT algorithm. Default value: Auto (the implementation
                             selects radix-2 or mixed-radix algorithm based on heuristics). */
  };

  /**
   * A function that returns the default value of [NTTConfig](@ref NTTConfig) for the [NTT](@ref NTT) function.
   * @return Default value of [NTTConfig](@ref NTTConfig).
   */
  template <typename S>
  NTTConfig<S>
  default_ntt_config(const device_context::DeviceContext& ctx = device_context::get_default_device_context());

  /**
   * A function that computes NTT or iNTT in-place. It's necessary to call [init_domain](@ref init_domain) with an
   * appropriate primitive root before calling this function (only one call to `init_domain` should suffice for all
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
  cudaError_t ntt(const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output);

} // namespace ntt

#endif