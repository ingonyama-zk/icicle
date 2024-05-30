#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

using namespace field_config;

namespace icicle {

  /*************************** Frontend APIs ***************************/
  /**
   * @enum NTTDir
   * Whether to perform forward NTT, or inverse NTT (iNTT). Mathematically, forward NTT computes polynomial
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

  // TODO Yuval: move to cuda backend
  /**
   * @enum NttAlgorithm
   * Which NTT algorithm to use. options are:
   * - Auto: implementation selects automatically based on heuristic. This value is a good default for most cases.
   * - Radix2: explicitly select radix-2 NTT algorithm
   * - MixedRadix: explicitly select mixed-radix NTT algorithm
   */
  // enum class NttAlgorithm { Auto, Radix2, MixedRadix };

  /**
   * @struct NTTConfig
   * Struct that encodes NTT parameters to be passed into the [NTT](@ref NTT) function.
   */
  template <typename S>
  struct NTTConfig {
    icicleStreamHandle stream;  /**< stream for async execution. */
    S coset_gen;                /**< Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()`
                                 *   (corresponding to no coset being used). */
    int batch_size;             /**< The number of NTTs to compute. Default value: 1. */
    bool columns_batch;         /**< True if the batches are the columns of an input matrix
                                (they are strided in memory with a stride of ntt size) Default value: false.  */
    Ordering ordering;          /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
                                 *   `Ordering::kNN`. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async;              /**< Whether to run the NTT asynchronously. If set to `true`, the NTT function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the NTT
                                 *   function will block the current CPU thread. */

    ConfigExtension ext; /** backend specific extensions*/
  };

  /**
   * A function that returns the default value of [NTTConfig](@ref NTTConfig) for the [NTT](@ref NTT) function.
   * @return Default value of [NTTConfig](@ref NTTConfig).
   */
  template <typename S>
  NTTConfig<S> default_ntt_config()
  {
    NTTConfig<S> config = {
      nullptr,       // stream
      S::one(),      // coset_gen
      1,             // batch_size
      false,         // columns_batch
      Ordering::kNN, // ordering
      false,         // are_inputs_on_device
      false,         // are_outputs_on_device
      false,         // is_async
    };
    return config;
  }

  // template APIs

  template <typename S, typename E>
  eIcicleError ntt(const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output);

  // field specific APIs. TODO Yuval move to api headers like icicle V2
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, ntt)(
    const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output);

  /*************************** Backend registration ***************************/

  using NttImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)>;

  void register_ntt(const std::string& deviceType, NttImpl impl);

#define REGISTER_NTT_BACKEND(DEVICE_TYPE, FUNC)                                                                        \
  namespace {                                                                                                          \
    static bool _reg_vec_add = []() -> bool {                                                                          \
      register_ntt(DEVICE_TYPE, FUNC);                                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle