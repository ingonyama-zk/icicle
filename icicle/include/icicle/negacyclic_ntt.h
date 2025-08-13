#pragma once

#include "errors.h"
#include "icicle/errors.h"
#include "icicle/ntt.h"
#include "ntt.h"

namespace icicle {

  /**
   * @brief Configuration for negacyclic NTT over R_q = Z_q[x] / (x^d + 1)
   *
   * Supports stream execution, device/host memory selection, and optional async behavior.
   */
  struct NegacyclicNTTConfig {
    icicleStreamHandle stream = nullptr; ///< CUDA or host stream for async execution
    bool are_inputs_on_device = false;   ///< True if inputs reside on device memory
    bool are_outputs_on_device = false;  ///< True if outputs reside on device memory
    bool is_async = false;               ///< True if operation is non-blocking
    ConfigExtension* ext = nullptr;      ///< Optional backend-specific configuration
  };

  /**
   * @brief Negacyclic NTT for R_q = Z_q[x] / (x^d + 1)
   *
   * @tparam PolyRing    Polynomial type with static degree and base field Zq
   * @param input        Input polynomials (in coefficient or evaluation domain)
   * @param size         Number of polynomials
   * @param dir          Direction (ntt/intt)
   * @param config       NTT execution and layout configuration
   * @param output       Output polynomials (in evaluation or coefficient domain)
   */
  template <typename PolyRing>
  eIcicleError ntt(const PolyRing* input, size_t size, NTTDir dir, const NegacyclicNTTConfig& config, PolyRing* output);

} // namespace icicle
