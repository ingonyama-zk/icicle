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
   * @brief Negacyclic NTT wrapper for R_q = Z_q[x] / (x^d + 1)
   *
   * Applies an NTT or inverse NTT (depending on `dir`) using a root of unity
   * compatible with negacyclic convolution, i.e., using ψ where ω = ψ².
   *
   * @tparam PolyRing    Polynomial type with static degree and base field Zq
   * @param Rq_vec       Input polynomials (in coefficient or evaluation domain)
   * @param size         Number of polynomials
   * @param dir          Direction: NTT_FORWARD or NTT_INVERSE
   * @param config       NTT execution and layout configuration
   * @param Tq_vec       Output polynomials (in evaluation or coefficient domain)
   */
  template <typename PolyRing>
  eIcicleError ntt(const PolyRing* input, size_t size, NTTDir dir, const NegacyclicNTTConfig& config, PolyRing* output)
  {
    using Zq = typename PolyRing::Base;
    constexpr int degree = PolyRing::d;

    // Use ψ as a primitive 2n-th root of unity for negacyclic transform
    const Zq psi = Zq::omega(log2(2 * degree));
    // Note that this is called only once per device
    ICICLE_CHECK(ntt_init_domain(psi, NTTInitDomainConfig{}));

    // Computing a coset-NTT in Zq. It is equivalent to a negacyclic NTT
    NTTConfig<Zq> cyclic_ntt_config = default_ntt_config<Zq>();
    cyclic_ntt_config.stream = config.stream;
    cyclic_ntt_config.are_inputs_on_device = config.are_inputs_on_device;
    cyclic_ntt_config.are_outputs_on_device = config.are_outputs_on_device;
    cyclic_ntt_config.is_async = config.is_async;
    cyclic_ntt_config.ext = config.ext;
    cyclic_ntt_config.batch_size = size;
    cyclic_ntt_config.columns_batch = false;
    cyclic_ntt_config.coset_gen = psi;
    cyclic_ntt_config.ordering =
      dir == NTTDir::kForward ? Ordering::kNR : Ordering::kRN; // Maybe can consider Mixed-order but it's not faster

    return ntt(reinterpret_cast<const Zq*>(input), degree, dir, cyclic_ntt_config, reinterpret_cast<Zq*>(output));
  }

} // namespace icicle