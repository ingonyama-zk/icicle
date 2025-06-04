#pragma once

#include "errors.h"
#include "icicle/errors.h"
#include "icicle/ntt.h"
#include "ntt.h"

namespace icicle {

  struct NegacyclicNTTConfig {
    icicleStreamHandle stream = nullptr; /**< Stream for asynchronous execution. */
    bool are_inputs_on_device = false;   /**< True if inputs are on device, false if on host. Default value is false. */
    bool are_outputs_on_device = false; /**< True if outputs are on device, false if on host. Default value is false. */
    bool is_async = false;              /**< True if operation is asynchronous. Default value is false. */
    ConfigExtension* ext = nullptr;     /**< Backend-specific extensions. */
  };

  template <typename Rq, typename Tq>
  eIcicleError ntt(const Rq* rq_polynomials, size_t size, const NegacyclicNTTConfig& config, Tq* tq_polynomials)
  {
    using Zq = typename Rq::Base;
    const int degree = Rq::d;

    const Zq psi = Zq::omega(log2(2 * degree));
    ICICLE_CHECK(ntt_init_domain(psi, NTTInitDomainConfig{}));

    NTTConfig<Zq> cyclic_ntt_config;
    cyclic_ntt_config.stream = config.stream;
    cyclic_ntt_config.are_inputs_on_device = config.are_inputs_on_device;
    cyclic_ntt_config.are_outputs_on_device = config.are_outputs_on_device;
    cyclic_ntt_config.is_async = config.is_async;
    cyclic_ntt_config.ext = config.ext;
    cyclic_ntt_config.batch_size = size;
    cyclic_ntt_config.coset_gen = psi;

    // TODO oredering

    return ntt((Zq*)rq_polynomials, degree, NTTDir::kForward, cyclic_ntt_config, (Zq*)tq_polynomials);
  }

  template <typename Rq, typename Tq>
  eIcicleError intt(const Tq* tq_polynomials, size_t size, const NegacyclicNTTConfig& config, Rq* rq_polynomials)
  {
    using Zq = typename Rq::Base;
    const int degree = Rq::d;

    const Zq psi = Zq::omega(log2(2 * degree));
    ICICLE_CHECK(ntt_init_domain(psi, NTTInitDomainConfig{}));

    NTTConfig<Zq> cyclic_ntt_config;
    cyclic_ntt_config.stream = config.stream;
    cyclic_ntt_config.are_inputs_on_device = config.are_inputs_on_device;
    cyclic_ntt_config.are_outputs_on_device = config.are_outputs_on_device;
    cyclic_ntt_config.is_async = config.is_async;
    cyclic_ntt_config.ext = config.ext;
    cyclic_ntt_config.batch_size = size;
    cyclic_ntt_config.coset_gen = psi;

    // TODO oredering

    return ntt((Zq*)tq_polynomials, degree, NTTDir::kInverse, cyclic_ntt_config, (Zq*)rq_polynomials);
  }

} // namespace icicle