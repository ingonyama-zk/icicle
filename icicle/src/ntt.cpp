#include "icicle/backend/ntt_backend.h"
#include "icicle/dispatcher.h"

using namespace field_config;
namespace icicle {

  /*************************** NTT ***************************/
  ICICLE_DISPATCHER_INST(NttDispatcher, ntt, NttImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, ntt)(
    const scalar_t* input, int size, NTTDir dir, const NTTConfig<scalar_t>* config, scalar_t* output)
  {
    return NttDispatcher::execute(input, size, dir, *config, output);
  }

  template <>
  eIcicleError ntt(const scalar_t* input, int size, NTTDir dir, const NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return CONCAT_EXPAND(FIELD, ntt)(input, size, dir, &config, output);
  }

  /*************************** INIT DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttInitDomainDispatcher, ntt_init_domain, NttInitDomainImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(FIELD, ntt_init_domain)(const scalar_t* primitive_root, const NTTInitDomainConfig* config)
  {
    return NttInitDomainDispatcher::execute(*primitive_root, *config);
  }

  template <>
  eIcicleError ntt_init_domain(const scalar_t& primitive_root, const NTTInitDomainConfig& config)
  {
    return CONCAT_EXPAND(FIELD, ntt_init_domain)(&primitive_root, &config);
  }

  /*************************** RELEASE DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttReleaseDomainDispatcher, ntt_release_domain, NttReleaseDomainImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, ntt_release_domain)()
  {
    // Note: passing zero is a workaround for the function required per field but need to differentiate by type when
    // calling
    return NttReleaseDomainDispatcher::execute(scalar_t::zero());
  }

  template <>
  eIcicleError ntt_release_domain<scalar_t>()
  {
    return CONCAT_EXPAND(FIELD, ntt_release_domain)();
  }

  /*************************** GET ROOT OF UNITY ***************************/
  extern "C" eIcicleError CONCAT_EXPAND(FIELD, get_root_of_unity)(uint64_t max_size, scalar_t* rou)
  {
    const auto log_max_size = static_cast<uint32_t>(std::ceil(std::log2(max_size)));
    if (scalar_t::get_omegas_count() < log_max_size) {
      ICICLE_LOG_ERROR << "no root-of-unity of order " << log_max_size << " in field " << typeid(scalar_t).name();
      return eIcicleError::INVALID_ARGUMENT;
    }
    *rou = scalar_t::omega(log_max_size);
    return eIcicleError::SUCCESS;
  }

  template <>
  eIcicleError get_root_of_unity(uint64_t max_size, scalar_t* rou)
  {
    return CONCAT_EXPAND(FIELD, get_root_of_unity)(max_size, rou);
  }

  /*************************** GET ROOT OF UNITY FROM DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttRouFromDomainDispatcher, ntt_get_rou_from_domain, NttGetRouFromDomainImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, get_root_of_unity_from_domain)(uint64_t logn, scalar_t* rou)
  {
    return NttRouFromDomainDispatcher::execute(logn, rou);
  }

  template <>
  eIcicleError get_root_of_unity_from_domain<scalar_t>(uint64_t logn, scalar_t* rou)
  {
    return CONCAT_EXPAND(FIELD, get_root_of_unity_from_domain)(logn, rou);
  }

#ifdef EXT_FIELD
  // extensions fields reuse the scalar domain (??)
  ICICLE_DISPATCHER_INST(NttExtFieldDispatcher, extension_ntt, NttExtFieldImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, extension_ntt)(
    const extension_t* input, int size, NTTDir dir, const NTTConfig<scalar_t>* config, extension_t* output)
  {
    return NttExtFieldDispatcher::execute(input, size, dir, *config, output);
  }

  template <>
  eIcicleError
  ntt(const extension_t* input, int size, NTTDir dir, const NTTConfig<scalar_t>& config, extension_t* output)
  {
    return CONCAT_EXPAND(FIELD, extension_ntt)(input, size, dir, &config, output);
  }
#endif // EXT_FIELD

#ifdef RING
  // /*************************** NTT ***************************/
  ICICLE_DISPATCHER_INST(NttRingRnsDispatcher, ring_rns_ntt, NttRingRnsImpl);

  extern "C" eIcicleError CONCAT_EXPAND(RING, ring_rns_ntt)(
    const scalar_rns_t* input, int size, NTTDir dir, const NTTConfig<scalar_rns_t>* config, scalar_rns_t* output)
  {
    return NttRingRnsDispatcher::execute(input, size, dir, *config, output);
  }

  template <>
  eIcicleError
  ntt(const scalar_rns_t* input, int size, NTTDir dir, const NTTConfig<scalar_rns_t>& config, scalar_rns_t* output)
  {
    return CONCAT_EXPAND(RING, ring_rns_ntt)(input, size, dir, &config, output);
  }

  // /*************************** INIT DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttInitDomainRingRnsDispatcher, ring_rns_ntt_init_domain, NttInitDomainRingRnsImpl);

  extern "C" eIcicleError
  CONCAT_EXPAND(RING, ring_rns_ntt_init_domain)(const scalar_rns_t* primitive_root, const NTTInitDomainConfig* config)
  {
    return NttInitDomainRingRnsDispatcher::execute(*primitive_root, *config);
  }

  template <>
  eIcicleError ntt_init_domain(const scalar_rns_t& primitive_root, const NTTInitDomainConfig& config)
  {
    return CONCAT_EXPAND(RING, ring_rns_ntt_init_domain)(&primitive_root, &config);
  }

  //   /*************************** RELEASE DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(NttReleaseDomainRingRnsDispatcher, ring_rns_ntt_release_domain, NttReleaseDomainRingRnsImpl);

  extern "C" eIcicleError CONCAT_EXPAND(RING, ring_rns_ntt_release_domain)()
  {
    // Note: passing zero is a workaround for the function required per ring but need to differentiate by type
    // when calling
    return NttReleaseDomainRingRnsDispatcher::execute(scalar_rns_t::zero());
  }

  template <>
  eIcicleError ntt_release_domain<scalar_rns_t>()
  {
    return CONCAT_EXPAND(RING, ring_rns_ntt_release_domain)();
  }

  /*************************** GET ROOT OF UNITY ***************************/
  extern "C" eIcicleError CONCAT_EXPAND(RING, ring_rns_get_root_of_unity)(uint64_t max_size, scalar_rns_t* rou)
  {
    const auto log_max_size = static_cast<uint32_t>(std::ceil(std::log2(max_size)));
    if (scalar_rns_t::get_omegas_count() < log_max_size) {
      ICICLE_LOG_ERROR << "no root-of-unity of order " << log_max_size << " in ring " << typeid(scalar_rns_t).name();
      return eIcicleError::INVALID_ARGUMENT;
    }
    *rou = scalar_rns_t::omega(log_max_size);
    return eIcicleError::SUCCESS;
  }

  template <>
  eIcicleError get_root_of_unity(uint64_t max_size, scalar_rns_t* rou)
  {
    return CONCAT_EXPAND(RING, ring_rns_get_root_of_unity)(max_size, rou);
  }

  /*************************** GET ROOT OF UNITY FROM DOMAIN ***************************/
  ICICLE_DISPATCHER_INST(
    NttRouFromDomainRingRnsDispatcher, ring_rns_ntt_get_rou_from_domain, NttGetRouFromDomainRingRnsImpl);

  extern "C" eIcicleError CONCAT_EXPAND(RING, ring_rns_get_root_of_unity_from_domain)(uint64_t logn, scalar_rns_t* rou)
  {
    return NttRouFromDomainRingRnsDispatcher::execute(logn, rou);
  }

  template <>
  eIcicleError get_root_of_unity_from_domain<scalar_rns_t>(uint64_t logn, scalar_rns_t* rou)
  {
    return CONCAT_EXPAND(RING, ring_rns_get_root_of_unity_from_domain)(logn, rou);
  }
#endif // RING
} // namespace icicle