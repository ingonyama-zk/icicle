#include "ntt.cuh"

/************************************** BACKEND REGISTRATION **************************************/

namespace icicle {

  eIcicleError
  ntt_cuda_init_domain(const Device& device, const scalar_t& primitive_root, const NTTInitDomainConfig& config)
  {
    using namespace device_context;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
    DeviceContext device_context{cuda_stream, (size_t)device.id, nullptr /*mempool*/};
    device_context.device_id = device.id;
    bool fast_twiddles = false;

    if (config.ext && config.ext->has(CUDA_NTT_FAST_TWIDDLES_MODE)) {
      fast_twiddles = config.ext->get<bool>(CUDA_NTT_FAST_TWIDDLES_MODE);
    }
    auto err = ntt::Domain<scalar_t>::init_domain(primitive_root, device_context, fast_twiddles);
    return translateCudaError(err);
  }

  eIcicleError ntt_cuda_release_domain(const Device& device, const scalar_t& dummy)
  {
    using namespace device_context;
    DeviceContext device_context = get_default_device_context();
    device_context.device_id = device.id;
    auto err = ntt::Domain<scalar_t>::release_domain(device_context);
    return translateCudaError(err);
  }

  template <typename S, typename E>
  eIcicleError ntt_cuda(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    auto err = ntt::ntt_cuda<S, E>(input, size, dir, config, device.id, output);
    return translateCudaError(err);
  }

  template <typename S>
  eIcicleError get_root_of_unity_from_domain_cuda(const Device& device, uint64_t logn, S& rou)
  {
    using namespace device_context;
    DeviceContext device_context = get_default_device_context();
    device_context.device_id = device.id;
    rou = ntt::Domain<S>::get_root_of_unity_from_domain(logn, device_context);
    return eIcicleError::SUCCESS;
  }

  REGISTER_NTT_INIT_DOMAIN_BACKEND("CUDA", ntt_cuda_init_domain)
  REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CUDA", ntt_cuda_release_domain);
  REGISTER_NTT_BACKEND("CUDA", (ntt_cuda<scalar_t, scalar_t>));
  REGISTER_NTT_GET_ROU_FROM_DOMAIN_BACKEND("CUDA", get_root_of_unity_from_domain_cuda<scalar_t>)
#ifdef EXT_FIELD
  REGISTER_NTT_EXT_FIELD_BACKEND("CUDA", (ntt_cuda<scalar_t, extension_t>));
#endif // EXT_FIELD

} // namespace icicle