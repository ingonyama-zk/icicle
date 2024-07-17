
#include "icicle/polynomials/polynomials.h"
#include "icicle/polynomials/default_backend/default_poly_context.h"
#include "icicle/polynomials/default_backend/default_poly_backend.h"

#include "gpu-utils/error_handler.h"
#include "cuda_runtime.h"

namespace polynomials {

  using icicle::DefaultPolynomialBackend;
  using icicle::DefaultPolynomialContext;

  /*============================== Polynomial CUDA-factory ==============================*/

  template <typename C = scalar_t, typename D = C, typename I = C>
  class CUDAPolynomialFactory : public AbstractPolynomialFactory<C, D, I>
  {
    std::vector<cudaStream_t> m_device_streams; // device-id --> device stream

  public:
    CUDAPolynomialFactory();
    ~CUDAPolynomialFactory();
    std::shared_ptr<IPolynomialContext<C, D, I>> create_context() override;
    std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() override;
  };

  template <typename C, typename D, typename I>
  CUDAPolynomialFactory<C, D, I>::CUDAPolynomialFactory()
  {
    int nof_cuda_devices = -1;
    CHK_STICKY(cudaGetDeviceCount(&nof_cuda_devices));
    int orig_device = -1;

    CHK_STICKY(cudaGetDevice(&orig_device));
    m_device_streams.resize(nof_cuda_devices, nullptr);

    for (int dev_id = 0; dev_id < nof_cuda_devices; ++dev_id) {
      CHK_STICKY(cudaSetDevice(dev_id));
      CHK_STICKY(cudaStreamCreate(&m_device_streams[dev_id]));
    }
    CHK_STICKY(cudaSetDevice(orig_device)); // setting back original device
  }

  template <typename C, typename D, typename I>
  CUDAPolynomialFactory<C, D, I>::~CUDAPolynomialFactory()
  {
    // Note: Should release the streams but since this executes when the process terminates, there may be a race with
    // cuda driver shutting down, this cuda APIs would fail at this point.
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialContext<C, D, I>> CUDAPolynomialFactory<C, D, I>::create_context()
  {
    int cuda_device_id = -1;
    CHK_STICKY(cudaGetDevice(&cuda_device_id));
    return std::make_shared<DefaultPolynomialContext<C, D, I>>(m_device_streams[cuda_device_id]);
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialBackend<C, D, I>> CUDAPolynomialFactory<C, D, I>::create_backend()
  {
    int cuda_device_id = -1;
    CHK_STICKY(cudaGetDevice(&cuda_device_id));
    return std::make_shared<DefaultPolynomialBackend<C, D, I>>(m_device_streams[cuda_device_id]);
  }

  /************************************** BACKEND REGISTRATION **************************************/

  REGISTER_SCALAR_POLYNOMIAL_FACTORY_BACKEND("CUDA", CUDAPolynomialFactory<scalar_t>)

} // namespace polynomials