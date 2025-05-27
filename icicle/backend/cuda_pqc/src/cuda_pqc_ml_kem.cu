#include "icicle/backend/pqc/ml_kem_backend.h"
#include "cuda_ml_kem_device_code.cuh"
namespace icicle {
  namespace pqc {
    namespace ml_kem {

      static eIcicleError cuda_keygen(
        const Device& device,
        SecurityCategory category,
        const std::byte* entropy,
        MlKemConfig config,
        std::byte* public_keys,
        std::byte* secret_keys)
      {
        ml_kem_keygen_kernel_stub<<<1, 32>>>();
        return eIcicleError::API_NOT_IMPLEMENTED;
      }

      static eIcicleError cuda_encapsulate(
        const Device& device,
        SecurityCategory category,
        const std::byte* public_keys,
        MlKemConfig config,
        std::byte* ciphertext,
        std::byte* shared_secrets)
      {
        return eIcicleError::API_NOT_IMPLEMENTED;
      }

      static eIcicleError cuda_decapsulate(
        const Device& device,
        SecurityCategory category,
        const std::byte* secret_keys,
        const std::byte* ciphertext,
        MlKemConfig config,
        std::byte* shared_secrets)
      {
        return eIcicleError::API_NOT_IMPLEMENTED;
      }

      REGISTER_ML_KEM_BACKEND("CUDA-PQC", cuda_keygen, cuda_encapsulate, cuda_decapsulate)
    } // namespace ml_kem
  } // namespace pqc
} // namespace icicle