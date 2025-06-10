#include "icicle/backend/pqc/ml_kem_backend.h"
#include "gpu-utils/utils.h"
#include "gpu-utils/error_handler.h"
#include "gpu-utils/error_translation.h"
#include "cuda_ml_kem_kernels.cuh"
#include "ml_kem/ring/cuda_zq.cuh"
#include "ml_kem/ring/cuda_poly.cuh"

namespace icicle::pqc::ml_kem {

  template <typename Category>
  static cudaError
  cuda_keygen(const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
    entropy =
      config.entropy_on_device
        ? entropy
        : allocate_and_copy_to_device(entropy, config.batch_size * ENTROPY_BYTES * sizeof(std::byte), cuda_stream);
    std::byte* d_public_keys = config.public_keys_on_device
                                 ? public_keys
                                 : allocate_on_device<std::byte>(
                                     config.batch_size * Category::PUBLIC_KEY_BYTES * sizeof(std::byte), cuda_stream);
    std::byte* d_secret_keys = config.secret_keys_on_device
                                 ? secret_keys
                                 : allocate_on_device<std::byte>(
                                     config.batch_size * Category::SECRET_KEY_BYTES * sizeof(std::byte), cuda_stream);

    Zq* d_A; // TODO: move to arguments? (send a buffer of bytes and cast to Zq*)
    CHK_IF_RETURN(cudaMallocAsync(
      &d_A, config.batch_size * PolyMatrix<256, Category::K, Category::K, Zq>::byte_size(), cuda_stream));

    ml_kem_keygen_kernel<Category::K, Category::ETA1><<<config.batch_size, 128, 0, cuda_stream>>>(
      (uint8_t*)entropy, (uint8_t*)d_public_keys, (uint8_t*)d_secret_keys, d_A);

    CHK_IF_RETURN(cudaFreeAsync((void*)d_A, cuda_stream));

    if (!config.public_keys_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(
        public_keys, d_public_keys, config.batch_size * Category::PUBLIC_KEY_BYTES * sizeof(std::byte),
        cudaMemcpyDeviceToHost, cuda_stream));
      CHK_IF_RETURN(cudaFreeAsync((void*)d_public_keys, cuda_stream));
    } else {
      secret_keys = d_secret_keys;
    }
    if (!config.secret_keys_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(
        secret_keys, d_secret_keys, config.batch_size * Category::SECRET_KEY_BYTES * sizeof(std::byte),
        cudaMemcpyDeviceToHost, cuda_stream));
      CHK_IF_RETURN(cudaFreeAsync((void*)d_secret_keys, cuda_stream));
    } else {
      public_keys = d_public_keys;
    }

    if (!config.entropy_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)entropy, cuda_stream)); }

    if (config.is_async) { CHK_IF_RETURN(cudaStreamSynchronize(cuda_stream)); }

    return CHK_LAST();
  }

  template <typename Category>
  static cudaError cuda_encapsulate(
    const std::byte* message,
    const std::byte* public_keys,
    MlKemConfig config,
    std::byte* ciphertext,
    std::byte* shared_secrets)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
    message =
      config.messages_on_device
        ? message
        : allocate_and_copy_to_device(message, config.batch_size * MESSAGE_BYTES * sizeof(std::byte), cuda_stream);
    public_keys = config.public_keys_on_device
                    ? public_keys
                    : allocate_and_copy_to_device(
                        public_keys, config.batch_size * Category::PUBLIC_KEY_BYTES * sizeof(std::byte), cuda_stream);
    std::byte* d_ciphertext = config.ciphertexts_on_device
                                ? ciphertext
                                : allocate_on_device<std::byte>(
                                    config.batch_size * Category::CIPHERTEXT_BYTES * sizeof(std::byte), cuda_stream);
    std::byte* d_shared_secrets =
      config.shared_secrets_on_device
        ? shared_secrets
        : allocate_on_device<std::byte>(
            config.batch_size * Category::SHARED_SECRET_BYTES * sizeof(std::byte), cuda_stream);

    Zq* d_A; // TODO: move to arguments? (send a buffer of bytes and cast to Zq*)
    CHK_IF_RETURN(cudaMallocAsync(
      &d_A, config.batch_size * PolyMatrix<256, Category::K, Category::K, Zq>::byte_size(), cuda_stream));

    ml_kem_encaps_kernel<Category::K, Category::ETA1, Category::ETA2, Category::DU, Category::DV>
      <<<config.batch_size, 128, 0, cuda_stream>>>(
        (uint8_t*)public_keys, (uint8_t*)message, (uint8_t*)d_shared_secrets, (uint8_t*)d_ciphertext, d_A);

    CHK_IF_RETURN(cudaFreeAsync((void*)d_A, cuda_stream));

    if (!config.ciphertexts_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(
        ciphertext, d_ciphertext, config.batch_size * Category::CIPHERTEXT_BYTES * sizeof(std::byte),
        cudaMemcpyDeviceToHost, cuda_stream));
      CHK_IF_RETURN(cudaFreeAsync((void*)d_ciphertext, cuda_stream));
    } else {
      ciphertext = d_ciphertext;
    }

    if (!config.shared_secrets_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(
        shared_secrets, d_shared_secrets, config.batch_size * Category::SHARED_SECRET_BYTES * sizeof(std::byte),
        cudaMemcpyDeviceToHost, cuda_stream));
      CHK_IF_RETURN(cudaFreeAsync((void*)d_shared_secrets, cuda_stream));
    } else {
      shared_secrets = d_shared_secrets;
    }

    if (!config.messages_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)message, cuda_stream)); }

    if (!config.public_keys_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)public_keys, cuda_stream)); }

    if (config.is_async) { CHK_IF_RETURN(cudaStreamSynchronize(cuda_stream)); }

    return CHK_LAST();
  }

  template <typename Category>
  static cudaError cuda_decapsulate(
    const std::byte* secret_keys, const std::byte* ciphertext, MlKemConfig config, std::byte* shared_secrets)
  {
    CHK_INIT_IF_RETURN();

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);
    secret_keys = config.secret_keys_on_device
                    ? secret_keys
                    : allocate_and_copy_to_device(
                        secret_keys, config.batch_size * Category::SECRET_KEY_BYTES * sizeof(std::byte), cuda_stream);
    ciphertext = config.ciphertexts_on_device
                   ? ciphertext
                   : allocate_and_copy_to_device(
                       ciphertext, config.batch_size * Category::CIPHERTEXT_BYTES * sizeof(std::byte), cuda_stream);
    std::byte* d_shared_secrets =
      config.shared_secrets_on_device
        ? shared_secrets
        : allocate_on_device<std::byte>(
            config.batch_size * Category::SHARED_SECRET_BYTES * sizeof(std::byte), cuda_stream);

    Zq* d_A; // TODO: move to arguments? (send a buffer of bytes and cast to Zq*)
    CHK_IF_RETURN(cudaMallocAsync(
      &d_A, config.batch_size * PolyMatrix<256, Category::K, Category::K, Zq>::byte_size(), cuda_stream));

    ml_kem_decaps_kernel<Category::K, Category::ETA1, Category::ETA2, Category::DU, Category::DV>
      <<<config.batch_size, 128, 0, cuda_stream>>>(
        (uint8_t*)secret_keys, (uint8_t*)ciphertext, (uint8_t*)d_shared_secrets, d_A);

    CHK_IF_RETURN(cudaFreeAsync((void*)d_A, cuda_stream));

    if (!config.shared_secrets_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(
        shared_secrets, d_shared_secrets, config.batch_size * Category::SHARED_SECRET_BYTES * sizeof(std::byte),
        cudaMemcpyDeviceToHost, cuda_stream));
      CHK_IF_RETURN(cudaFreeAsync((void*)d_shared_secrets, cuda_stream));
    } else {
      shared_secrets = d_shared_secrets;
    }

    if (!config.secret_keys_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)secret_keys, cuda_stream)); }

    if (!config.ciphertexts_on_device) { CHK_IF_RETURN(cudaFreeAsync((void*)ciphertext, cuda_stream)); }

    if (config.is_async) { CHK_IF_RETURN(cudaStreamSynchronize(cuda_stream)); }

    return CHK_LAST();
  }

  template <typename Params>
  static eIcicleError cuda_keygen_handler(
    const Device& device, const std::byte* entropy, MlKemConfig config, std::byte* public_keys, std::byte* secret_keys)
  {
    return translateCudaError(cuda_keygen<Params>(entropy, config, public_keys, secret_keys));
  }

  template <typename Params>
  static eIcicleError cuda_encapsulate_handler(
    const Device& device,
    const std::byte* message,
    const std::byte* public_keys,
    MlKemConfig config,
    std::byte* ciphertext,
    std::byte* shared_secrets)
  {
    return translateCudaError(cuda_encapsulate<Params>(message, public_keys, config, ciphertext, shared_secrets));
  }

  template <typename Params>
  static eIcicleError cuda_decapsulate_handler(
    const Device& device,
    const std::byte* secret_keys,
    const std::byte* ciphertext,
    MlKemConfig config,
    std::byte* shared_secrets)
  {
    return translateCudaError(cuda_decapsulate<Params>(secret_keys, ciphertext, config, shared_secrets));
  }

  // REGISTER_ML_KEM_BACKEND("CUDA-PQC", Kyber1024Params, cuda_keygen_handler, cuda_encapsulate_handler,
  // cuda_decapsulate_handler); REGISTER_ML_KEM_BACKEND("CUDA-PQC", Kyber768Params, cuda_keygen_handler,
  // cuda_encapsulate_handler, cuda_decapsulate_handler);
  REGISTER_ML_KEM512_BACKEND("CUDA-PQC", cuda_keygen_handler, cuda_encapsulate_handler, cuda_decapsulate_handler);
  REGISTER_ML_KEM768_BACKEND("CUDA-PQC", cuda_keygen_handler, cuda_encapsulate_handler, cuda_decapsulate_handler);
  REGISTER_ML_KEM1024_BACKEND("CUDA-PQC", cuda_keygen_handler, cuda_encapsulate_handler, cuda_decapsulate_handler);
} // namespace icicle::pqc::ml_kem
