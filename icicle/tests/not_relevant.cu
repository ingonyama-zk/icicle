#include <vector>
#include <cupqc.hpp>
#include <cassert>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>   
using namespace cupqc;
    
#define DEBUG_KEY_GEN false // Disable debugging for key generation
    
using MLKEM768Key = decltype(ML_KEM_768()
                            + Function<function::Keygen>()
                            + Block()
                            + BlockDim<32>());  // Optional operator with default config
    
using MLKEM768Encaps = decltype(ML_KEM_768()
                               + Function<function::Encaps>()
                               + Block()
                               + BlockDim<32>());  // Optional operator with default config

using MLKEM768Decaps = decltype(ML_KEM_768()
                               + Function<function::Decaps>()
                               + Block()
                               + BlockDim<32>());  // Optional operator with default config

__global__ void keygen_kernel(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* workspace, uint8_t* randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM768Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLKEM768Key::public_key_size;
    auto secret_key = secret_keys + block * MLKEM768Key::secret_key_size;
    auto entropy    = randombytes + block * MLKEM768Key::entropy_size;
    auto work       = workspace   + block * MLKEM768Key::workspace_size;
    
    MLKEM768Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void encaps_kernel(uint8_t* ciphertexts, uint8_t* shared_secrets, const uint8_t* public_keys, uint8_t* workspace, uint8_t* randombytes)
{   
    __shared__ uint8_t smem_ptr[MLKEM768Encaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM768Encaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM768Encaps::ciphertext_size;
    auto public_key    = public_keys + block * MLKEM768Encaps::public_key_size;
    auto entropy       = randombytes + block * MLKEM768Encaps::entropy_size;
    auto work          = workspace   + block * MLKEM768Encaps::workspace_size;

    MLKEM768Encaps().execute(ciphertext, shared_secret, public_key, entropy, work, smem_ptr);
}
        
__global__ void decaps_kernel(uint8_t* shared_secrets, const uint8_t* ciphertexts, const uint8_t* secret_keys, uint8_t* workspace)
{       
    __shared__ uint8_t smem_ptr[MLKEM768Decaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM768Decaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM768Decaps::ciphertext_size;
    auto secret_key    = secret_keys + block * MLKEM768Decaps::secret_key_size;
    auto work          = workspace   + block * MLKEM768Decaps::workspace_size;

    MLKEM768Decaps().execute(shared_secret, ciphertext, secret_key, work, smem_ptr);
}

void benchmark(const std::string& operation_name, const cudaEvent_t& start, const cudaEvent_t& stop, unsigned int batch) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;
    double throughput = batch / seconds;
    printf("%s Throughput: ~%.2f ops/sec\n", operation_name.c_str(), throughput);
}

void ml_kem_keygen(std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    auto length_public_key = MLKEM768Key::public_key_size;
    auto length_secret_key = MLKEM768Key::secret_key_size;

    auto workspace   = make_workspace<MLKEM768Key>(batch);
    auto randombytes = get_entropy<MLKEM768Key>(batch);

    uint8_t* d_public_key = nullptr;
    uint8_t* d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    keygen_kernel<<<batch, MLKEM768Key::BlockDim>>>(d_public_key, d_secret_key, workspace, randombytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    benchmark("Key Generation", start, stop, batch);

    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ml_kem_encaps(std::vector<uint8_t>& ciphertexts, std::vector<uint8_t>& shared_secrets,
                   const std::vector<uint8_t>& public_keys, const unsigned int batch)
{
    auto length_ciphertext   = MLKEM768Encaps::ciphertext_size;
    auto length_sharedsecret = MLKEM768Encaps::shared_secret_size;
    auto length_public_key   = MLKEM768Encaps::public_key_size;

    auto workspace   = make_workspace<MLKEM768Encaps>(batch);
    auto randombytes = get_entropy<MLKEM768Encaps>(batch);

    uint8_t* d_ciphertext   = nullptr;
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_public_key   = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);

    cudaMemcpy(d_public_key, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    encaps_kernel<<<batch, MLKEM768Encaps::BlockDim>>>(d_ciphertext, d_sharedsecret, d_public_key, workspace, randombytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(ciphertexts.data(), d_ciphertext, length_ciphertext * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    benchmark("Encapsulation", start, stop, batch);

    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_public_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ml_kem_decaps(std::vector<uint8_t>& shared_secrets, const std::vector<uint8_t>& ciphertexts,
                   const std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    auto length_ciphertext   = MLKEM768Decaps::ciphertext_size;
    auto length_sharedsecret = MLKEM768Decaps::shared_secret_size;
    auto length_secret_key   = MLKEM768Decaps::secret_key_size;

    auto workspace   = make_workspace<MLKEM768Decaps>(batch);

    uint8_t* d_ciphertext   = nullptr;
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_secret_key   = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);

    cudaMemcpy(d_ciphertext, ciphertexts.data(), length_ciphertext * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secret_key, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    decaps_kernel<<<batch, MLKEM768Decaps::BlockDim>>>(d_sharedsecret, d_ciphertext, d_secret_key, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    benchmark("Decapsulation", start, stop, batch);

    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    for (unsigned int batch = 1 << 9; batch <= (1 << 22); batch <<= 1) {
        printf("\nBenchmarking with batch size: %u\n", batch);

        std::vector<uint8_t> public_keys(MLKEM768Key::public_key_size * batch);
        std::vector<uint8_t> secret_keys(MLKEM768Key::secret_key_size * batch);
        std::vector<uint8_t> ciphertexts(MLKEM768Encaps::ciphertext_size * batch);
        std::vector<uint8_t> encaps_shared_secrets(MLKEM768Encaps::shared_secret_size * batch);
        std::vector<uint8_t> decaps_shared_secrets(MLKEM768Decaps::shared_secret_size * batch);

        ml_kem_keygen(public_keys, secret_keys, batch);

        ml_kem_encaps(ciphertexts, encaps_shared_secrets, public_keys, batch);

        ml_kem_decaps(decaps_shared_secrets, ciphertexts, secret_keys, batch);

        printf("Benchmarking completed successfully for batch size %u\n", batch);
    }
}