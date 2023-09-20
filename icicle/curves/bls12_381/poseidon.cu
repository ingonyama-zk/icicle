#ifndef _BLS12_381_POSEIDON
#define _BLS12_381_POSEIDON
#include "../../appUtils/poseidon/poseidon.cu"
#include "curve_config.cuh"
#include <cuda.h>
#include <stdexcept>

template class Poseidon<BLS12_381::scalar_t>;

extern "C" int poseidon_multi_cuda_bls12_381(
  BLS12_381::scalar_t input[],
  BLS12_381::scalar_t* out,
  size_t number_of_blocks,
  int arity,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    if (stream == 0) { cudaStreamCreate(&stream); }

    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventRecord(start_event, stream);
    Poseidon<BLS12_381::scalar_t> poseidon(arity, stream);
    poseidon.hash_blocks(input, number_of_blocks, out, Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree, stream);
    cudaEventRecord(end_event, stream);
    cudaEventSynchronize(end_event);

#ifdef DEBUG
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_event, end_event);
    printf("Time elapsed: %f", elapsedTime);
#endif

    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);

    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}
#endif