#ifndef _POSEIDON
#define _POSEIDON
#include <cuda.h>
#include <stdexcept>
#include "../appUtils/poseidon/poseidon.cu"
#include "curve_config.cuh"

template class Poseidon<scalar_t>;

extern "C" int poseidon_multi_cuda(scalar_t input[], scalar_t* out,
                                             size_t number_of_blocks, int arity, size_t device_id = 0)
{
  try
  {
    Poseidon<scalar_t> poseidon(arity);
    poseidon.hash_blocks(input, number_of_blocks, out, Poseidon<scalar_t>::HashType::MerkleTree);

    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}
#endif