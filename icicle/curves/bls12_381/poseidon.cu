#ifndef _BLS12_381_POSEIDON
#define _BLS12_381_POSEIDON
#include <cuda.h>
#include <stdexcept>
#include "../../appUtils/poseidon/poseidon.cu"
#include "curve_config.cuh"

template class Poseidon<BLS12_381::scalar_t>;

extern "C" int poseidon_multi_cuda_bls12_381(BLS12_381::scalar_t input[], BLS12_381::scalar_t* out,
                                             size_t number_of_blocks, int arity, size_t device_id = 0)
{
  try
  {
    Poseidon<BLS12_381::scalar_t> poseidon(arity);
    poseidon.hash_blocks(input, number_of_blocks, out, Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree);

    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}
#endif