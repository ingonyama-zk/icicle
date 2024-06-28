#include "utils/utils.h"

#include "gpu-utils/error_handler.cuh"
#include "merkle-tree/merkle.cuh"
#include "matrix/matrix.cuh"
#include "mmcs.cu"

#include "hash/hash.cuh"

#include "fields/field_config.cuh"
using namespace field_config;

using matrix::Matrix;

namespace merkle_tree {
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, mmcs_commit_cuda)(
    const Matrix<scalar_t>* leaves,
    unsigned int number_of_inputs,
    scalar_t* digests,
    const hash::SpongeHasher<scalar_t, scalar_t>* hasher,
    const hash::SpongeHasher<scalar_t, scalar_t>* compression,
    const TreeBuilderConfig& tree_config)
  {
    return mmcs_commit<scalar_t, scalar_t>(leaves, number_of_inputs, digests, *hasher, *compression, tree_config);
  }
} // namespace merkle_tree