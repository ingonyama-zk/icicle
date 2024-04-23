#include "poseidon/poseidon.cuh"

/// These are pre-calculated constants for different curves
#include "fields/id.h"
#if FIELD_ID == BN254
#include "poseidon/constants/bn254_poseidon.h"
using namespace poseidon_constants_bn254;
#elif FIELD_ID == BLS12_381
#include "poseidon/constants/bls12_381_poseidon.h"
using namespace poseidon_constants_bls12_381;
#elif FIELD_ID == BLS12_377
#include "poseidon/constants/bls12_377_poseidon.h"
using namespace poseidon_constants_bls12_377;
#elif FIELD_ID == BW6_761
#include "poseidon/constants/bw6_761_poseidon.h"
using namespace poseidon_constants_bw6_761;
#elif FIELD_ID == GRUMPKIN
#include "poseidon/constants/grumpkin_poseidon.h"
using namespace poseidon_constants_grumpkin;
#endif

namespace poseidon {
  template <typename S>
  cudaError_t create_optimized_poseidon_constants(
    int arity,
    int full_rounds_half,
    int partial_rounds,
    const S* constants,
    device_context::DeviceContext& ctx,
    PoseidonConstants<S>* poseidon_constants)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = ctx.stream;
    int width = arity + 1;
    int round_constants_len = width * full_rounds_half * 2 + partial_rounds;
    int mds_matrix_len = width * width;
    int sparse_matrices_len = (width * 2 - 1) * partial_rounds;
    int constants_len = round_constants_len + mds_matrix_len * 2 + sparse_matrices_len;

    // Malloc memory for copying constants
    S* d_constants;
    CHK_IF_RETURN(cudaMallocAsync(&d_constants, sizeof(S) * constants_len, stream));

    // Copy constants
    CHK_IF_RETURN(cudaMemcpyAsync(d_constants, constants, sizeof(S) * constants_len, cudaMemcpyHostToDevice, stream));

    S* round_constants = d_constants;
    S* mds_matrix = round_constants + round_constants_len;
    S* non_sparse_matrix = mds_matrix + mds_matrix_len;
    S* sparse_matrices = non_sparse_matrix + mds_matrix_len;

    // Pick the domain_tag accordinaly
    // For now, we only support Merkle tree mode
    uint32_t tree_domain_tag_value = 1;
    tree_domain_tag_value = (tree_domain_tag_value << (width - 1)) - tree_domain_tag_value;
    S domain_tag = S::from(tree_domain_tag_value);

    // Make sure all the constants have been copied
    CHK_IF_RETURN(cudaStreamSynchronize(stream));
    *poseidon_constants = {arity,      partial_rounds,    full_rounds_half, round_constants,
                           mds_matrix, non_sparse_matrix, sparse_matrices,  domain_tag};

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t init_optimized_poseidon_constants(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<S>* poseidon_constants)
  {
    CHK_INIT_IF_RETURN();
    int full_rounds_half = FULL_ROUNDS_DEFAULT;
    int partial_rounds;
    unsigned char* constants;
    switch (arity) {
    case 2:
      constants = poseidon_constants_2;
      partial_rounds = partial_rounds_2;
      break;
    case 4:
      constants = poseidon_constants_4;
      partial_rounds = partial_rounds_4;
      break;
    case 8:
      constants = poseidon_constants_8;
      partial_rounds = partial_rounds_8;
      break;
    case 11:
      constants = poseidon_constants_11;
      partial_rounds = partial_rounds_11;
      break;
    default:
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument, "init_optimized_poseidon_constants: #arity must be one of [2, 4, 8, 11]");
    }
    S* h_constants = reinterpret_cast<S*>(constants);

    create_optimized_poseidon_constants(arity, full_rounds_half, partial_rounds, h_constants, ctx, poseidon_constants);

    return CHK_LAST();
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, create_optimized_poseidon_constants_cuda)(
    int arity,
    int full_rounds_half,
    int partial_rounds,
    const scalar_t* constants,
    device_context::DeviceContext& ctx,
    PoseidonConstants<scalar_t>* poseidon_constants)
  {
    return create_optimized_poseidon_constants<scalar_t>(
      arity, full_rounds_half, partial_rounds, constants, ctx, poseidon_constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, init_optimized_poseidon_constants_cuda)(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<scalar_t>* constants)
  {
    return init_optimized_poseidon_constants<scalar_t>(arity, ctx, constants);
  }
} // namespace poseidon