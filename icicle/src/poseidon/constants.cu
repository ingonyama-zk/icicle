#include "poseidon/constants.cuh"
#include "gpu-utils/device_context.cuh"

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
    unsigned int arity,
    unsigned int alpha,
    unsigned int partial_rounds,
    unsigned int full_rounds_half,
    const S* round_constants,
    const S* mds_matrix,
    const S* non_sparse_matrix,
    const S* sparse_matrices,
    const S domain_tag,
    PoseidonConstants<S>* poseidon_constants,
    device_context::DeviceContext& ctx)
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

    S* d_round_constants = d_constants;
    S* d_mds_matrix = d_round_constants + round_constants_len;
    S* d_non_sparse_matrix = d_mds_matrix + mds_matrix_len;
    S* d_sparse_matrices = d_non_sparse_matrix + mds_matrix_len;

    // Copy constants
    CHK_IF_RETURN(cudaMemcpyAsync(
      d_round_constants, round_constants, sizeof(S) * round_constants_len, cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(
      cudaMemcpyAsync(d_mds_matrix, mds_matrix, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMemcpyAsync(
      d_non_sparse_matrix, non_sparse_matrix, sizeof(S) * mds_matrix_len, cudaMemcpyHostToDevice, stream));
    CHK_IF_RETURN(cudaMemcpyAsync(
      d_sparse_matrices, sparse_matrices, sizeof(S) * sparse_matrices_len, cudaMemcpyHostToDevice, stream));

    // Make sure all the constants have been copied
    CHK_IF_RETURN(cudaStreamSynchronize(stream));
    *poseidon_constants = {
      arity,
      alpha,
      partial_rounds,
      full_rounds_half,
      d_round_constants,
      d_mds_matrix,
      d_non_sparse_matrix,
      d_sparse_matrices,
      domain_tag};

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t init_optimized_poseidon_constants(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<S>* poseidon_constants)
  {
    CHK_INIT_IF_RETURN();
    unsigned int full_rounds_half = FULL_ROUNDS_DEFAULT;
    unsigned int partial_rounds;
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

    unsigned int width = arity + 1;
    unsigned int round_constants_len = width * full_rounds_half * 2 + partial_rounds;
    unsigned int mds_matrix_len = width * width;

    S* round_constants = h_constants;
    S* mds_matrix = round_constants + round_constants_len;
    S* non_sparse_matrix = mds_matrix + mds_matrix_len;
    S* sparse_matrices = non_sparse_matrix + mds_matrix_len;

    // Pick the domain_tag accordinaly
    // For now, we only support Merkle tree mode
    uint32_t tree_domain_tag_value = 1;
    tree_domain_tag_value = (tree_domain_tag_value << (width - 1)) - tree_domain_tag_value;
    S domain_tag = S::from(tree_domain_tag_value);

    create_optimized_poseidon_constants<S>(
      arity, 5, partial_rounds, full_rounds_half, round_constants, mds_matrix, non_sparse_matrix, sparse_matrices,
      domain_tag, poseidon_constants, ctx);

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t release_optimized_poseidon_constants(PoseidonConstants<S>* constants, device_context::DeviceContext& ctx)
  {
    CHK_INIT_IF_RETURN();
    CHK_IF_RETURN(cudaFreeAsync(constants->round_constants, ctx.stream));

    constants->arity = 0;
    constants->partial_rounds = 0;
    constants->full_rounds_half = 0;
    constants->round_constants = nullptr;
    constants->mds_matrix = nullptr;
    constants->non_sparse_matrix = nullptr;
    constants->sparse_matrices = nullptr;
    return CHK_LAST();
  }
} // namespace poseidon