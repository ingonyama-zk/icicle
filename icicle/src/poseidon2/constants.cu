#include "poseidon2/poseidon2.cuh"

/// These are pre-calculated constants for different curves
#include "fields/id.h"
#if FIELD_ID == BN254
#include "poseidon2/constants/bn254_poseidon2.h"
using namespace poseidon2_constants_bn254;
#elif FIELD_ID == BLS12_381
#include "poseidon2/constants/bls12_381_poseidon2.h"
using namespace poseidon2_constants_bls12_381;
#elif FIELD_ID == BLS12_377
#include "poseidon2/constants/bls12_377_poseidon2.h"
using namespace poseidon2_constants_bls12_377;
#elif FIELD_ID == BW6_761
#include "poseidon2/constants/bw6_761_poseidon2.h"
using namespace poseidon2_constants_bw6_761;
#elif FIELD_ID == GRUMPKIN
#include "poseidon2/constants/grumpkin_poseidon2.h"
using namespace poseidon2_constants_grumpkin;
#elif FIELD_ID == BABY_BEAR
#include "poseidon2/constants/babybear_poseidon2.h"
using namespace poseidon2_constants_grumpkin;
#endif

namespace poseidon2 {
  template <typename S>
  cudaError_t create_optimized_poseidon2_constants(
    int width,
    int alpha,
    int internal_rounds,
    int external_rounds,
    const S* round_constants,
    const S* internal_matrix_diag,
    device_context::DeviceContext& ctx,
    Poseidon2Constants<S>* poseidon_constants)
  {
    if (!(alpha == 3 || alpha == 5 || alpha == 7 || alpha == 11)) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Invalid alpha value");
    }
    if (external_rounds % 2) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Invalid external rounds");
    }

    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = ctx.stream;

    int round_constants_len = width * external_rounds + internal_rounds;
    int internal_matrix_len = width;

    // Malloc memory for copying round constants and internal matrix
    S* d_constants;
    CHK_IF_RETURN(cudaMallocAsync(&d_constants, sizeof(S) * (internal_matrix_len + round_constants_len), stream));

    S* d_internal_matrix = d_constants;
    S* d_round_constants = d_constants + internal_matrix_len;

    // Copy internal matrix
    CHK_IF_RETURN(cudaMemcpyAsync(d_internal_matrix, internal_matrix_diag, sizeof(S) * internal_matrix_len, cudaMemcpyHostToDevice, stream));
    // Copy round constants
    CHK_IF_RETURN(cudaMemcpyAsync(d_round_constants, round_constants, sizeof(S) * round_constants_len, cudaMemcpyHostToDevice, stream));

    // Make sure all the constants have been copied
    CHK_IF_RETURN(cudaStreamSynchronize(stream));
    *poseidon_constants = {
      width,
      alpha,
      internal_rounds,
      external_rounds,
      d_round_constants,
      d_internal_matrix,
    };

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t init_optimized_poseidon2_constants(
    int width, device_context::DeviceContext& ctx, Poseidon2Constants<S>* poseidon2_constants)
  {
    CHK_INIT_IF_RETURN();

#define P2_CONSTANTS_DEF(width) \
case width:\
  internal_rounds = t##width::internal_rounds;\
  round_constants = t##width::round_constants;\
  internal_matrix = t##width::mat_diag_m_1;\
  alpha = t##width::alpha;\
  break;

    int alpha;
    int external_rounds = EXTERNAL_ROUNDS_DEFAULT;
    int internal_rounds;
    unsigned char* round_constants;
    unsigned char* internal_matrix;
    switch (width) {
      P2_CONSTANTS_DEF(2)
      P2_CONSTANTS_DEF(3)
      P2_CONSTANTS_DEF(4)
      P2_CONSTANTS_DEF(8)
      P2_CONSTANTS_DEF(12)
      P2_CONSTANTS_DEF(16)
      P2_CONSTANTS_DEF(20)
      P2_CONSTANTS_DEF(24)
      default:
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "init_optimized_poseidon2_constants: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
    }
    S* h_round_constants = reinterpret_cast<S*>(round_constants);
    S* h_internal_matrix = reinterpret_cast<S*>(internal_matrix);

    create_optimized_poseidon2_constants(width, alpha, internal_rounds, external_rounds, h_round_constants, h_internal_matrix, ctx, poseidon2_constants);

    return CHK_LAST();
  }
} // namespace poseidon2