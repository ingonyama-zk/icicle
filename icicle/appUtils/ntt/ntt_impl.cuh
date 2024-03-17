#pragma once
#ifndef _NTT_IMPL_H
#define _NTT_IMPL_H

#include <stdint.h>
#include "appUtils/ntt/ntt.cuh" // for enum Ordering

namespace mxntt {

  template <typename S>
  cudaError_t generate_external_twiddles_generic(
    const S& basic_root,
    S* external_twiddles,
    S*& internal_twiddles,
    S*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);

  template <typename S>
  cudaError_t generate_external_twiddles_fast_twiddles_mode(
    const S& basic_root,
    S* external_twiddles,
    S*& internal_twiddles,
    S*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);

  template <typename E, typename S>
  cudaError_t mixed_radix_ntt(
    E* d_input,
    E* d_output,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    int ntt_size,
    int max_logn,
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    ntt::Ordering ordering,
    S* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream);

} // namespace ntt
#endif //_NTT_IMPL_H