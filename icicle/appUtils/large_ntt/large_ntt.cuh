#pragma once
#ifndef _LARGE_NTT_H
#define _LARGE_NTT_H

#include <stdint.h>
#include "appUtils/ntt/ntt.cuh" // for enum Ordering

namespace ntt {
  class MixedRadixNTT
  {
  public:
    MixedRadixNTT(int ntt_size, bool is_inverse, Ordering ordering, cudaStream_t cuda_stream = cudaStreamPerThread);
    ~MixedRadixNTT();

    // disable copy
    MixedRadixNTT(const MixedRadixNTT&) = delete;
    MixedRadixNTT(MixedRadixNTT&&) = delete;
    MixedRadixNTT& operator=(const MixedRadixNTT&) = delete;
    MixedRadixNTT& operator=(MixedRadixNTT&&) = delete;

    template <typename E>
    cudaError_t operator()(E* d_input, E* d_output);

  private:
    cudaError_t init();
    cudaError_t generate_external_twiddles(curve_config::scalar_t basic_root);

    const int m_ntt_size;
    const int m_ntt_log_size;
    const bool m_is_inverse;
    const Ordering m_ordering;
    cudaStream_t m_cuda_stream;

    curve_config::scalar_t* m_gpuTwiddles = nullptr;
    curve_config::scalar_t* m_gpuIntTwiddles = nullptr;
    curve_config::scalar_t* m_gpuBasicTwiddles = nullptr;

    curve_config::scalar_t* m_w6_table = nullptr;
    curve_config::scalar_t* m_w12_table = nullptr;
    curve_config::scalar_t* m_w18_table = nullptr;
    curve_config::scalar_t* m_w24_table = nullptr;
    curve_config::scalar_t* m_w30_table = nullptr;
  };

} // namespace ntt
#endif //_LARGE_NTT_H