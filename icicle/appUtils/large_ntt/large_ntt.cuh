#pragma once
#ifndef _LARGE_NTT_H
#define _LARGE_NTT_H

#include <stdint.h>
#include "appUtils/ntt/ntt.cuh" // for enum Ordering

namespace ntt {

  template <typename E, typename S>
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

    cudaError_t operator()(E* d_input, E* d_output);

  private:
    cudaError_t init();
    cudaError_t generate_external_twiddles(S basic_root);

    const int m_ntt_size;
    const int m_ntt_log_size;
    const bool m_is_inverse;
    const Ordering m_ordering;
    cudaStream_t m_cuda_stream;

    static inline S* m_gpuTwiddles = nullptr;
    static inline S* m_gpuIntTwiddles = nullptr;
    static inline S* m_gpuBasicTwiddles = nullptr;
    static inline S* m_w6_table = nullptr;
    static inline S* m_w12_table = nullptr;
    static inline S* m_w18_table = nullptr;
    static inline S* m_w24_table = nullptr;
    static inline S* m_w30_table = nullptr;
  };

} // namespace ntt
#endif //_LARGE_NTT_H