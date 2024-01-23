#pragma once
#ifndef _LARGE_NTT_H
#define _LARGE_NTT_H

#include <stdint.h>

namespace ntt {
  class MixedRadixNTT
  {
  public:
    MixedRadixNTT(int ntt_size, bool is_inverse, bool is_dit, cudaStream_t cuda_stream = cudaStreamPerThread);
    ~MixedRadixNTT();

    // disable copy
    MixedRadixNTT(const MixedRadixNTT&) = delete;
    MixedRadixNTT(MixedRadixNTT&&) = delete;
    MixedRadixNTT& operator=(const MixedRadixNTT&) = delete;
    MixedRadixNTT& operator=(MixedRadixNTT&&) = delete;

    template <typename E>
    cudaError_t operator()(E* d_input, E* d_output);

  private:
    void generate_external_twiddles(curve_config::scalar_t basic_root);

    const int m_ntt_size;
    const int m_ntt_log_size;
    const bool m_is_inverse;
    const bool m_is_dit;
    cudaStream_t m_cuda_stream;

    uint4* m_gpuTwiddles = nullptr;
    uint4* m_gpuIntTwiddles = nullptr;
    uint4* m_gpuBasicTwiddles = nullptr;

    uint4* m_w6_table = nullptr;
    uint4* m_w12_table = nullptr;
    uint4* m_w18_table = nullptr;
    uint4* m_w24_table = nullptr;
    uint4* m_w30_table = nullptr;
  };

} // namespace ntt
#endif //_LARGE_NTT_H