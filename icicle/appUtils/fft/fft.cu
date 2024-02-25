#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>
#include <vector>

#include "curves/curve_config.cuh"
#include "utils/device_context.cuh"
#include "utils/vec_ops.cu"

namespace fft {
  template <typename S>
  cudaError_t fft(
    S* input, S* output, S* ws, int n, bool invert)
  {
    CHK_INIT_IF_RETURN();

    std::cout << STREAM_CHUNK_SIZE << std::endl;

    return CHK_LAST();
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, FftEvaluate)(
    curve_config::scalar_t* inout,
    curve_config::scalar_t* ws,
    int n)
  {

    return fft<curve_config::scalar_t>(inout, ws, n);
  }
}
