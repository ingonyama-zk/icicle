#pragma once
#ifndef __FFT_H
#define __FFT_H

#include <cstdint>
#include <iostream>

#include "curves/curve_config.cuh"
#include "utils/error_handler.cuh"
#include "utils/device_context.cuh"
#include "utils/utils.h"

namespace fft {
  static constexpr size_t STREAM_CHUNK_SIZE = 1024 * 1024 * 1024;
  template <typename S>
  cudaError_t fft(S* input, S* output, S* ws, int n, bool invert);
}

#endif
