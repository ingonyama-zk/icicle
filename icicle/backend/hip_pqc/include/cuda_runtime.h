#pragma once
// Shim so CUDA-spelled translation units (#include <cuda_runtime.h>) resolve to
// the HIP-PQC compatibility layer under the HIP toolchain. See cuda_to_hip.h.
#include "cuda_to_hip.h"
