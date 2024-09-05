
// Note: currently icicle frontend is sharing headers with cuda backend so need this hack. TODO Yuval: decouple.
#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#if defined(DEVMODE) || defined(DEBUG)
#define INLINE_MACRO
#define UNROLL
#else
#define INLINE_MACRO __forceinline__
#define UNROLL       #pragma unroll
#endif

#define HOST_INLINE        __host__ INLINE_MACRO
#define DEVICE_INLINE      __device__ INLINE_MACRO
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE INLINE_MACRO
#else // not CUDA
#define INLINE_MACRO
#define UNROLL
#define HOST_INLINE
#define DEVICE_INLINE
#define HOST_DEVICE
#define HOST_DEVICE_INLINE
#define __host__
#define __device__
#endif
