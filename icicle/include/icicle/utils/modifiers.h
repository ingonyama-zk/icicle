
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
#define HOST_DEVICE_INLINE __host__ __device__ INLINE_MACRO
#else // not CUDA
#define INLINE_MACRO
#define UNROLL
#define HOST_INLINE
#define DEVICE_INLINE
#define HOST_DEVICE_INLINE
#define __host__
#define __device__
#endif


//---BLAKE2S Modifiers---//
#if defined(_MSC_VER)
#define BLAKE2_PACKED(x) __pragma(pack(push, 1)) x __pragma(pack(pop))
#else
#define BLAKE2_PACKED(x) x __attribute__((packed))
#endif

#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L)
#if defined(_MSC_VER)
#define BLAKE2_INLINE __inline
#elif defined(__GNUC__)
#define BLAKE2_INLINE __inline__
#else
#define BLAKE2_INLINE
#endif
#else
#define BLAKE2_INLINE inline
#endif


//---End of BLAKE2S Modifiers---//