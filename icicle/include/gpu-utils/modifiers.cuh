#if defined(DEVMODE) || defined(DEBUG)
#define INLINE_MACRO
#define UNROLL
#else
#define INLINE_MACRO __forceinline__
#define UNROLL       #pragma unroll
#endif

#define        __host__ INLINE_MACRO
#define      __device__ INLINE_MACRO
#define __host__ __device__ INLINE_MACRO
