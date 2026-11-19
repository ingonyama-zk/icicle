#pragma once

// Compatibility shim for the additive HIP-PQC backend.
//
// This header lets the ML-KEM device/kernel sources -- which are written in the
// CUDA spelling and shared in spirit with backend/cuda_pqc -- compile under the
// HIP toolchain. It provides three things:
//   1. The small cuda* runtime surface the device API uses, aliased to hip*.
//   2. Shims for the streaming-cache load/store hints that HIP does not expose.
//   3. Wave-agnostic, 32-lane-subgroup variants of the warp-synchronous
//      primitives, so the warp-cooperative Keccak and CBD/NTT samplers stay
//      correct on a 64-lane wavefront (gfx90a) and a 32-lane one (gfx1100).
//
// The ML-KEM kernels are organized as four logical 32-thread warps inside a
// 128-thread block, selected by (threadIdx.x / 32) and (threadIdx.x % 32). On
// wave64 a single 64-lane wavefront therefore carries TWO logical warps. The
// warp-synchronous intrinsics here are confined to the caller's own logical
// 32-lane group so the algorithm runs once per logical warp exactly as on a
// 32-wide architecture; the cross-lane data path (shared memory keyed by the
// same indices) is already subgroup-local.

#include <hip/hip_runtime.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// 1. Runtime surface: cuda* -> hip*
// ----------------------------------------------------------------------------
using cudaError_t = hipError_t;
using cudaError = hipError_t;
using cudaStream_t = hipStream_t;
using cudaMemcpyKind = hipMemcpyKind;

#define cudaSuccess hipSuccess

#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaSetDevice         hipSetDevice
#define cudaGetDeviceCount    hipGetDeviceCount
#define cudaMalloc            hipMalloc
#define cudaMallocAsync       hipMallocAsync
#define cudaFree              hipFree
#define cudaFreeAsync         hipFreeAsync
#define cudaMemGetInfo        hipMemGetInfo
#define cudaMemset            hipMemset
#define cudaMemsetAsync       hipMemsetAsync
#define cudaMemcpy            hipMemcpy
#define cudaMemcpyAsync       hipMemcpyAsync
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreate      hipStreamCreate
#define cudaStreamDestroy     hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaGetErrorString    hipGetErrorString
#define cudaGetLastError      hipGetLastError

// ----------------------------------------------------------------------------
// 2. Streaming-cache load/store hint shims (no HIP equivalent; plain access).
// ----------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T __ldcs(const T* ptr)
{
  return __builtin_nontemporal_load(ptr);
}

template <typename T>
__device__ __forceinline__ void __stcs(T* ptr, T value)
{
  __builtin_nontemporal_store(value, ptr);
}

// __stcs is called as __stcs((uint16_t*)poly + j, coeffs_vec.x) where the value
// argument is a uint32_t (uint4 component); accept it without ambiguity.
__device__ __forceinline__ void __stcs(uint16_t* ptr, uint32_t value)
{
  __builtin_nontemporal_store((uint16_t)value, ptr);
}

// __ldlu is a last-use load hint on CUDA; map to the non-temporal load on HIP.
template <typename T>
__device__ __forceinline__ T __ldlu(const T* ptr)
{
  return __builtin_nontemporal_load(ptr);
}

// ----------------------------------------------------------------------------
// 3. Wave-agnostic 32-lane-subgroup primitives.
//
// Each logical warp is the contiguous 32-lane group (threadIdx.x % 32 == lane,
// threadIdx.x / 32 == logical warp index). On wave64 the wavefront holds two of
// these groups; width-32 shuffles and a per-group ballot/reduce keep every
// operation inside the caller's own group, matching the 32-wide CUDA semantics.
// ----------------------------------------------------------------------------
namespace icicle_hip_pqc {

  // Index of the calling lane within its own 32-lane logical group.
  __device__ __forceinline__ int sub32_lane() { return threadIdx.x & 31; }

  // Index of the logical 32-lane group within the physical wavefront (0 or 1 on
  // wave64, always 0 on wave32).
  __device__ __forceinline__ int sub32_group_in_wave() { return (int)((threadIdx.x & (warpSize - 1)) >> 5); }

  // 32-bit active mask for the caller's own 32-lane logical group.
  __device__ __forceinline__ uint32_t sub32_ballot(bool pred)
  {
    unsigned long long full = __ballot(pred);
    return (uint32_t)(full >> (32 * sub32_group_in_wave()));
  }

  // True if any lane in the caller's 32-lane group satisfies pred.
  __device__ __forceinline__ bool sub32_any(bool pred) { return sub32_ballot(pred) != 0u; }

  // Width-32 shuffle: src is the within-group lane index [0, 31].
  template <typename T>
  __device__ __forceinline__ T sub32_shfl(T val, int src)
  {
    return __shfl(val, src, 32);
  }

  // Butterfly add-reduction restricted to the caller's 32-lane group. Lanes
  // that did not contribute must pass 0; the result is broadcast to all lanes
  // of the group (matching __reduce_add_sync over the masked lanes when the
  // remaining lanes contribute 0).
  __device__ __forceinline__ uint32_t sub32_reduce_add(uint32_t v)
  {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor(v, off, 32);
    }
    return v;
  }

} // namespace icicle_hip_pqc

// Map the CUDA warp-sync spellings used by the ML-KEM samplers/hashes onto the
// subgroup-confined helpers. The first argument (the CUDA 32-bit lane mask) is
// ignored: confinement is by hardware lane group, not an explicit mask, so the
// reworked partial reductions (e.g. the 28-lane SHAKE coefficient counter) stay
// correct -- inactive lanes contribute 0 rather than relying on mask width.
#define __ballot_sync(mask, pred)    ::icicle_hip_pqc::sub32_ballot(pred)
#define __any_sync(mask, pred)       ::icicle_hip_pqc::sub32_any(pred)
#define __shfl_sync(mask, val, src)  ::icicle_hip_pqc::sub32_shfl((val), (src))
#define __reduce_add_sync(mask, val) ::icicle_hip_pqc::sub32_reduce_add((val))

// __syncwarp: the ML-KEM kernels use it to order cross-lane shared-memory
// traffic within a logical warp. HIP's native __syncwarp is a wavefront barrier
// with the right memory ordering; on wave64 it fences a superset of the 32-lane
// group, which is sound (stronger, never weaker). Leave it as HIP provides it.
