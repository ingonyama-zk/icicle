# ICICLE HIP-PQC Backend (ROCm)

This is the **HIP-PQC backend** for ICICLE: a free and open-source ROCm/HIP backend for post-quantum cryptography (ML-KEM / Kyber-512/768/1024), registering the device `"HIP-PQC"`. It is an additive sibling of the `cuda_pqc` backend and brings AMD GPU acceleration to ICICLE's PQC path; the NVIDIA `"CUDA-PQC"` backend is untouched.

It is unrelated to ICICLE's main CUDA/Metal/Vulkan compute backends (MSM / NTT / EC / field arithmetic); those primitives live in the separate backend repositories and are not part of this backend.

## Build

From the repository root:

```
cmake -S icicle -B build -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
```

Set `-DCMAKE_HIP_ARCHITECTURES` to the target GPU (gfx90a, gfx1100, ...), or leave it unset to build for the GPUs CMake detects on the build host. The kernels are wave-agnostic (correct on 64-lane and 32-lane wavefronts).

## Design

The ML-KEM kernels are kept in the CUDA spelling and compiled as HIP via a small `cuda_to_hip.h` compatibility shim that aliases the `cuda*` runtime surface to `hip*` and provides wave-agnostic, 32-lane-subgroup variants of the warp-synchronous primitives (`__ballot_sync`, `__shfl_sync`, `__any_sync`, `__reduce_add_sync`). Each ML-KEM block is organized as four logical 32-thread warps; on a 64-lane wavefront a wavefront carries two logical warps, and the subgroup primitives confine every cross-lane operation to the caller's own logical warp so the Keccak/sampler algorithm runs identically to 32-wide hardware.

## Tests

The gtests under `tests/ml_kem/` validate keygen/encaps/decaps and the PKE/SHA3 building blocks against the NIST KAT vectors in `backend/cuda_pqc/tests/ml_kem/test_data` (exact match), which the build copies next to the tests. Run them from the repository root with `ctest --test-dir build/backend/hip_pqc/tests`.
