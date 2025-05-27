# ICICLE CUDA-PQC Backend

This is the **CUDA-PQC backend** for ICICLE — a free and open-source CUDA backend designed **specifically for post-quantum cryptography (PQC)**. It is entirely separate from the closed-source CUDA backend and supports a different, orthogonal set of ICICLE features.

## ⚙️ Overview

- **CUDA-PQC** is built independently from the closed CUDA backend.
- It is **open-source**, **license-free**, and intended only for PQC functionality.
- It does **not** implement general ICICLE APIs like hash or Merkle tree — only PQC APIs are supported.
- The backend is compiled with `nvcc` and **linked statically** into ICICLE; there's no need to load it explicitly.

## 🔄 Comparison with Closed CUDA Backend

| Feature                 | CUDA-PQC Backend        | Closed CUDA Backend        |
|------------------------|-------------------------|----------------------------|
| License required       | ❌ No                   | ✅ Yes (license checked)   |
| Source available       | ✅ Open-source          | ❌ Closed-source           |
| General ICICLE APIs    | ❌ Not supported        | ✅ Fully supported         |
| PQC APIs               | ✅ Supported            | ✅ Supported               |
| Registration mechanism | ✅ Statically linked    | ✅ Dynamically loaded      |

These backends are **separate**. If both are compiled, they coexist as distinct devices.

## 🛠️ Building

To build ICICLE with the CUDA-PQC backend:

```bash
cmake -DCUDA_PQC_BACKEND=ON ...
```

- This assumes `nvcc` is available in your environment.
- The CUDA-PQC backend will be **compiled and statically linked** into the ICICLE libraries.
- No runtime loading or additional configuration is needed.

If you also have access to the closed CUDA backend source, you can build both:

```bash
cmake -DCUDA_PQC_BACKEND=ON -DCUDA_BACKEND=<branch|commit|local> ...
```

This results in **three available devices**:

1. `CPU`
2. `CUDA`
3. `CUDA-PQC`

## 🦀 Rust Usage

In Rust, enable the feature:

```bash
cargo build --features=cuda_pqc
```

Then use it in your code:

```rust
icicle::set_device("CUDA-PQC"); // Select the PQC-only CUDA backend
```

- Now you can call the ICICLE PQC APIs (e.g., keygen, encaps, decaps).
- Note: **Other APIs will fail** (e.g., hash, Merkle), since this backend does not implement them.
- To use the full CUDA backend, switch to:

```rust
icicle::set_device("CUDA");
```

Think of them as **two orthogonal, mutually exclusive backends**, each supporting a different feature set.

## 🧪 Testing Strategy

- Can test **each backend independently**, or build both and test them together.
- ICICLE will recognize all available devices (`CUDA`, `CUDA-PQC`, `CPU`) and route API tests accordingly.

## 📎 Summary

- ✅ Free and open-source
- 🔒 No license required
- 🚀 Focused on PQC
- 🔧 Statically linked, no runtime loading
- 🧬 Compatible with ICICLE's multi-device architecture
