---
slug: /
displayed_sidebar: GettingStartedSidebar
title: ''
---

# Welcome to Ingonyama's Developer Documentation

Ingonyama is a next-generation semiconductor company building hardware accelerators for high-speed cryptography.
We design custom architectures that enable real-time performance for advanced cryptographic workloads. Our mission is to democratize access to compute-intensive cryptography, making it easy and accessible for developers to build the future of secure applications.

Our flagship product is **ICICLE**

#### **ICICLE v3**
[ICICLE v3](https://github.com/ingonyama-zk/icicle) ICICLE is a versatile cryptography library supporting multiple compute backends—including CUDA, CPU, Metal, and upcoming backends like WebGPU, Vulkan, and ZPU. Originally focused on GPU acceleration, ICICLE has evolved into a backend-agnostic framework for cryptographic acceleration. It enables you to build ZK provers and other cryptographic applications with ease, leveraging the best available hardware for your needs.

- **Multiple Backend Support:** Develop on CPU and deploy on various backends including CUDA, Metal, and eventually WebGPU, Vulkan, ZPU, or even remote machines.
- **Cross-Language Compatibility:** Use ICICLE across multiple programming languages such as C++, Rust, Go, and possibly Python.
- **Optimized for ZKPs:** Accelerate cryptographic operations like elliptic curve operations, MSM, NTT, Poseidon hash, and more.

**Learn more about ICICLE and its multi-backend support [here][ICICLE-OVERVIEW].**

---

## Our Approach to Hardware Acceleration

We believe GPUs are as essential for ZK as they are for AI.

- **Parallelism:** Around 97% of ZK protocol runtime is naturally parallel, perfect for GPU architectures.
- **Developer-Friendly:** GPUs offer simpler scaling and tooling compared to other hardware platforms.
- **Cost-Effective:** GPUs strike an ideal balance of performance and price—often 3× cheaper than FPGAs.

For a more in-depth understanding on this topic we suggest you read [our article on the subject](https://www.ingonyama.com/blog/revisiting-paradigm-hardware-acceleration-for-zero-knowledge-proofs).


## Get in Touch

If you have any questions, ideas, or are thinking of building something in this space, join the discussion on [Discord]. You can explore our code on [github](https://github.com/ingonyama-zk) or read some of [our research papers](https://github.com/ingonyama-zk/papers).

Follow us on [Twitter](https://x.com/Ingo_zk) and [YouTube](https://www.youtube.com/@ingo_ZK), or join us IRL at our [next event](https://www.ingonyama.com/events).

[ICICLE-OVERVIEW]: ./icicle/overview.md
[Discord]: https://discord.gg/6vYrE7waPj
