---
slug: /
displayed_sidebar: GettingStartedSidebar
title: ''
---

# Welcome to Ingonyama's Developer Documentation

Ingonyama is a next-generation semiconductor company focusing on Zero-Knowledge Proof hardware acceleration. We build accelerators for advanced cryptography, unlocking real-time applications. Our focus is on democratizing access to compute-intensive cryptography and making it accessible for developers to build upon.

Our flagship product is **ICICLE**

#### **ICICLE v3**
[ICICLE v3](https://github.com/ingonyama-zk/icicle) is a versatile cryptography library designed to support multiple compute backends, including CUDA, CPU, and potentially others like Metal, WebGPU, Vulkan, and ZPU. Originally focused on GPU acceleration, ICICLE has evolved to offer backend-agnostic cryptographic acceleration, allowing you to build ZK provers or other cryptographic applications with ease, leveraging the best available hardware for your needs.

- **Multiple Backend Support:** Develop on CPU and deploy on various backends including CUDA and potentially Metal, WebGPU, Vulkan, ZPU, or even remote machines.
- **Cross-Language Compatibility:** Use ICICLE across multiple programming languages such as C++, Rust, Go, and possibly Python.
- **Optimized for ZKPs:** Accelerate cryptographic operations like elliptic curve operations, MSM, NTT, Poseidon hash, and more.

**Learn more about ICICLE and its multi-backend support [here][ICICLE-OVERVIEW].**

---

## Our Approach to Hardware Acceleration

We believe that GPUs are as critical for ZK as they are for AI.

- **Parallelism:** Approximately 97% of ZK protocol runtime is naturally parallel, making GPUs an ideal match.
- **Developer-Friendly:** GPUs offer simplicity in scaling and usage compared to other hardware platforms.
- **Cost-Effective:** GPUs provide a competitive balance of power, performance, and cost, often being 3x cheaper than FPGAs.

For a more in-depth understanding on this topic we suggest you read [our article on the subject](https://www.ingonyama.com/blog/revisiting-paradigm-hardware-acceleration-for-zero-knowledge-proofs).


## Get in Touch

If you have any questions, ideas, or are thinking of building something in this space, join the discussion on [Discord]. You can explore our code on [github](https://github.com/ingonyama-zk) or read some of [our research papers](https://github.com/ingonyama-zk/papers).

Follow us on [X (formerly Twitter)](https://x.com/Ingo_zk) and [YouTube](https://www.youtube.com/@ingo_ZK) and join us IRL at our [next event](https://www.ingonyama.com/events)

[ICICLE-OVERVIEW]: ./icicle/overview.md
[Discord]: https://discord.gg/6vYrE7waPj
