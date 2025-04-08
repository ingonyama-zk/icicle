---
slug: /icicle/overview
title: ICICLE Overview
---

# ICICLE Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/02080cc6-9761-4764-8ae4-05778cc93cfa" alt="Untitled design (29)" width="500"/>
</p>

## What is ICICLE?

[![GitHub Release](https://img.shields.io/github/v/release/ingonyama-zk/icicle)](https://github.com/ingonyama-zk/icicle/releases)

[ICICLE](https://github.com/ingonyama-zk/icicle) is a math library designed to accelerate high-speed cryptography—starting with zero-knowledge proofs (ZKPs)—across multiple compute backends, including GPUs, CPUs, Apple Silicon, and more. Its key strength lies in ultra-fast implementations of cryptographic primitives, enabling developers to significantly reduce proving times with minimal effort.

## Key Features

- **High-Speed Cryptographic Math:** ICICLE delivers optimized performance for core primitives like elliptic curve operations, MSM, NTT, and Poseidon hash—essential building blocks in modern cryptographic protocols.
- **Set of Libraries:** ICICLE includes a comprehensive set of libraries supporting various fields, curves, and other cryptographic needs.
- **Cross-Language Support:** Available bindings for C++, Rust, Go, and potentially Python make ICICLE accessible across different development environments.
- **Backend Agnosticism:** Develop on CPU and deploy across various backends, including GPUs, Metal, specialized hardware, and emerging platforms, based on your project's needs.
- **Extensibility:** Built for seamless integration and expansion, ICICLE enables you to develop and deploy custom backends and cryptographic primitives.

## Evolution from v2 to v3

Originally focused solely on GPU acceleration, ICICLE has evolved. With the release of v3, it now supports multiple backends, making it more versatile and adaptable across diverse hardware environments. Whether you're harnessing the power of GPUs or exploring alternative compute platforms, ICICLE v3 is built to meet your needs.

## Who Uses ICICLE?

ICICLE has been successfully integrated and used by leading cryptography companies such as [Brevis](https://www.ingonyama.com/blog/icicle-case-study-accelerating-zk-proofs-with-brevis), [Gnark](https://github.com/Consensys/gnark), [Zircuit](https://www.ingonyama.com/blog/case-study-accelerating-zircuits-zero-knowledge-proofs-with-icicle), [zkWASM](https://www.ingonyama.com/blog/how-icicle-helps-grow-the-zkwasm-ecosystem), [Kroma Network](https://www.ingonyama.com/blog/icicle-case-study-accelerating-zk-proofs-with-kroma-network) and others to enhance their ZK proving pipelines.

## Don't Have Access to a GPU?

We understand that not all developers have access to GPUs—but that shouldn’t limit your ability to build with ICICLE. Here are a few ways to get access:

### Grants

At Ingonyama, we are committed to accelerating progress in ZK and cryptography. If you're an engineer, developer, or academic researcher, we invite you to check out [our grant program](https://www.ingonyama.com/post/ingonyama-research-grant-2025). We can provide access to GPUs or even fund your research.

### Google Colab

Google Colab is a great way to start experimenting with ICICLE right away. It provides free access to NVIDIA T4 GPUs—sufficient enough for prototyping and exploring ICICLE’s capabilities.

For a detailed guide on setting up Google Colab with ICICLE, refer to [this article](./colab-instructions.md).

### Vast.ai

[Vast.ai](https://vast.ai/) offers a global GPU marketplace where you can rent different types of GPUs by the hour at competitive prices. Whether you need on-demand or interruptible rentals, Vast.ai provides flexibility for various use cases. Learn more about their rental options [here](https://vast.ai/faq#rental-types).

## What Can You Do with ICICLE?

[ICICLE](https://github.com/ingonyama-zk/icicle) can be used much like any other cryptography library—but with the added benefit of acceleration. Thanks to multiple integrations, it's already proven effective across a range of use cases:

### Boost Your ZK Prover Performance

If you're a circuit developer facing performance bottlenecks, integrating ICICLE into your prover could offer immediate relief. ICICLE is already integrated with popular ZK frameworks like [Gnark](https://github.com/Consensys/gnark) and [Halo2](https://github.com/zkonduit/halo2), enabling GPU acceleration without requiring changes to your existing code.

### Integrate with Existing ZK Provers

ICICLE supports selective acceleration, allowing you to target and optimize specific bottlenecks in your prover without a full rewrite.

### Build Custom ZK Provers

If you’re building a ZK prover from scratch, ICICLE offers a powerful foundation for creating optimized, scalable systems. Its ability to scale across multiple GPUs and machines makes it ideal for high-performance environments like data centers.

### Develop Proof-of-Concepts

ICICLE is also great for prototyping and smaller projects. With bindings for Golang and Rust, you can quickly build libraries implementing specific cryptographic primitives—like a KZG commitment scheme—with minimal overhead.

---

## Get Started with ICICLE

Explore the full capabilities of ICICLE by diving into the [Architecture](./arch_overview.md), [Getting Started Guide](./getting_started.md) and the [Programmer's Guide](./programmers_guide/general.md) to learn how to integrate, deploy, and extend ICICLE across different backends.

If you have any questions or need support, feel free to reach out on [Discord], [GitHub] or [via support email][SupportEmail]. We're here to help you accelerate your ZK development with ICICLE.

<!-- Being Links -->
[Discord]: https://discord.gg/6vYrE7waPj
[Github]: https://github.com/ingonyama-zk
[SupportEmail]: mailto:support@ingonyama.com
<!-- End Links -->
