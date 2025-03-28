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

[ICICLE](https://github.com/ingonyama-zk/icicle) is a cryptography library built to accelerate high-speed cryptography, starting with zero-knowledge proofs (ZKPs), using multiple compute backends, including GPUs, CPUs, and other platforms. Its key strength lies in implementing ultra-fast cryptographic primitives, allowing developers to drastically reduce proving times with minimal effort.

## Key Features

- **Acceleration of “zk” Math:** ICICLE provides optimized implementations for cryptographic primitives crucial to zero-knowledge proofs, such as elliptic curve operations, MSM, NTT, Poseidon hash, and more.
- **Set of Libraries:** ICICLE includes a comprehensive set of libraries supporting various fields, curves, and other cryptographic needs.
- **Cross-Language Support:** Available bindings for C++, Rust, Go, and potentially Python make ICICLE accessible across different development environments.
- **Backend Agnosticism:** Develop on CPU and deploy across various backends, including GPUs, specialized hardware, and emerging platforms, based on your project's needs.
- **Extensibility:** Built for seamless integration and expansion, enabling you to develop and deploy custom backends and cryptographic primitives.

## Evolution from v2 to v3

Originally, ICICLE was focused solely on GPU acceleration. With the release of v3, ICICLE now supports multiple backends, making it more versatile and adaptable to different hardware environments. Whether you're leveraging the power of GPUs or exploring other compute platforms, ICICLE v3 is designed to fit your needs.

## Who Uses ICICLE?

ICICLE has been successfully integrated and used by leading cryptography companies such as [Brevis](https://www.ingonyama.com/blog/icicle-case-study-accelerating-zk-proofs-with-brevis), [Gnark](https://github.com/Consensys/gnark), [Zircuit](https://www.ingonyama.com/blog/case-study-accelerating-zircuits-zero-knowledge-proofs-with-icicle), [zkWASM](https://www.ingonyama.com/blog/how-icicle-helps-grow-the-zkwasm-ecosystem), [Kroma Network](https://www.ingonyama.com/blog/icicle-case-study-accelerating-zk-proofs-with-kroma-network) and others to enhance their ZK proving pipelines.

## Don't Have Access to a GPU?

We understand that not all developers have access to GPUs, but this shouldn't limit your ability to develop with ICICLE. Here are some ways to gain access to GPUs:

### Grants

At Ingonyama, we are committed to accelerating progress in ZK and cryptography. If you're an engineer, developer, or academic researcher, we invite you to check out [our grant program](https://www.ingonyama.com/blog/ingonyama-research-grants-2025). We can provide access to GPUs and even fund your research.

### Google Colab

Google Colab is a great platform to start experimenting with ICICLE instantly. It offers free access to NVIDIA T4 GPUs, which are more than sufficient for experimenting and prototyping with ICICLE.

For a detailed guide on setting up Google Colab with ICICLE, refer to [this article](./colab-instructions.md).

### Vast.ai

[Vast.ai](https://vast.ai/) offers a global GPU marketplace where you can rent different types of GPUs by the hour at competitive prices. Whether you need on-demand or interruptible rentals, Vast.ai provides flexibility for various use cases. Learn more about their rental options [here](https://vast.ai/faq#rental-types).

## What Can You Do with ICICLE?

[ICICLE](https://github.com/ingonyama-zk/icicle) can be used similarly to any other cryptography library. Through various integrations, ICICLE has proven effective in multiple use cases:

### Boost Your ZK Prover Performance

If you're a circuit developer facing bottlenecks, integrating ICICLE into your prover may solve performance issues. ICICLE is integrated into popular ZK provers like [Gnark](https://github.com/Consensys/gnark) and [Halo2](https://github.com/zkonduit/halo2), enabling immediate GPU acceleration without additional code changes.

### Integrating into Existing ZK Provers

ICICLE allows for selective acceleration of specific parts of your ZK prover, helping to address specific bottlenecks without requiring a complete rewrite of your prover.

### Developing Your Own ZK Provers

For those building ZK provers from the ground up, ICICLE is an ideal tool for creating optimized and scalable provers. The ability to scale across multiple machines within a data center is a key advantage when using ICICLE with GPUs.

### Developing Proof of Concepts

ICICLE is also well-suited for prototyping and developing small-scale projects. With bindings for Golang and Rust, you can easily create a library implementing specific cryptographic primitives, such as a KZG commitment library, using ICICLE.

---

## Get Started with ICICLE

Explore the full capabilities of ICICLE by diving into the [Architecture](./arch_overview.md), [Getting Started Guide](./getting_started.md) and the [Programmer's Guide](./programmers_guide/general.md) to learn how to integrate, deploy, and extend ICICLE across different backends.

If you have any questions or need support, feel free to reach out on [Discord], [GitHub] or [via support email][SupportEmail]. We're here to help you accelerate your ZK development with ICICLE.

<!-- Being Links -->
[Discord]: https://discord.gg/6vYrE7waPj
[Github]: https://github.com/ingonyama-zk
[SupportEmail]: mailto:support@ingonyama.com
<!-- End Links -->
