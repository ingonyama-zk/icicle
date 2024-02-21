---
slug: /
displayed_sidebar: GettingStartedSidebar
title: ''
---

# Welcome to Ingonyama's Developer Documentation

Ingonyama is a next-generation semiconductor company, focusing on Zero-Knowledge Proof hardware acceleration. We build accelerators for advanced cryptography, unlocking real-time applications. Our focus is on democratizing access to compute intensive cryptography and making it accessible for developers to build on top of.

Currently our flagship products are:

- **ICICLE**:
  [ICICLE](https://github.com/ingonyama-zk/icicle) is a fully featured GPU accelerated cryptography library for building ZK provers. ICICLE allows you to accelerate your ZK existing protocols in a matter of hours or implement your protocol from scratch on GPU.

---

## Our current take on hardware acceleration

We believe GPUs are as important for ZK as for AI.

- GPUs are a perfect match for ZK compute - around 97% of ZK protocol runtime is parallel by nature.
- GPUs are simple for developers to use and scale compared to other hardware platforms.
- GPUs are extremely competitive in terms of power / performance and price (3x cheaper).
- GPUs are popular and readily available.

For a more in-depth understanding on this topic we suggest you read [our article on the subject](https://www.ingonyama.com/blog/revisiting-paradigm-hardware-acceleration-for-zero-knowledge-proofs).

Despite our current focus on GPUs we are still hard at work developing a ZPU (ZK Processing Unit), with the goal of offering a programmable hardware platform for ZK. To read more about ZPUs we suggest you read this [article](https://medium.com/@ingonyama/zpu-the-zero-knowledge-processing-unit-f886a48e00e0).

## ICICLE

[ICICLE](https://github.com/ingonyama-zk/icicle) is a cryptography library for ZK using GPUs.
ICICLE implements blazing fast cryptographic primitives such as EC operations, MSM, NTT, Poseidon hash and more on GPU.

ICICLE is designed to be easy to use, developers don't have to touch a single line of CUDA code. Our Rust and Golang bindings allow your team to transition from CPU to GPU with minimal changes.

Learn more about ICICLE and GPUs [here][ICICLE-OVERVIEW].

## Get in Touch

If you have any questions, ideas, or are thinking of building something in this space join the discussion on [Discord]. You can explore our code on [github](https://github.com/ingonyama-zk) or read some of [our research papers](https://github.com/ingonyama-zk/papers).

Follow us on [Twitter](https://x.com/Ingo_zk) and [YouTube](https://www.youtube.com/@ingo_ZK) and sign up for our [mailing list](https://wkf.ms/3LKCbdj) to get our latest announcements.

[ICICLE-OVERVIEW]: ./icicle/overview.md
[Discord]: https://discord.gg/6vYrE7waPj
