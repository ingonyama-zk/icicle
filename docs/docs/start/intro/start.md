---
slug: /
title: Overview
---

# Introduction to ICICLE

ICICLE is a versatile cryptography library supporting multiple compute backends—including CUDA, CPU, Metal, and upcoming backends like WebGPU, Vulkan, and ZPU. It enables you to build cryptographic applications with ease, leveraging the best available hardware for your needs.

import Link from '@docusaurus/Link';

<div className="card-grid">

  <Link to="/start/architecture/arch_overview" className="card-link">
    <div className="card-box">
    <img src="/img/Arch.png" alt="Arch Icon" className="card-icon-left" />
      <h3>Architecture</h3>
      <p>Flexible, extensible framework for cryptographic multi-backend support.</p>
    </div>
  </Link>

  <Link to="/start/programmers_guide/general" className="card-link">
    <div className="card-box">
      <img src="/img/Programguide.png" alt="Prog Icon" className="card-icon-left" />
      <div className="card-text">
        <h3>Programmer's Guide</h3>
        <p>Consistent cryptographic APIs: NTT, MSM, ECNTT, polynomials.</p>
      </div>
    </div>
  </Link>

  <Link to="/start/integration-&-support/benchmarks" className="card-link">
    <div className="card-box">
      <img src="/img/Intandsup.png" alt="Int&Sup Icon" className="card-icon-left" />
      <div className="card-text">
        <h3>Integrations & Support</h3>
        <p>Broad integrations with strong community and developer support.</p>
      </div>
    </div>
  </Link>

  <Link to="/apioverview" className="card-link">
    <div className="card-box">
      <img src="/img/API.png" alt="API Icon" className="card-icon-left" />
      <div className="card-text">
        <h3>API</h3>
        <p>Core ICICLE APIs: MSM, NTT, hashing with examples.</p>
      </div>
    </div>
  </Link>

</div>

## Overview

#### High-Speed Cryptography

ICICLE delivers optimized performance for the core building blocks used in modern cryptographic protocols, ensuring efficiency and scalability across diverse use cases.

#### Modular and Extensible Libraries

ICICLE includes a comprehensive set of libraries for various fields and curves, and is built for seamless integration and expansion—allowing you to add custom backends or cryptographic primitives as needed.

#### Cross-Platform and Cross-Language Support

ICICLE works across multiple languages (C++, Rust, Go, potentially Python) and supports development on CPU with deployment across diverse backends, including GPUs, Metal, specialized hardware, and emerging platforms.

## What Can You Do with ICICLE?

At Ingonyama, we’re committed to accelerating progress in cryptography—not just through hardware and software, but by supporting the people building with them. Whether you're an engineer, developer, or academic researcher, [our grant program](https://www.ingonyama.com/post/ingonyama-research-grant-2025) can provide access to GPUs or even fund your research.

ICICLE can be used much like any other cryptographic library—with the added benefit of acceleration. Thanks to multiple integrations, it's already proven effective across a range of use cases:

#### Boost Your Prover Performance

If you're a circuit developer facing performance bottlenecks, integrating ICICLE into your prover could offer immediate relief. ICICLE is already integrated with popular frameworks like [Gnark](https://github.com/Consensys/gnark) and [Halo2](https://github.com/zkonduit/halo2), enabling GPU acceleration without requiring changes to your existing code.

#### Integrate with Existing Provers

ICICLE supports selective acceleration, allowing you to target and optimize specific bottlenecks in your prover without a full rewrite.

#### Build Custom Provers

If you’re building a prover from scratch, ICICLE offers a powerful foundation for creating optimized, scalable systems. Its ability to scale across multiple GPUs and machines makes it ideal for high-performance environments like data centers.

#### Develop Proof-of-Concepts

ICICLE is also great for prototyping and smaller projects. With bindings for Golang and Rust, you can quickly build libraries implementing specific cryptographic primitives—like a KZG commitment scheme—with minimal overhead.

## Ecosystem

ICICLE is already trusted by leading cryptography teams to supercharge their proving systems. From general-purpose ZK frameworks to specialized proof systems, ICICLE helps teams break performance bottlenecks and scale efficiently.

Below are just a select *few* of the teams building with ICICLE:

<div className="ecosystem-grid">

  <a href="https://www.ingonyama.com/post/icicle-case-study-accelerating-zk-proofs-with-brevis" className="ecosystemcard" target="_blank" rel="noopener">
    <img src="/img/brevislogo.png" alt="Brevis logo" />
    <p>Accelerating proof generation for ZK coprocessors.</p>
  </a>

  <a href="https://www.eigenda.xyz/" className="ecosystemcard" target="_blank" rel="noopener">
    <img src="/img/eigendalogo.png" alt="EigenDA logo" />
    <p>Improving data availability bandwidth, utilizing GPUs to parallelize encoding.</p>
  </a>

  <a href="https://github.com/Consensys/gnark" className="ecosystemcard" target="_blank" rel="noopener">
    <img src="/img/gnarklogo.png" alt="Gnark logo" />
    <p>Speeding up general-purpose zero-knowledge circuits.</p>
  </a>

  <a href="https://www.ingonyama.com/blog/case-study-accelerating-zircuits-zero-knowledge-proofs-with-icicle" className="ecosystemcard" target="_blank" rel="noopener">
    <img 
    src="/img/zircuitlogo.png" 
    alt="Zircuit logo"
    style={{ width: '140px', height: 'auto' }}/>
    <p>Boosting throughput for custom ZK proof systems.</p>
  </a>

  <a href="https://www.ingonyama.com/blog/how-icicle-helps-grow-the-zkwasm-ecosystem" className="ecosystemcard" target="_blank" rel="noopener">
  <img
  src="/img/zkwasmlogo.png"
  alt="zkWASM logo"
  style={{ width: '150px', height: 'auto' }}
/>
    <p>Powering scalable zkVMs with modular acceleration.</p>
  </a>

  <a href="https://www.ingonyama.com/blog/icicle-case-study-accelerating-zk-proofs-with-kroma-network" className="ecosystemcard" target="_blank" rel="noopener">
    <img 
    src="/img/kromalogo.png" 
    alt="Kroma Network logo"
    style={{ width: '150px', height: 'auto' }} />
    <p>Accelerating optimistic rollup proof generation.</p>
  </a>

</div>

