# Architecture Overview

## Introduction

ICICLE V3 is designed with flexibility and extensibility in mind, offering a robust framework that supports multiple compute backends and accommodates various cryptographic needs. This section provides an overview of ICICLE's architecture, highlighting its open and closed components, multi-device support, and extensibility.

## Open Frontend and CPU Backend

- **Frontend (FE):** The ICICLE frontend is open-source and designed to provide a unified API across different programming languages, including C++, Rust, and Go. This frontend abstracts the complexity of working with different backends, allowing developers to write backend-agnostic code that can be deployed across various platforms.
- **CPU Backend:** ICICLE includes an open-source CPU backend that allows for development and testing on standard hardware. This backend is ideal for prototyping and for environments where specialized hardware is not available.

## Closed CUDA Backend

- **CUDA Backend:** ICICLE also includes a high-performance CUDA backend that is closed-source. This backend is optimized for NVIDIA GPUs and provides significant acceleration for cryptographic operations. 
- **Installation and Licensing:** The CUDA backend requires a valid license for deployment. For detailed instructions on installing and licensing the CUDA backend, refer to the [installation guide](./install_cuda_backend.md).

## Extensible Design

ICICLE is designed to be extensible, allowing developers to integrate new backends or customize existing ones to suit their specific needs. The architecture supports:

- **Custom Backends:** Developers can create their own backends to leverage different hardware or optimize for specific use cases. The process of building and integrating a custom backend is documented in the [Build Your Own Backend](./build_your_own_backend.md) section.
- **Pluggable Components:** ICICLE's architecture allows for easy integration of additional cryptographic primitives or enhancements, ensuring that the framework can evolve with the latest advancements in cryptography and hardware acceleration.

TODO ADD diagram

## Multi-Device Support

- **Scalability:** ICICLE supports multi-device configurations, enabling the distribution of workloads across multiple GPUs or other hardware accelerators. This feature allows for scaling ZK proofs and other cryptographic operations across larger data centers or high-performance computing environments.
- **Device Management:** The architecture includes tools for managing multiple devices, ensuring that resources are efficiently utilized and that workloads are balanced across available hardware.

---

### Conclusion

The architecture of ICICLE V3 is built to be flexible, scalable, and extensible, making it a powerful tool for developers working with zero-knowledge proofs and other cryptographic operations. Whether you're working with open-source CPU backends or closed-source CUDA backends, ICICLE provides the tools and flexibility needed to achieve high performance and scalability in cryptographic computations.

Explore the following sections to learn more about building your own backend, using ICICLE across multiple devices, and integrating it into your projects.
