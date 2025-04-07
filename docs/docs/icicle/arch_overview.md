# Architecture Overview


ICICLE is designed with flexibility and extensibility in mind, offering a robust framework that supports multiple compute backends and accommodates various cryptographic needs. This section provides an overview of ICICLE's architecture, highlighting its open and closed components, multi-device support, and extensibility.

### Frontend and CPU Backend

- **Frontend (FE):** The ICICLE frontend is open-source and provides a unified API across multiple programming languages, including C++, Rust, and Go. It abstracts away the complexity of different backends, enabling developers to write backend-agnostic code that runs seamlessly across various platforms.
- **CPU Backend:** ICICLE includes an open-source CPU backend, enabling development and testing on standard hardware. It’s ideal for prototyping or for use in environments where specialized hardware is unavailable.

## CUDA Backend

- **CUDA Backend:** ICICLE also features a high-performance, closed-source CUDA backend optimized for NVIDIA GPUs. It delivers significant acceleration for cryptographic operations.
- **Installation and Licensing:** The CUDA backend must be downloaded and installed separately. For detailed instructions, refer to the [installation guide](./install_gpu_backend).

## Multi-Device Support

- **Scalability:** ICICLE supports multi-device configurations, allowing workloads to be distributed across multiple GPUs or other hardware accelerators. This enables scalable ZK proof generation and other cryptographic operations in data centers and high-performance computing environments.


## Build Your Own Backend

ICICLE is designed with a modular architecture that allows developers to integrate new backends or customize existing ones to meet their specific requirements. The architecture supports:

- **Custom Backends:** Developers can build their own backends to target specific hardware or optimize for particular use cases. The integration process is outlines in the [Build Your Own Backend](./build_your_own_backend.md) section.
- **Pluggable Components:** ICICLE’s architecture supports the seamless integration of additional cryptographic primitives and enhancements, enabling the framework to evolve alongside advances in cryptography and hardware acceleration.

