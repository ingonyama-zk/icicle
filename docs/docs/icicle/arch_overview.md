# Architecture Overview


ICICLE v3 is designed with flexibility and extensibility in mind, offering a robust framework that supports multiple compute backends and accommodates various cryptographic needs. This section provides an overview of ICICLE's architecture, highlighting its open and closed components, multi-device support, and extensibility.

### Open Frontend and CPU Backend

- **Frontend (FE):** The ICICLE frontend is open-source and designed to provide a unified API across different programming languages, including C++, Rust, and Go. This frontend abstracts the complexity of working with different backends, allowing developers to write backend-agnostic code that can be deployed across various platforms.
- **CPU Backend:** ICICLE includes an open-source CPU backend that allows for development and testing on standard hardware. This backend is ideal for prototyping and for environments where specialized hardware is not available.

## CUDA Backend

- **CUDA Backend:** ICICLE also includes a high-performance CUDA backend that is closed-source. This backend is optimized for NVIDIA GPUs and provides significant acceleration for cryptographic operations. 
- **Installation and Licensing:** The CUDA backend needs to be downloaded and installed. Refer to the [installation guide](./install_cuda_backend.md) for detailed instructions.

## Multi-Device Support

- **Scalability:** ICICLE supports multi-device configurations, enabling the distribution of workloads across multiple GPUs or other hardware accelerators. This feature allows for scaling ZK proofs and other cryptographic operations across larger data centers or high-performance computing environments.


## Build Your Own Backend

ICICLE is designed to be extensible, allowing developers to integrate new backends or customize existing ones to suit their specific needs. The architecture supports:

- **Custom Backends:** Developers can create their own backends to leverage different hardware or optimize for specific use cases. The process of building and integrating a custom backend is documented in the [Build Your Own Backend](./build_your_own_backend.md) section.
- **Pluggable Components:** ICICLE's architecture allows for easy integration of additional cryptographic primitives or enhancements, ensuring that the framework can evolve with the latest advancements in cryptography and hardware acceleration.


