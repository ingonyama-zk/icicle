# GPU Backends

## Overview

ICICLE v3 supports high-performance GPU backends, including **CUDA** (for NVIDIA GPUs) and **Metal** (for Apple Silicon GPUs), to accelerate cryptographic computations. These backends provide optimized libraries for various cryptographic primitives such as MSM, NTT, and elliptic curve operations, offering significant performance improvements across supported hardware platforms.

## Installation

GPU backends are closed-source components that require a valid license. [To install and configure GPU backends in ICICLE, see here](../../setup#installing-and-using-icicle)

## Licensing

:::note
By default, ICICLE connects to Ingonyama’s backend license server and runs with a **free Research & Development license**, enabling GPU backends (CUDA and Metal) for development and testing purposes—no setup required.
For **production use**, please contact sales@ingonyama.com.  
  
For additional support, email support@ingonyama.com.
:::

Each backend (CUDA or Metal) requires a valid license to function. ICICLE supports two types of licenses:

1. **Floating license**:  
   You host a license server (provided as a binary). This license supports a limited number of concurrent GPUs (N), which can be used across multiple machines. Each process using a GPU decrements N by 1. When the process ends or crashes, the license is released.

2. **Node-locked license**:  
   The license is bound to a specific machine. The backend will only operate on the licensed device.

### Setting the License

To specify the license server or license file:

```bash
export ICICLE_LICENSE=port@ip            # For license server
export ICICLE_LICENSE=/path/to/license   # For node-locked license
```

For further assistance, contact our support team for assistance support@ingonyama.com.
