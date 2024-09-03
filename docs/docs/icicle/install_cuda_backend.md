
# CUDA Backend

## Overview

The CUDA backend in ICICLE v3 is a high-performance, closed-source component designed to accelerate cryptographic computations using NVIDIA GPUs. This backend includes specialized libraries optimized for various cryptographic fields and curves, providing significant speedups for operations such as MSM, NTT, and elliptic curve operations.

## Installation

The CUDA backend is a closed-source component that requires a license. [To install the CUDA backend, see here](./getting_started#installing-and-using-icicle).

### Licensing

:::note
Currently, the CUDA backend is free to use via Ingonyama’s ICICLE-CUDA-backend-license server. By default, the CUDA backend will attempt to access this server. For more details, please contact support@ingonyama.com.
:::

The CUDA backend requires a valid license to function. There are two types of CUDA backend licenses:

   1. **Floating license**: In this mode, you host a license server, provided as a binary. This license supports a limited number of concurrent GPUs (N), which can be distributed across your machines as needed. N is decremented by 1 for each GPU using ICICLE per process. Once the process terminates (or crashes), the licenses are released.
   2. **Node locked license**:  In this mode, the license is tied to a specific machine. The CUDA backend will accept the license only if it is used on the licensed machine.

**To specify the license server address or file path::**

```
export ICICLE_LICENSE=port@ip            # For license server
export ICICLE_LICENSE=/path/to/license   # For node-locked license
```

For further assist , contact our support team for assistance support@ingonyama.com
