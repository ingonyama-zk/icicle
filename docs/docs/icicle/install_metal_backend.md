
# METAL Backend

## Overview

The METAL backend in ICICLE v3.6 is a high-performance, closed-source component designed to accelerate cryptographic computations using METAL GPUs. This backend includes specialized libraries optimized for various cryptographic fields and curves, providing significant speedups for operations such as MSM, NTT, and elliptic curve operations.

## Installation

The METAL backend is a closed-source component that requires a license.
To install the METAL backend, (same as CUDA backend), download, extract and set the 'ICICLE_BACKEND_INSTALL_DIR' env var.

### Licensing

:::note
Currently, the METAL backend is free to use **only for research and development purposes** via Ingonyama’s backend license server. By default, the METAL backend will attempt to access this server. For **production use**, please contact sales@ingonyama.com.

For more details, please contact support@ingonyama.com.
:::

The METAL backend requires a valid license to function. There are two types of METAL backend licenses:

   1. **Floating license**: In this mode, you host a license server, provided as a binary. This license supports a limited number of concurrent GPUs (N), which can be distributed across your machines as needed. N is decremented by 1 for each GPU using ICICLE per process. Once the process terminates (or crashes), the licenses are released.
   2. **Node locked license**:  In this mode, the license is tied to a specific machine. The METAL backend will accept the license only if it is used on the licensed machine.

**To specify the license server address or file path::**

```
export ICICLE_LICENSE=port@ip            # For license server
export ICICLE_LICENSE=/path/to/license   # For node-locked license
```

For further assist , contact our support team for assistance support@ingonyama.com
