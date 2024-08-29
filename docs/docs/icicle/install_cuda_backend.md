
# CUDA Backend

## Overview

The CUDA backend in ICICLE V3 is a high-performance, closed-source component designed to accelerate cryptographic computations using NVIDIA GPUs. This backend includes a set of specialized libraries optimized for different cryptographic fields and curves, providing significant speedups for operations such as MSM, NTT, and elliptic curve operations.

## Installation

The CUDA backend is a closed-source component that requires a license. [To install the CUDA backend, see here](./install_and_use#installing-and-using-icicle).

### Licensing 

The CUDA backend requires a valid license to function. There are two CUDA backend license types:

   1. **Floating license**: In this mode, you will host a license-server that is supplied as binary. This license is limited to N concurrent gpus but can be distributed however the user needs between his machines. N is decremented by 1 for every GPU that is using ICICLE, per process. Once the process is terminated (or crashes), the licenses are released.
   2. **Node locked license**: in this mode, you will get a license for a specific machine. It is accepted by the CUDA backend only if used on the licensed machine.

:::note
As for now CUDA backend can be accessed without purchasing a license. Ingonyama is hosting a license server that will allow access to anyone.
By default CUDA backend will try to access this server if no other license is available.
To manually specify it, set `export ICICLE_LICENSE=5053@ec2-50-16-150-188.compute-1.amazonaws.com`.
:::

Licenses are available for purchase [here TODO](#) . After purchasing, you will receive a license key that must be installed on the license-server or node-locked machine.
For license-server, you will have to tell the application that is using ICICLE, where the server is.

**Specify the license server address or filepath:**

```
export ICICLE_LICENSE=port@ip # for license server
export ICICLE_LICENSE=/path/to/license # for node locked license
```

For further assist , contact our support team for assistance. `support@ingonyama.com` (TODO make sure this exists).
