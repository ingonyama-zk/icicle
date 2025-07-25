# ICICLE PQC Example (C++)

This example demonstrates how to use ICICLE PQC (Post-Quantum Cryptography) in a C++ application to perform ML-KEM (Module-Lattice-based Key Encapsulation Mechanism) operations.

## Prerequisites

- **NVIDIA GPU**: CUDA-capable NVIDIA hardware
- **NVIDIA Drivers**: Properly installed NVIDIA drivers
- **ICICLE PQC**: ICICLE PQC library installation

## Installation

### Build and Install ICICLE PQC from Source

1. **Create build directory and configure with PQC flags**

    ```bash
    mkdir -p build
    cmake -S . -B build \
        -DPQC=ON \
        -DCUDA_PQC_BACKEND=ON \
        -DPQC_PACKAGE=ON \
        -DICICLE_STATIC_LINK=ON \
        -DCMAKE_INSTALL_PREFIX=/path/to/icicle/pqc/install
    ```

2. **Build and install**

    ```bash
    cmake --build build -j
    cmake --install build
    ```

3. **Set environment variable**

    ```bash
    export ICICLE_PQC_INSTALL_DIR=/path/to/icicle/pqc/install
    ```

> **Note**: Replace `/path/to/icicle/pqc/install` with your desired installation directory.

## Building the Example

### Using the Run Script (Recommended)

The easiest way to build and run the example is using the `run.sh` script:

```bash
# Build and run with ICICLE PQC installation path
./run.sh /path/to/icicle/pqc/install

# Build and run with custom batch size
./run.sh /path/to/icicle/pqc/install 10

# Use environment variable for install path
export ICICLE_PQC_INSTALL_DIR=/path/to/icicle/pqc/install
./run.sh

# Get help and see all usage options
./run.sh --help
```

### Using the Build Script Only

If you prefer to build without running, use the build script:

```bash
./build.sh
```

## Running the Example

### Using the Run Script (Recommended)

The easiest way is to use the `run.sh` script which builds and runs in one command:

```bash
# Build and run with default batch size
./run.sh /path/to/icicle/pqc/install

# Build and run with custom batch size
./run.sh /path/to/icicle/pqc/install 10
```

### Manual Execution

If you built using `build.sh` or manual build, you can run the compiled example directly:

```bash
# Run with default batch size (1)
./build/pqc_example

# Run with custom batch size
./build/pqc_example 10
./build/pqc_example 100
```

## Troubleshooting

### NVIDIA Drivers Not Found

```txt
❌ NVIDIA drivers not found. This example requires NVIDIA GPU support.
Please install NVIDIA drivers and ensure your GPU is properly configured.
```

**Solution**: Install NVIDIA drivers using your system's package manager or download from NVIDIA's website.

### ICICLE PQC Installation Not Found

```txt
❌ ICICLE_PQC_INSTALL_DIR environment variable not set.
Please set it to your ICICLE PQC installation directory:
  export ICICLE_PQC_INSTALL_DIR=/path/to/icicle/pqc/install
```

**Solution**: Set the environment variable to point to your ICICLE PQC installation directory.

### Build Failures

- Ensure all dependencies are properly installed
- Verify that the `ICICLE_PQC_INSTALL_DIR` path is correct
- Check that your CUDA installation is compatible

## Performance Notes

- **Batch Processing**: The example demonstrates how batch operations can significantly improve throughput
- **GPU Memory**: Larger batch sizes require more GPU memory
- **Parameter Sets**: Higher security levels (ML-KEM-1024) have larger keys and slower operations

## About ML-KEM

ML-KEM is a post-quantum key encapsulation mechanism standardized by NIST. It's designed to be secure against attacks by both classical and quantum computers, making it essential for future cryptographic applications.
