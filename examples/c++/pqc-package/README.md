# Example: Install and use ICICLE PQC (C++)

This example demonstrates how to install ICICLE PQC (Post-Quantum Cryptography) binaries and use them in a C++ application to perform ML-KEM (Module-Lattice-based Key Encapsulation Mechanism) operations.

## Prerequisites

- **NVIDIA GPU**: This example requires CUDA-capable NVIDIA hardware
- **NVIDIA Drivers**: Properly installed NVIDIA drivers
- **ICICLE PQC**: ICICLE PQC library installation

## Installation

### Build and Install ICICLE PQC from Source

Build ICICLE PQC from the source repository:

```bash

# Create build directory and configure with PQC flags
mkdir -p build
cmake -S . -B build \
    -DPQC=ON \
    -DCUDA_PQC_BACKEND=ON \
    -DPQC_PACKAGE=ON \
    -DICICLE_STATIC_LINK=ON \
    -DCMAKE_INSTALL_PREFIX=/path/to/icicle/pqc/install

# Build and install
cmake --build build -j
cmake --install build

# Set environment variable
export ICICLE_PQC_INSTALL_DIR=/path/to/icicle/pqc/install
```

> [!NOTE]
> Replace `/path/to/icicle/pqc/install` with your desired installation directory. The CMake flags enable PQC support, CUDA backend, packaging, and static linking.

## Building the Example

### Using the Build Script (Recommended)

The example includes a comprehensive build script that checks for dependencies:

```bash
./build.sh
```

The build script will:

1. ✅ Check for NVIDIA drivers
2. ✅ Verify ICICLE PQC installation
3. 🔨 Configure and build the project using CMake

### Manual Build

If you prefer to build manually:

```bash
# Ensure ICICLE_PQC_INSTALL_DIR is set
export ICICLE_PQC_INSTALL_DIR=/path/to/icicle/pqc/install

# Create build directory and configure
mkdir -p build
cmake -S . -B build -DICICLE_PQC_INSTALL_DIR="$ICICLE_PQC_INSTALL_DIR"

# Build the project
cmake --build build
```

## Running the Example

After successful compilation, you can run the PQC example:

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
