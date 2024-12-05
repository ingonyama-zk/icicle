# ICICLE best practices: Concurrent Data Transfer and NTT Computation

The [Number Theoretic Transform (NTT)](https://dev.ingonyama.com/icicle/primitives/ntt) is an integral component of many cryptographic algorithms, such as polynomial multiplication in Zero Knowledge Proofs. The performance bottleneck of NTT on GPUs is the data transfer between the host (CPU) and the device (GPU). In a typical NVIDIA GPU this transfer dominates the total NTT execution time.

## Key-Takeaway

When you have to run several NTTs, consider Concurrent Data Download, Upload, and Computation to improve data bus (PCIe) and GPU utilization, and get better total execution time.

Typically, you concurrently

1. Download the output of a previous NTT back to the host
2. Upload the input for a next NTT on the device
3. Run current NTT

> [!NOTE]
> This approach requires two on-device memory vectors, decreasing the maximum size of NTT by 2x.

## Best-Practices

1. Use three separate streams for Download to device, Upload from device, and Compute operations
2. Future: Use pinned (page-locked) memory on host to speed data bus transfers.
3. Compute in-place NTT.

## Running the example

To change the default curve BN254, edit `run.sh` and `CMakeLists.txt`

```sh
# for CPU
./run.sh -d CPU
# for CUDA
./run.sh -d CUDA -b /path/to/cuda/backend/install/dir
# for METAL
./run.sh -d METAL -b /path/to/cuda/backend/install/dir
```

To compare with ICICLE baseline (i.e. non-concurrent) NTT, you can run [this example](../ntt/README.md).
