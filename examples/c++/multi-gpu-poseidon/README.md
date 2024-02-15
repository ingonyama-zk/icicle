# Icicle example: using muliple GPU to hash large dataset

## Best-Practices

This example builds on [single GPU Poseidon example](../poseidon/README.md) so we recommend to run it first.

## Key-Takeaway

Use `device_context::DeviceContext` variable to select GPU to use. 
Use C++ threads to compute `Icicle` primitives on different GPUs in parallel.

## Concise Usage Explanation

1. Include c++ threads

```c++
#include <thread>
```

2. Define a __thread function__. Importantly, device context `ctx` will hold the GPU id.

```c++
void threadPoseidon(device_context::DeviceContext ctx, ...) {...}
```

3. Initialize device contexts for different GPUs

```c++
device_context::DeviceContext ctx0 = device_context::get_default_device_context();
ctx0.device_id=0;
device_context::DeviceContext ctx1 = device_context::get_default_device_context();
ctx1.device_id=1;
``` 
4. Finally, spawn the threads and wait for their completion

```c++
std::thread thread0(threadPoseidon, ctx0, ...);
std::thread thread1(threadPoseidon, ctx1, ...);
thread0.join();
thread1.join();
```

## What's in the example

This is a **toy** example executing the first step of the Filecoin's Pre-Commit 2 phase: compute $2^{30}$ Poseison hashes for each column of $11 \times 2^{30}$ matrix.

1. Define the size of the example: $2^{30}$ won't fit on a typical machine, so we partition the problem into `nof_partitions`
2. Hash two partitions in parallel on two GPUs
3. Hash two partitions in series on one GPU
4. Compare execution times

