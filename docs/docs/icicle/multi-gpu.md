# Multi GPU with ICICLE

:::info

If you are looking for the Multi GPU API documentation refer here for [Rust](./rust-bindings/multi-gpu.md).

:::

One common challenge with Zero-Knowledge computation is managing the large input sizes. It's not uncommon to encounter circuits surpassing 2^25 constraints, pushing the capabilities of even advanced GPUs to their limits. To effectively scale and process such large circuits, leveraging multiple GPUs in tandem becomes a necessity.

Multi-GPU programming involves developing software to operate across multiple GPU devices. Lets first explore different approaches to Multi-GPU programming then we will cover how ICICLE allows you to easily develop youR ZK computations to run across many GPUs.


## Approaches to Multi GPU programming

There are many [different strategies](https://github.com/NVIDIA/multi-gpu-programming-models) available for implementing multi GPU, however, it can be split into two categories.

### GPU Server approach 

This approach usually involves a single or multiple CPUs opening threads to read / write from multiple GPUs. You can think about it as a scaled up HOST - Device model.

![alt text](image.png)

This approach wont let us tackle larger computation sizes but it will allow us to compute multiple computations which we wouldn't be able to load onto a single GPU.

For example lets say that you had to compute two MSMs of size 2^26 on a 16GB VRAM GPU you would normally have to perform them asynchronously. However, if you double the number of GPUs in your system you can now run them in parallel. 


### Inter GPU approach

This approach involves a more sophisticated approach to multi GPU computation. Using technologies such as [GPUDirect, NCCL, NVSHMEM](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-cwes1084/) and NVLink it's possible to combine multiple GPUs and split a computation among different devices.

This approach requires redesigning the algorithm at the software level to be compatible with splitting amongst devices. In some cases, to lower latency to a minimum, special inter GPU connections would be installed on a server to allow direct communication between multiple GPUs.


# Writing ICICLE Code for Multi GPUs

The approach we have taken for the moment is a GPU Server approach; we assume you have a machine with multiple GPUs and you wish to run some computation on each GPU.

To dive deeper and learn about the API check out the docs for our different ICICLE API

- [Rust Multi GPU APIs](./rust-bindings/multi-gpu.md)
- C++ Multi GPU APIs


## Best practices 

- Never hardcode device IDs, if you want your software to take advantage of all GPUs on a machine use methods such as `get_device_count` to support arbitrary number of GPUs.

- Launch one CPU thread per GPU. To avoid [nasty errors](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/) and hard to read code we suggest that for every GPU you create a dedicated thread. Within a CPU thread you should be able to launch as many tasks as you wish for a GPU as long as they all run on the same GPU id. This will make your code way more manageable, easy to read and performant.

## ZKContainer support for multi GPUs

Multi GPU support should work with ZK-Containers by simply defining which devices the docker container should interact with:

```sh
docker run -it --gpus '"device=0,2"' zk-container-image
```

If you wish to expose all GPUs 

```sh
docker run --gpus all zk-container-image
```
