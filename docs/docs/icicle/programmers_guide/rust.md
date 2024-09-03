
# Icicle Rust Usage Guide

## Overview

This guide covers the usage of ICICLE’s Rust API, including device management, memory operations, data transfer, synchronization, and compute APIs.

## Build the Rust Application and Execute

To successfully build and execute the Rust application using ICICLE, you need to define the ICICLE dependencies in your Cargo.toml file:

```bash
[dependencies]
icicle-runtime = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
icicle-core = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
icicle-babybear = { git = "https://github.com/ingonyama-zk/icicle.git", branch="main" }
# add other ICICLE crates here as needed
```

Once the dependencies are defined, you can build and run your application using the following command:
```bash
cargo run --release
```

This will compile your Rust application with optimizations and execute it.

:::note
The icicle-runtime crate is used to load backends, select a device, and interact with the device in an abstract way when managing memory, streams, and other resources, as explained in this guide.
:::

## Device Management

### Loading a Backend

The backend can be loaded from a specific path or from an environment variable. This is essential for setting up the computing environment.

```rust
use icicle_runtime::runtime;

runtime::load_backend_from_env_or_default().unwrap();
// or load from custom install dir
runtime::load_backend("/path/to/backend/installdir").unwrap();
```

### Setting and Getting Active Device

You can set the active device for the current thread and retrieve it when needed:

```rust
use icicle_runtime::Device;

let device = Device::new("CUDA", 0); // or other
icicle_runtime::set_device(&device).unwrap();

let active_device = icicle_runtime::get_active_device().unwrap();
```

### Querying Device Information

Retrieve the number of available devices and check if a pointer is allocated on the host or on the active device:

```rust
let device_count = icicle_runtime::get_device_count().unwrap();
```

## Memory Management

### Allocating and Freeing Memory

Memory can be allocated on the active device using the `DeviceVec` API. This memory allocation is flexible, as it supports allocation on any device, including the CPU if the CPU backend is used.

```rust
use icicle_runtime::memory::DeviceVec;

// Allocate 1024 elements on the device
let mut device_memory: DeviceVec<u8> = DeviceVec::<u8>::device_malloc(1024).unwrap();
```

The memory is released when the `DeviceVec` object is dropped.

### Asynchronous Memory Operations

Asynchronous memory operations can be performed using streams. This allows for non-blocking execution, with memory allocation and deallocation occurring asynchronously.
```rust
use icicle_runtime::stream::IcicleStream;
use icicle_runtime::memory::DeviceVec;

let mut stream = IcicleStream::create().unwrap(); // mutability is for the destroy() method

// Allocate 1024 elements asynchronously on the device
let mut device_memory: DeviceVec<u8> = DeviceVec::<u8>::device_malloc_async(1024, &stream).unwrap();

// dispatch additional copy, compute etc. ops to the stream

// Synchronize the stream to ensure all operations are complete
stream.synchronize().unwrap();
stream.destroy().unwrap(); //
```

:::note
Streams need be explicitly destroyed before being dropped.
:::

### Querying Available Memory

You can retrieve the total and available memory on the active device using the `get_available_memory` function.

```rust
use icicle_runtime::memory::get_available_memory;

// Retrieve total and available memory on the active device
let (total_memory, available_memory) = get_available_memory().unwrap();

println!("Total memory: {}", total_memory);
println!("Available memory: {}", available_memory);
```

This function returns a tuple containing the total memory and the currently available memory on the device. It is essential for managing and optimizing resource usage in your applications.

## Data Transfer

### Copying Data

Data can be copied between the host and device, or between devices. The location of the memory is handled by the `HostOrDeviceSlice` and `DeviceSlice` traits:

```rust
use icicle_runtime::memory::{DeviceVec, HostSlice};

// Copy data from host to device
let input = vec![1, 2, 3, 4];
let mut d_mem = DeviceVec::<u32>::device_malloc(input.len()).unwrap();
d_mem.copy_from_host(HostSlice::from_slice(&input)).unwrap();
// OR
d_mem.copy_from_host_async(HostSlice::from_slice(&input, &stream)).unwrap();

// Copy data back from device to host
let mut output = vec![0; input.len()];
d_mem.copy_to_host(HostSlice::from_mut_slice(&mut output)).unwrap();
// OR
d_mem.copy_to_host_async(HostSlice::from_mut_slice(&mut output, &stream)).unwrap();
```
## Stream Management

### Creating and Destroying Streams

Streams in Icicle are used to manage asynchronous operations, ensuring that computations can run in parallel without blocking the CPU thread:

```rust
use icicle_runtime::stream::IcicleStream;

// Create a stream
let mut stream = IcicleStream::create().unwrap();

// Destroy the stream
stream.destroy().unwrap();
```

## Synchronization

### Synchronizing Streams and Devices

Synchronization ensures that all previous operations on a stream or device are completed before moving on to the next task. This is crucial when coordinating between multiple dependent operations:

```rust
use icicle_runtime::stream::IcicleStream;

// Synchronize the stream
stream.synchronize().unwrap();

// Synchronize the device
icicle_runtime::device_synchronize().unwrap();
```
These functions ensure that your operations are properly ordered and completed before the program proceeds, which is critical in parallel computing environments.

## Device Properties

### Checking Device Availability

Check if a specific device is available and retrieve a list of registered devices:
```rust
use icicle_runtime::Device;

let cuda_device = Device::new("CUDA", 0);
if icicle_runtime::is_device_available(&cuda_device) {
    println!("CUDA device is available.");
} else {
    println!("CUDA device is not available.");
}

let registered_devices = icicle_runtime::get_registered_devices().unwrap();
println!("Registered devices: {:?}", registered_devices);
```

### Querying Device Properties

Retrieve properties of the active device to understand its capabilities and configurations:

```rust
use icicle_runtime::Device;

let cuda_device = Device::new("CUDA", 0);
if icicle_runtime::is_device_available(&cuda_device) {
    icicle_runtime::set_device(&cuda_device);
    let device_props = icicle_runtime::get_device_properties().unwrap();
    println!("Device using host memory: {}", device_props.using_host_memory);
}
```

These functions allow you to query device capabilities and ensure that your application is running on the appropriate hardware.

## Compute APIs

### Multi-Scalar Multiplication (MSM) Example

Icicle provides high-performance compute APIs such as Multi-Scalar Multiplication (MSM) for cryptographic operations. Here's a simple example of how to use the MSM API in Rust.

```rust
// Using bls12-377 curve
use icicle_bls12_377::curve::{CurveCfg, G1Projective, ScalarCfg};
use icicle_core::{curve::Curve, msm, msm::MSMConfig, traits::GenerateRandom};
use icicle_runtime::{device::Device, memory::HostSlice};

fn main() {
    // Load backend and set device
    let _ = icicle_runtime::runtime::load_backend_from_env_or_default();
    let cuda_device = Device::new("CUDA", 0);
    if icicle_runtime::is_device_available(&cuda_device) {
        icicle_runtime::set_device(&cuda_device).unwrap();
    }

    let size = 1024;

    // Randomize inputs
    let points = CurveCfg::generate_random_affine_points(size);
    let scalars = ScalarCfg::generate_random(size);

    let mut msm_results = vec![G1Projective::zero(); 1];
    msm::msm(
        HostSlice::from_slice(&scalars),
        HostSlice::from_slice(&points),
        &MSMConfig::default(),
        HostSlice::from_mut_slice(&mut msm_results[..]),
    )
    .unwrap();
    println!("MSM result = {:?}", msm_results);
}
```
