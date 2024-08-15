# Icicle Programmer's Guide

## General Concepts

### Device Abstraction

Icicle provides a device abstraction layer that allows you to interact with different compute devices such as CPUs and GPUs seamlessly. The device abstraction ensures that your code can work across multiple hardware platforms without modification.

#### Device Management

- **Loading Backends**: Backends are loaded dynamically based on the environment configuration or a specified path.
- **Setting Active Device**: The active device for a thread can be set, allowing for targeted computation on a specific device.

### Streams

Streams in Icicle allow for asynchronous execution and memory operations, enabling parallelism and non-blocking execution. Streams are associated with specific devices, and you can create, destroy, and synchronize streams to manage your workflow.

### Memory Management

Icicle provides functions for allocating, freeing, and managing memory across devices. Memory operations can be performed synchronously or asynchronously, depending on the use case.

### Data Transfer

Data transfer between the host and devices, or between different devices, is handled through a set of APIs that ensure efficient and error-checked operations. Asynchronous operations are supported to maximize performance.

### Synchronization

Synchronization ensures that all previous operations on a stream or device are completed. This is crucial when coordinating between multiple operations that depend on one another.

## Additional Information

- **Error Handling**: Icicle uses a specific error enumeration (`eIcicleError`) to handle and return error states across its APIs.
- **Device Properties**: You can query various properties of devices to tailor operations according to the hardware capabilities.


