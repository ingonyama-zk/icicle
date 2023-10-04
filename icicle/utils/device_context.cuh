#pragma once
#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

#include <cuda_runtime.h>

namespace device_context {

/**
 * Properties of the device used in icicle functions.
 */
struct DeviceContext {
    unsigned device_id;                 /**< Index of the currently used GPU. Default value: 0. */
    cudaStream_t stream;                /**< Stream to use. Default value: 0. */
    cudaMemPool_t mempool;              /**< Mempool to use. Default value: 0. */
};

} // namespace device_context

#endif
