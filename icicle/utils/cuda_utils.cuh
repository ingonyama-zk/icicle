#pragma once
#include <cuda_runtime.h>

struct cuda_ctx {
    int device_id;
    cudaMemPool_t mempool;
    cudaStream_t stream;

    cuda_ctx(int gpu_id) {
        gpu_id = gpu_id;
        cudaMemPoolProps pool_props;
        pool_props.allocType = cudaMemAllocationTypePinned;
        pool_props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
        pool_props.location.type = cudaMemLocationTypeDevice;
        pool_props.location.id = device_id;

        cudaMemPoolCreate(&mempool, &pool_props);
        cudaStreamCreate(&stream);
    }

    void set_device() {
        cudaSetDevice(device_id);
    }

    void sync_stream() {
        cudaStreamSynchronize(stream);
    }

    void malloc(void *ptr, size_t bytesize) {
        cudaMallocFromPoolAsync(&ptr, bytesize, mempool, stream);
    }

    void free(void *ptr) {
        cudaFreeAsync(ptr, stream);
    }


};

// -- Proposed Function Tops --------------------------------------------------
// ----------------------------------------------------------------------------
