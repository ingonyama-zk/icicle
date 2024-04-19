#include <iostream>
#include <thread>
#include <chrono>
#include <nvml.h>

#include "api/bn254.h"
#include "gpu-utils/error_handler.cuh"

using namespace poseidon;
using namespace bn254;

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // Handle the error, e.g., exit the program or throw an exception.
    }
}

// these global constants go into template calls
const int size_col = 11;

// this function executes the Poseidon thread
void threadPoseidon(device_context::DeviceContext ctx, unsigned size_partition, scalar_t * layers, scalar_t * column_hashes, PoseidonConstants<scalar_t> * constants) {
    cudaError_t err_result =  CHK_STICKY(cudaSetDevice(ctx.device_id));
    if (err_result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err_result) << std::endl;
        return; 
    }
    // CHK_IF_RETURN(); I can't use it in a standard thread function
    PoseidonConfig column_config = {
        ctx,   // ctx
        false, // are_inputes_on_device
        false, // are_outputs_on_device
        false, // input_is_a_state
        false, // aligned
        false, // loop_state
        false, // is_async
        };
    cudaError_t err = bn254PoseidonHash(layers, column_hashes, (size_t) size_partition, size_col, *constants, column_config);
    checkCudaError(err);
}

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());


#define CHECK_ALLOC(ptr) if ((ptr) == nullptr) { \
    std::cerr << "Memory allocation for '" #ptr "' failed." << std::endl; \
    exit(EXIT_FAILURE); \
}

int main() {
    const unsigned size_row = (1<<30);
    const unsigned nof_partitions = 64;
    const unsigned size_partition = size_row / nof_partitions;
    // layers is allocated only for one partition, need to reuse for different partitions
    const uint32_t size_layers = size_col * size_partition;
    
    nvmlInit();
    unsigned int deviceCount;
    nvmlDeviceGetCount(&deviceCount);
    std::cout << "Available GPUs: " << deviceCount << std::endl;

    for (unsigned int i = 0; i < deviceCount; ++i) {
        nvmlDevice_t device;
        nvmlMemory_t memory;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlDeviceGetHandleByIndex(i, &device);
        nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        nvmlDeviceGetMemoryInfo(device, &memory);
        std::cout << "Device ID: " << i << ", Type: " << name << ", Memory Total/Free (MiB) " << memory.total/1024/1024 << "/"  << memory.free/1024/1024 << std::endl;
    }

    const unsigned memory_partition = sizeof(scalar_t)*(size_col+1)*size_partition/1024/1024;
    std::cout << "Required Memory (MiB) " << memory_partition << std::endl;

    //===============================================================================
    // Key: multiple devices are supported by device context
    //===============================================================================

    device_context::DeviceContext ctx0 = device_context::get_default_device_context();
    ctx0.device_id=0;
    device_context::DeviceContext ctx1 = device_context::get_default_device_context();
    ctx1.device_id=1;
    
    std::cout << "Allocate and initialize the memory for layers and hashes" << std::endl;
    scalar_t* layers0 = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    CHECK_ALLOC(layers0);
    scalar_t s = scalar_t::zero();
    for (unsigned i = 0; i < size_col*size_partition ; i++) {
        layers0[i] = s;
        s = s + scalar_t::one();
    }
    scalar_t* layers1 = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    CHECK_ALLOC(layers1);
    s = scalar_t::zero() + scalar_t::one();
    for (unsigned i = 0; i < size_col*size_partition ; i++) {
        layers1[i] = s;
        s = s + scalar_t::one();
    }

    scalar_t* column_hash0 = static_cast<scalar_t*>(malloc(size_partition * sizeof(scalar_t)));
    CHECK_ALLOC(column_hash0);
    scalar_t* column_hash1 = static_cast<scalar_t*>(malloc(size_partition * sizeof(scalar_t)));
    CHECK_ALLOC(column_hash1);

    PoseidonConstants<scalar_t> column_constants0, column_constants1;
    bn254InitOptimizedPoseidonConstants(size_col, ctx0, &column_constants0);
    cudaError_t err_result =  CHK_STICKY(cudaSetDevice(ctx1.device_id));
    if (err_result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err_result) << std::endl;
        return; 
    }
    bn254InitOptimizedPoseidonConstants(size_col, ctx1, &column_constants1);

    std::cout << "Parallel execution of Poseidon threads" << std::endl;
    START_TIMER(parallel);
    std::thread thread0(threadPoseidon, ctx0, size_partition, layers0, column_hash0, &column_constants0);
    std::thread thread1(threadPoseidon, ctx1, size_partition, layers1, column_hash1, &column_constants1);

    // Wait for the threads to finish
    thread0.join();
    thread1.join();
    END_TIMER(parallel,"2 GPUs");
    std::cout << "Output Data from Thread 0: ";
    std::cout << column_hash0[0] << std::endl;
    std::cout << "Output Data from Thread 1: ";
    std::cout << column_hash1[0] << std::endl;

    std::cout << "Sequential execution of Poseidon threads" << std::endl;
    START_TIMER(sequential);
    std::thread thread2(threadPoseidon, ctx0, size_partition, layers0, column_hash0, &column_constants0);
    thread2.join();
    std::thread thread3(threadPoseidon, ctx0, size_partition, layers1, column_hash1, &column_constants0);
    thread3.join();
    END_TIMER(sequential,"1 GPU");
    std::cout << "Output Data from Thread 2: ";
    std::cout << column_hash0[0] << std::endl;
    std::cout << "Output Data from Thread 3: ";
    std::cout << column_hash1[0] << std::endl;

    nvmlShutdown();
    return 0;
}
