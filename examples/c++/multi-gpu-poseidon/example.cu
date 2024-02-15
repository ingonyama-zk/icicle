#include <iostream>
#include <thread>
#include <vector>

// select the curve
#define CURVE_ID 2
#include "appUtils/poseidon/poseidon.cu"

using namespace poseidon;
using namespace curve_config;

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // Handle the error, e.g., exit the program or throw an exception.
    }
}

// these global varibales go into template calls
const int size_col = 11;

// this function executes the Poseidon thread
void threadPoseidon(device_context::DeviceContext ctx, unsigned size_partition, scalar_t * layers, scalar_t * column_hashes) {
    PoseidonConstants<scalar_t> column_constants;
    init_optimized_poseidon_constants<scalar_t>(size_col, ctx, &column_constants);
    PoseidonConfig column_config = default_poseidon_config<scalar_t>(size_col+1);
    cudaError_t err = poseidon_hash<scalar_t, size_col+1>(layers, column_hashes, (size_t) size_partition, column_constants, column_config);
    checkCudaError(err);
}

int main() {
    const uint32_t size_col=11;
    const unsigned size_row = (1<<30);
    const unsigned nof_partitions = 64;
    const unsigned size_partition = size_row / nof_partitions;
    // layers is allocated only for one partition, need to resuse for different partitions
    const uint32_t size_layers = size_col * size_partition; // size_col * size_row
    
    // Key: multiple devices are supported by device context

    device_context::DeviceContext ctx0 = device_context::get_default_device_context();
    ctx0.device_id=0;
    device_context::DeviceContext ctx1 = device_context::get_default_device_context();
    ctx1.device_id=1;
    
    // Allocate and initialize memory for the layers and hashes
    scalar_t* layers0 = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    if (layers0 == nullptr) {
        std::cerr << "Memory allocation for 'layers' failed." << std::endl;
    }
    scalar_t s = scalar_t::zero();
    for (unsigned i = 0; i < size_col*size_partition ; i++) {
        layers0[i] = s;
        s = s + scalar_t::one();
    }
    scalar_t* layers1 = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    if (layers1 == nullptr) {
        std::cerr << "Memory allocation for 'layers' failed." << std::endl;
    }
    s = scalar_t::zero() + scalar_t::one();
    for (unsigned i = 0; i < size_col*size_partition ; i++) {
        layers1[i] = s;
        s = s + scalar_t::one();
    }

    scalar_t* column_hash0 = static_cast<scalar_t*>(malloc(size_partition * sizeof(scalar_t)));
    scalar_t* column_hash1 = static_cast<scalar_t*>(malloc(size_partition * sizeof(scalar_t)));

    // Start threads
    std::thread thread0(threadPoseidon, ctx0, size_partition, layers0, column_hash0);
    std::thread thread1(threadPoseidon, ctx1, size_partition, layers1, column_hash1);

    // Wait for the threads to finish
    thread0.join();
    thread1.join();

    // Process the output data (example: print the data)
    std::cout << "Output Data from Thread 0: ";
    std::cout << std::endl;

    std::cout << "Output Data from Thread 1: ";
    std::cout << std::endl;

    return 0;
}
