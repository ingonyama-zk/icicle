#include <iostream>
#include <thread>
#include <vector>

// select the curve (only 2 available so far)
#define CURVE_ID 2
#include "appUtils/poseidon/poseidon.cu"

using namespace poseidon;
using namespace curve_config;

void setCudaDevice(const unsigned device_id) {
    // Example function to set the CUDA device
    std::cout << "Setting CUDA device to " << device_id << std::endl;
    // cudaSetDevice(device_id);
}

// function that a thread will execute
void processData(device_context::DeviceContext ctx, const std::vector<int>& inputData, std::vector<int>& outputData) {
    PoseidonConstants<scalar_t> column_constants;
    int size_col = 11;
    init_optimized_poseidon_constants<scalar_t>(size_col, ctx, &column_constants);
    PoseidonConfig column_config = default_poseidon_config<scalar_t>(size_col+1);
    column_config.are_inputs_on_device = true;
    column_config.are_outputs_on_device = true;

    for (int num : inputData) {
        outputData.push_back(num * 2); // Example operation
    }
}

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // Handle the error, e.g., exit the program or throw an exception.
    }
}

int main() {
    const uint32_t size_col=11;
    const unsigned size_partition = 1024; // size_row / nof_partitions;
    // layers is allocated only for one partition, need to resuse for different partitions
    const uint32_t size_layers = size_col * size_partition; // size_col * size_row
    // Input data for each thread
    std::vector<int> inputData1 = {1, 2, 3, 4};
    std::vector<int> inputData2 = {5, 6, 7, 8};

    // Output data for each thread
    std::vector<int> outputData1, outputData2;


    // Multiple devices are supported by device context

    // setCudaDevice(device_id);
    cudaStream_t stream0, stream1;
    cudaError_t err;
    err = cudaStreamCreate(&stream0);
    checkCudaError(err);
    err = cudaStreamCreate(&stream1);
    checkCudaError(err);

    device_context::DeviceContext ctx0 = device_context::get_default_device_context();
    ctx0.device_id=0;
    device_context::DeviceContext ctx1 = device_context::get_default_device_context();
    ctx1.device_id=1;

    
    

    // Allocate and initialize memory for the layers
    // scalar_t* layers0 = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    // if (layers0 == nullptr) {
    //     std::cerr << "Memory allocation for 'layers' failed." << std::endl;
    // }
    // scalar_t s = scalar_t::zero();
    // for (unsigned i = 0; i < size_col*size_partition ; i++) {
    //     layers0[i] = s;
    //     s = s + scalar_t::one();
    // }
    // scalar_t* layers1 = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    // if (layers1 == nullptr) {
    //     std::cerr << "Memory allocation for 'layers' failed." << std::endl;
    // }
    // s = scalar_t::zero() + scalar_t::one();
    // for (unsigned i = 0; i < size_col*size_partition ; i++) {
    //     layers1[i] = s;
    //     s = s + scalar_t::one();
    // }



    // Start threads
    std::thread thread1(processData, ctx0, std::ref(inputData1), std::ref(outputData1));
    std::thread thread2(processData, ctx1, std::ref(inputData2), std::ref(outputData2));

    // Wait for the threads to finish
    thread1.join();
    thread2.join();

    // Process the output data (example: print the data)
    std::cout << "Output Data from Thread 1: ";
    for (int num : outputData1) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Output Data from Thread 2: ";
    for (int num : outputData2) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
