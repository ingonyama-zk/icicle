#include <chrono>
#include <fstream>
#include <iostream>

// select the curve (only 2 available so far)
#define CURVE_ID 2
#include "appUtils/poseidon/poseidon.cu"
#include "appUtils/tree/merkle.cu"

using namespace poseidon;
using namespace merkle;
using namespace curve_config;
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

device_context::DeviceContext ctx= device_context::get_default_device_context();

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // Handle the error, e.g., exit the program or throw an exception.
    }
}
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

int main(int argc, char* argv[])
{
    // Outline of Filecoin PC2:
    // github.com/ingonyama-zk/research/blob/main/filecoin/doc/Filecoin.pdf

    cudaError_t err;

    std::cout << "Defining the size of the example" << std::endl;
    const uint32_t size_col=11;
    std::cout << "Number of layers: " << size_col << std::endl;
    const uint32_t height=10;
    const uint32_t height_icicle = height + 1;
    std::cout << "Tree height (edges, +1 to count levels): " << height <<  std::endl;
    const uint32_t tree_arity = 8;
    std::cout << "Tree arity: " << tree_arity << std::endl;
    const uint32_t size_row = pow(tree_arity,height); // (1<<(3*height));
    std::cout << "Tree width: " << size_row << std::endl;
    const unsigned nof_partitions = 64;
    const unsigned size_partition = size_row / nof_partitions;
    std::cout << "Using " <<  nof_partitions <<  " partitions of size " << size_partition <<std::endl;

    std::cout << "Allocating on-host memory" << std::endl;
    // layers is allocated only for one partition, need to resuse for different partitions
    const uint32_t size_layers = size_col * size_partition; // size_col * size_row
    std::cout << "Memory for partitioned layers (GiB): " << size_layers * sizeof(scalar_t) / 1024 / 1024 / 1024 << std::endl;
    scalar_t* layers = static_cast<scalar_t*>(malloc(size_layers * sizeof(scalar_t)));
    if (layers == nullptr) {
        std::cerr << "Memory allocation for 'layers' failed." << std::endl;
    }
    std::cout << "Size of column_hash (GiB): " << size_row * sizeof(scalar_t) / 1024 / 1024 / 1024 << std::endl;
    scalar_t* column_hash = static_cast<scalar_t*>(malloc(size_row * sizeof(scalar_t)));
    if (column_hash == nullptr) {
        std::cerr << "Memory allocation for 'column_hash' failed." << std::endl;
    }

    std::cout << "Allocating on-device memory" << std::endl;
    scalar_t* layers_d;
    err = cudaMalloc(&layers_d, sizeof(scalar_t) * size_layers );
    checkCudaError(err);
    scalar_t* column_hash_d;
    // on-device memory for column_hash_d is allocated for one partition, need to resuse for different partitions
    err = cudaMalloc(&column_hash_d, sizeof(scalar_t) * size_partition);
    checkCudaError(err);

    std::cout << "Generating random inputs" << std::endl;
    // scalar_t::RandHostMany(layers, size_col /* *size_row */);
    scalar_t s = scalar_t::zero();
    for (unsigned i = 0; i < size_col*size_partition /* size_raw */; i++) {
        layers[i] = s;
        s = s + scalar_t::one();
    }

    std::cout << "Moving inputs to device" << std::endl;
    START_TIMER(copy);
    cudaMemcpy(layers_d, layers, sizeof(scalar_t) * size_layers, cudaMemcpyHostToDevice);
    END_TIMER(copy, "Copy to device");

    std::cout << "Step 1: Column Hashing" << std::endl;
    START_TIMER(step1);
    PoseidonConstants<scalar_t> column_constants;
    init_optimized_poseidon_constants<scalar_t>(size_col, ctx, &column_constants);
    PoseidonConfig column_config = default_poseidon_config<scalar_t>(size_col+1);
    column_config.are_inputs_on_device = true;
    column_config.are_outputs_on_device = true;
    for (unsigned i = 0; i < nof_partitions; i++) {
        std::cout << "Hashing partition " <<  i << std::endl;
        // while debuging, use the same inputs for different partitions
        START_TIMER(hash_partition);
        err = poseidon_hash<curve_config::scalar_t, size_col+1>(layers_d, column_hash_d, size_partition, column_constants, column_config);
        checkCudaError(err);
        END_TIMER(hash_partition, "Hash partition");
        cudaMemcpy(column_hash, column_hash_d, sizeof(scalar_t) * size_partition, cudaMemcpyDeviceToHost);
        std::cout << "First 10 hashes: " << column_hash[0] << ", " << column_hash[1] << std::endl;
    }
    free(layers);
    cudaFree(layers_d);
    END_TIMER(step1, "Step1");
    
    std::cout << "Step 2: Merkle Tree-C" << std::endl;
    START_TIMER(step2);
    auto digests_len = get_digests_len<scalar_t>(height_icicle, tree_arity);  // keep all digests
    std::cout << "Digests length: " << digests_len << std::endl;
    scalar_t* digests = static_cast<scalar_t*>(malloc(digests_len * sizeof(scalar_t)));
    if (digests == nullptr) {
        std::cerr << "Memory allocation for 'digests' failed." << std::endl;
    }
    TreeBuilderConfig tree_config = default_merkle_config<scalar_t>(); // default: keep all digest raws
    tree_config.keep_rows=1;
    PoseidonConstants<scalar_t> tree_constants;
    init_optimized_poseidon_constants<scalar_t>(tree_arity, ctx, &tree_constants);
    err = build_merkle_tree<scalar_t, tree_arity+1>(column_hash, digests, height_icicle, tree_constants, tree_config);
    checkCudaError(err);
    END_TIMER(step2, "Step2");

    std::cout << "Cleaning up memory" << std::endl;
    
    free(column_hash);
    free(digests);

    return 0;
}
