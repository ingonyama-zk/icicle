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

device_context::DeviceContext ctx= device_context::get_default_device_context();

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // Handle the error, e.g., exit the program or throw an exception.
    }
}

int main(int argc, char* argv[])
{
    // Outline of Filecoin PC2:
    // github.com/ingonyama-zk/research/blob/main/filecoin/doc/Filecoin.pdf

    std::cout << "Defining the size of the example" << std::endl;
    const uint32_t size_col=11;
    std::cout << "Number of layers: " << size_col << std::endl;
    const uint32_t height=9;
    const uint32_t height_icicle = height + 1;
    std::cout << "Tree height (edges, +1 to count levels): " << height <<  std::endl;
    const uint32_t tree_arity = 8;
    std::cout << "Tree arity: " << tree_arity << std::endl;
    const uint32_t size_row = pow(tree_arity,height); // (1<<(3*height));
    std::cout << "Tree width: " << size_row << std::endl;
 
    std::cout << "Allocating memory" << std::endl;
    scalar_t* layers = static_cast<scalar_t*>(malloc(size_col * size_row * sizeof(scalar_t)));
    scalar_t* column_hash = static_cast<scalar_t*>(malloc(size_row * sizeof(scalar_t)));


    std::cout << "Generating random inputs" << std::endl;
    scalar_t::RandHostMany(layers, size_col /* *size_row */);
    for (unsigned i = size_col; i < size_col*size_row; i++) {
        layers[i] = scalar_t::one();
    }
    std::cout << "Data generated" << std::endl;
    cudaError_t err;
    std::cout << "Step 1: Column Hashing" << std::endl;
    PoseidonConstants<scalar_t> constants1;
    init_optimized_poseidon_constants<scalar_t>(size_col, ctx, &constants1);
    PoseidonConfig config1 = default_poseidon_config<scalar_t>(size_col+1);
    err = poseidon_hash<curve_config::scalar_t, size_col+1>(layers, column_hash, size_row, constants1, config1);
    checkCudaError(err);
    std::cout << "Step 2: Merkle Tree-C" << std::endl;
    auto digests_len = get_digests_len<scalar_t>(height_icicle, tree_arity);  // keep all digests
    // std::cout << "Digests length: " << digests_len << std::endl;
    scalar_t* digests = static_cast<scalar_t*>(malloc(digests_len * sizeof(scalar_t)));
    TreeBuilderConfig tree_config = default_merkle_config<scalar_t>(); // default: keep all digest raws
    PoseidonConstants<scalar_t> tree_constants;
    init_optimized_poseidon_constants<scalar_t>(tree_arity, ctx, &tree_constants);
    err = build_merkle_tree<scalar_t, tree_arity+1>(column_hash, digests, height_icicle, tree_constants, tree_config);
    checkCudaError(err);

    std::cout << "Cleaning up memory" << std::endl;
    free(layers);
    free(column_hash);
    free(digests);

    return 0;
}