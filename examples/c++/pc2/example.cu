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

int main(int argc, char* argv[])
{
    // Outline of Filecoin PC2:
    // github.com/ingonyama-zk/research/blob/main/filecoin/doc/Filecoin.pdf

    std::cout << "Defining the size of the example" << std::endl;
    const uint32_t size_col=11;
    std::cout << "Number of layers: " << size_col << std::endl;
    const uint32_t height=8; 
    std::cout << "Tree height: " << height << " (+1 to include the root)" << std::endl;
    const uint32_t tree_arity = 8;
    std::cout << "Tree arity: " << tree_arity << std::endl;
    const uint32_t size_row = pow(tree_arity,height); // (1<<(3*height));
    std::cout << "Tree width: " << size_row << std::endl;

    std::cout << "Allocating memory" << std::endl;
    scalar_t* layers = static_cast<scalar_t*>(malloc(size_col * size_row * sizeof(scalar_t)));
    scalar_t* column_hash = static_cast<scalar_t*>(malloc(size_row * sizeof(scalar_t)));


    std::cout << "Generating random inputs" << std::endl;
    for( unsigned int i = 0; i < size_col*size_row; i++)
    {
        layers[i] = scalar_t::rand_host();
    }
    std::cout << "Data generated" << std::endl;
    
    std::cout << "Step 1: Column Hashing" << std::endl;
    PoseidonConstants<scalar_t> constants1;
    init_optimized_poseidon_constants<scalar_t>(size_col, ctx, &constants1);
    PoseidonConfig config1 = default_poseidon_config<scalar_t>(size_col+1);
    poseidon_hash<curve_config::scalar_t, size_col+1>(layers, column_hash, size_row, constants1, config1);

    std::cout << "Step 2: Merkle Tree-C" << std::endl;
    auto digests_len = get_digests_len<scalar_t>(height+1, tree_arity);  // don't keep the leaves
    std::cout << "Digests length: " << digests_len << std::endl;
    scalar_t* digests = static_cast<scalar_t*>(malloc(digests_len * sizeof(scalar_t)));
    TreeBuilderConfig tree_config = default_merkle_config<scalar_t>();
    tree_config.keep_rows = height;
    PoseidonConstants<scalar_t> tree_constants;
    init_optimized_poseidon_constants<scalar_t>(tree_arity, ctx, &tree_constants);
    build_merkle_tree<scalar_t, tree_arity+1>(column_hash, digests, height, tree_constants, tree_config);

    std::cout << "Cleaning up memory" << std::endl;
    free(layers);
    free(column_hash);
    free(digests);

    return 0;
}