#include <stdio.h>
#include <iostream>
#include "../../primitives/base_curve.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "../../curves/curve_config.cuh"
#include "ve_mod_mult.cuh"


/// TESTING 
#define ELEMENTS_SIZE 8192
#define MAX_THREADS_PER_BLOCK 256

// TODO: headers for prototypes and .c .cpp .cu files for implementations
template <typename E, typename S>
__global__ void vectorModMult(S *scalar_vec, E *element_vec, E *result, size_t n_elments)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elments)
    {
        result[tid] = scalar_vec[tid] * element_vec[tid];
    }
}

template <typename E, typename S>
int vector_mod_mult(S *vec_a, E *vec_b, E *result, size_t n_elments) // TODO: in place so no need for third result vector
{
    // Set the grid and block dimensions
    int num_blocks = (int)ceil((float)n_elments / MAX_THREADS_PER_BLOCK);
    int threads_per_block = MAX_THREADS_PER_BLOCK;

    // Allocate memory on the device for the input vectors, the output vector, and the modulus
    S *d_vec_a;
    E *d_vec_b, *d_result;
    cudaMalloc(&d_vec_a, n_elments * sizeof(S));
    cudaMalloc(&d_vec_b, n_elments * sizeof(E));
    cudaMalloc(&d_result, n_elments * sizeof(E));

    // Copy the input vectors and the modulus from the host to the device
    cudaMemcpy(d_vec_a, vec_a, n_elments * sizeof(S), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_b, vec_b, n_elments * sizeof(E), cudaMemcpyHostToDevice);

    // Call the kernel to perform element-wise modular multiplication
    vectorModMult<<<num_blocks, threads_per_block>>>(d_vec_a, d_vec_b, d_result, n_elments);

    cudaMemcpy(result, d_result, n_elments * sizeof(E), cudaMemcpyDeviceToHost);

    cudaFree(d_vec_a);
    cudaFree(d_vec_b);
    cudaFree(d_result);

    return 0;
}


int main()
{
    // Allocate memory on the host for two input vectors, the output vector, and the modulus
    scalar_t *vec_a = (scalar_t*)malloc(ELEMENTS_SIZE * sizeof(scalar_t));
    projective_t *vec_b = (projective_t*)malloc(ELEMENTS_SIZE * sizeof(projective_t));
    projective_t *result = (projective_t*)malloc(ELEMENTS_SIZE * sizeof(projective_t));
    projective_t some_point; // ONE
    affine_t *res_affine = (affine_t*)malloc(ELEMENTS_SIZE * sizeof(affine_t));
    
    // Initialize the input vectors
    some_point.x = {0};
    some_point.y = {2};
    some_point.z = {1};

    vec_a[ELEMENTS_SIZE - 1] = {0};  // test correctness for last point is zero
    
    // Set the grid and block dimensions
    int num_blocks = (int)ceil((float)ELEMENTS_SIZE/MAX_THREADS_PER_BLOCK);
    int threads_per_block = MAX_THREADS_PER_BLOCK;
    
    // Allocate memory on the device for the input vectors, the output vector, and the modulus
    scalar_t *d_vec_a;
    projective_t *d_vec_b, *d_result;
    cudaMalloc(&d_vec_a, ELEMENTS_SIZE * sizeof(scalar_t));
    cudaMalloc(&d_vec_b, ELEMENTS_SIZE * sizeof(projective_t));
    cudaMalloc(&d_result, ELEMENTS_SIZE * sizeof(projective_t));
    
    // // Copy the input vectors and the modulus from the host to the device
    cudaMemcpy(d_vec_a, vec_a, ELEMENTS_SIZE * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_b, vec_b, ELEMENTS_SIZE * sizeof(projective_t), cudaMemcpyHostToDevice);
    
    // // Call the kernel to perform element-wise modular multiplication
    vectorModMult< projective_t, scalar_t><<<num_blocks, threads_per_block>>>(d_vec_a, d_vec_b, d_result, ELEMENTS_SIZE);
    
    cudaMemcpy(result, d_result, ELEMENTS_SIZE * sizeof(projective_t), cudaMemcpyDeviceToHost);    
    

    cudaFree(d_vec_a);
    cudaFree(d_vec_b);
    cudaFree(d_result);

}
