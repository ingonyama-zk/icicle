#include <stdio.h>
#include <iostream>
#include "../../primitives/field.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "../../curves/curve_config.cuh"
#include "ve_mod_mult.cuh"


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

template <typename E, typename S>
__global__ void batchVectorMult(S *scalar_vec, E *element_vec, size_t n_scalars, size_t batch_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_scalars * batch_size)
    {
        int scalar_id = tid % n_scalars;
        element_vec[tid] = scalar_vec[scalar_id] * element_vec[tid];
    }
}

template <typename E, typename S>
int batch_vector_mult(S *scalar_vec, E *element_vec, size_t n_scalars, size_t batch_size)
{
    // Set the grid and block dimensions
    int NUM_THREADS = MAX_THREADS_PER_BLOCK;
    int NUM_BLOCKS = (n_scalars * batch_size + NUM_THREADS - 1) / NUM_THREADS;
    batchVectorMult<<<NUM_BLOCKS, NUM_THREADS>>>(scalar_vec, element_vec, n_scalars, batch_size);
    return 0;
}

template <typename E>
__global__ void matrixVectorMult(E *matrix_elements, E *vector_elements, E *result, size_t dim)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < dim)
    {
        result[tid] = E::zero();
        for (int i = 0; i < dim; i++)
            result[tid] = result[tid] + matrix_elements[tid * dim + i] * vector_elements[i];
    }
}

template <typename E>
int matrix_mod_mult(E *matrix_elements, E *vector_elements, E *result, size_t dim)
{
    // Set the grid and block dimensions
    int num_blocks = (int)ceil((float)dim / MAX_THREADS_PER_BLOCK);
    int threads_per_block = MAX_THREADS_PER_BLOCK;

    // Allocate memory on the device for the input vectors, the output vector, and the modulus
    E *d_matrix, *d_vector, *d_result;
    cudaMalloc(&d_matrix, (dim * dim) * sizeof(E));
    cudaMalloc(&d_vector, dim * sizeof(E));
    cudaMalloc(&d_result, dim * sizeof(E));

    // Copy the input vectors and the modulus from the host to the device
    cudaMemcpy(d_matrix, matrix_elements, (dim * dim) * sizeof(E), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector_elements, dim * sizeof(E), cudaMemcpyHostToDevice);

    // Call the kernel to perform element-wise modular multiplication
    matrixVectorMult<<<num_blocks, threads_per_block>>>(d_matrix, d_vector, d_result, dim);

    cudaMemcpy(result, d_result, dim * sizeof(E), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}

extern "C" int32_t vec_mod_mult_point(projective_t *inout,
                                      scalar_t *scalar_vec,
                                      size_t n_elments,
                                      size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<projective_t, scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_scalar(scalar_t *inout,
                                       scalar_t *scalar_vec,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<scalar_t, scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult(scalar_t *matrix_flattened,
                                       scalar_t *input,
                                       scalar_t *output,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    matrix_mod_mult<scalar_t>(matrix_flattened, input, output, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
