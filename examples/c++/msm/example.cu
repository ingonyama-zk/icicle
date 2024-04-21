#include <fstream>
#include <iostream>
#include <iomanip>

#include "api/bn254.h"
using namespace bn254;

int main(int argc, char* argv[])
{
  std::cout << "Icicle example: Muli-Scalar Multiplication (MSM)" << std::endl;
  std::cout << "Example parameters" << std::endl;
  int batch_size = 1;
  std::cout << "Batch size: " << batch_size << std::endl;
  unsigned msm_size = 1048576;
  std::cout << "MSM size: " << msm_size << std::endl;
  int N = batch_size * msm_size;

  std::cout << "Part I: use G1 points" << std::endl;
  
  std::cout << "Generating random inputs on-host" << std::endl;
  scalar_t* scalars = new scalar_t[N];
  affine_t* points = new affine_t[N];
  projective_t result;
  scalar_t::rand_host_many(scalars, N);
  projective_t::rand_host_many_affine(points, N);

  std::cout << "Using default MSM configuration with on-host inputs" << std::endl;
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  msm::MSMConfig config = {
    ctx,   // ctx
    0,     // points_size
    1,     // precompute_factor
    0,     // c
    0,     // bitsize
    10,    // large_bucket_factor
    1,     // batch_size
    false, // are_scalars_on_device
    false, // are_scalars_montgomery_form
    false, // are_points_on_device
    false, // are_points_montgomery_form
    false, // are_results_on_device
    false, // is_big_triangle
    false, // is_async
  };
  config.batch_size = batch_size;
  
  std::cout << "Running MSM kernel with on-host inputs" << std::endl;
  cudaStream_t stream = config.ctx.stream;
  // Execute the MSM kernel
  bn254_msm_cuda(scalars, points, msm_size, config, &result);
  std::cout << projective_t::to_affine(result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  scalar_t* scalars_d;
  affine_t* points_d;
  projective_t* result_d;
  cudaMalloc(&scalars_d, sizeof(scalar_t) * N);
  cudaMalloc(&points_d, sizeof(affine_t) * N);
  cudaMalloc(&result_d, sizeof(projective_t));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(affine_t) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  // Execute the MSM kernel
  bn254_msm_cuda(scalars_d, points_d, msm_size, config, result_d);

  // Copy the result back to the host
  cudaMemcpy(&result, result_d, sizeof(projective_t), cudaMemcpyDeviceToHost);
  // Print the result
  std::cout << projective_t::to_affine(result) << std::endl;
  // Free the device memory
  cudaFree(scalars_d);
  cudaFree(points_d);
  cudaFree(result_d);
  // Free the host memory, keep scalars for G2 example
  delete[] points;

  std::cout << "Part II: use G2 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  // use the same scalars
  g2_affine_t* g2_points = new g2_affine_t[N];
  g2_projective_t::rand_host_many_affine(g2_points, N);

  std::cout << "Reconfiguring MSM to use on-host inputs" << std::endl;
  config.are_results_on_device = false;
  config.are_scalars_on_device = false;
  config.are_points_on_device = false;
  g2_projective_t g2_result;
  bn254_g2_msm_cuda(scalars, g2_points, msm_size, config, &g2_result);
  std::cout << g2_projective_t::to_affine(g2_result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  g2_affine_t* g2_points_d;
  g2_projective_t* g2_result_d;
  cudaMalloc(&scalars_d, sizeof(scalar_t) * N);
  cudaMalloc(&g2_points_d, sizeof(g2_affine_t) * N);
  cudaMalloc(&g2_result_d, sizeof(g2_projective_t));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(g2_points_d, g2_points, sizeof(g2_affine_t) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  bn254_g2_msm_cuda(scalars_d, g2_points_d, msm_size, config, g2_result_d);
  cudaMemcpy(&g2_result, g2_result_d, sizeof(g2_projective_t), cudaMemcpyDeviceToHost);
  std::cout << g2_projective_t::to_affine(g2_result) << std::endl;

  cudaFree(scalars_d);
  cudaFree(g2_points_d);
  cudaFree(g2_result_d);
  delete[] g2_points;
  delete[] scalars;
  cudaStreamDestroy(stream);
  return 0;
}
