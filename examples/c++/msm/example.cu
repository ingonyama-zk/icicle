#include <fstream>
#include <iostream>
#include <iomanip>

#include "api/bn254.h"
#include "api/bls12_377.h"

// using namespace bn254;
typedef bn254::scalar_t scalar_bn254;
typedef bn254::affine_t affine_bn254;
typedef bn254::g2_affine_t g2_affine_bn254;
typedef bn254::projective_t projective_bn254;
typedef bn254::g2_projective_t g2_projective_bn254;

typedef bls12_377::scalar_t scalar_bls12377;
typedef bls12_377::affine_t affine_bls12377;
typedef bls12_377::g2_affine_t g2_affine_bls12377;
typedef bls12_377::projective_t projective_bls12377;
typedef bls12_377::g2_projective_t g2_projective_bls12377;


int msm_bn254(int argc, char* argv[])
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
  scalar_bn254* scalars = new scalar_bn254[N];
  affine_bn254* points = new affine_bn254[N];
  projective_bn254 result;
  scalar_bn254::rand_host_many(scalars, N);
  projective_bn254::rand_host_many_affine(points, N);

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
  std::cout << projective_bn254::to_affine(result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  scalar_bn254* scalars_d;
  affine_bn254* points_d;
  projective_bn254* result_d;
  cudaMalloc(&scalars_d, sizeof(scalar_bn254) * N);
  cudaMalloc(&points_d, sizeof(affine_bn254) * N);
  cudaMalloc(&result_d, sizeof(projective_bn254));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_bn254) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(affine_bn254) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  // Execute the MSM kernel
  bn254_msm_cuda(scalars_d, points_d, msm_size, config, result_d);

  // Copy the result back to the host
  cudaMemcpy(&result, result_d, sizeof(projective_bn254), cudaMemcpyDeviceToHost);
  // Print the result
  std::cout << projective_bn254::to_affine(result) << std::endl;
  // Free the device memory
  cudaFree(scalars_d);
  cudaFree(points_d);
  cudaFree(result_d);
  // Free the host memory, keep scalars for G2 example
  delete[] points;

  std::cout << "Part II: use G2 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  // use the same scalars
  g2_affine_bn254* g2_points = new g2_affine_bn254[N];
  g2_projective_bn254::rand_host_many_affine(g2_points, N);

  std::cout << "Reconfiguring MSM to use on-host inputs" << std::endl;
  config.are_results_on_device = false;
  config.are_scalars_on_device = false;
  config.are_points_on_device = false;
  g2_projective_bn254 g2_result;
  bn254_g2_msm_cuda(scalars, g2_points, msm_size, config, &g2_result);
  std::cout << g2_projective_bn254::to_affine(g2_result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  g2_affine_bn254* g2_points_d;
  g2_projective_bn254* g2_result_d;
  cudaMalloc(&scalars_d, sizeof(scalar_bn254) * N);
  cudaMalloc(&g2_points_d, sizeof(g2_affine_bn254) * N);
  cudaMalloc(&g2_result_d, sizeof(g2_projective_bn254));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_bn254) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(g2_points_d, g2_points, sizeof(g2_affine_bn254) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  bn254_g2_msm_cuda(scalars_d, g2_points_d, msm_size, config, g2_result_d);
  cudaMemcpy(&g2_result, g2_result_d, sizeof(g2_projective_bn254), cudaMemcpyDeviceToHost);
  std::cout << g2_projective_bn254::to_affine(g2_result) << std::endl;

  cudaFree(scalars_d);
  cudaFree(g2_points_d);
  cudaFree(g2_result_d);
  delete[] g2_points;
  delete[] scalars;
  return 0;
}

int msm_bls12_377(int argc, char* argv[])
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
  scalar_bls12377* scalars = new scalar_bls12377[N];
  affine_bls12377* points = new affine_bls12377[N];
  projective_bls12377 result;
  scalar_bls12377::rand_host_many(scalars, N);
  projective_bls12377::rand_host_many_affine(points, N);

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
  cudaStreamCreate(&stream);
  // Execute the MSM kernel
  bls12_377_msm_cuda(scalars, points, msm_size, config, &result);
  std::cout << projective_bls12377::to_affine(result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  scalar_bls12377* scalars_d_bls;
  affine_bls12377* points_d_bls;
  projective_bls12377* result_d_bls;
  cudaMalloc(&scalars_d_bls, sizeof(scalar_bls12377) * N);
  cudaMalloc(&points_d_bls, sizeof(affine_bls12377) * N);
  cudaMalloc(&result_d_bls, sizeof(projective_bls12377));
  cudaMemcpy(scalars_d_bls, scalars, sizeof(scalar_bls12377) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d_bls, points, sizeof(affine_bls12377) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  // Execute the MSM kernel
  bls12_377_msm_cuda(scalars_d_bls, points_d_bls, msm_size, config, result_d_bls);

  // Copy the result back to the host
  cudaMemcpy(&result, result_d_bls, sizeof(projective_bls12377), cudaMemcpyDeviceToHost);
  // Print the result
  std::cout << projective_bls12377::to_affine(result) << std::endl;
  // Free the device memory
  cudaFree(scalars_d_bls);
  cudaFree(points_d_bls);
  cudaFree(result_d_bls);
  // Free the host memory, keep scalars for G2 example
  delete[] points;

  std::cout << "Part II: use G2 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  // use the same scalars
  g2_affine_bls12377* g2_points = new g2_affine_bls12377[N];
  g2_projective_bls12377::rand_host_many_affine(g2_points, N);

  std::cout << "Reconfiguring MSM to use on-host inputs" << std::endl;
  config.are_results_on_device = false;
  config.are_scalars_on_device = false;
  config.are_points_on_device = false;
  g2_projective_bls12377 g2_result;
  bls12_377_g2_msm_cuda(scalars, g2_points, msm_size, config, &g2_result);
  std::cout << g2_projective_bls12377::to_affine(g2_result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  g2_affine_bls12377* g2_points_d;
  g2_projective_bls12377* g2_result_d;
  cudaMalloc(&scalars_d_bls, sizeof(scalar_bls12377) * N);
  cudaMalloc(&g2_points_d, sizeof(g2_affine_bls12377) * N);
  cudaMalloc(&g2_result_d, sizeof(g2_projective_bls12377));
  cudaMemcpy(scalars_d_bls, scalars, sizeof(scalar_bls12377) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(g2_points_d, g2_points, sizeof(g2_affine_bls12377) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  bls12_377_g2_msm_cuda(scalars_d_bls, g2_points_d, msm_size, config, g2_result_d);
  cudaMemcpy(&g2_result, g2_result_d, sizeof(g2_projective_bn254), cudaMemcpyDeviceToHost);
  std::cout << g2_projective_bls12377::to_affine(g2_result) << std::endl;

  cudaFree(scalars_d_bls);
  cudaFree(g2_points_d);
  cudaFree(g2_result_d);
  delete[] g2_points;
  delete[] scalars;
  return 0;
}

int main(int argc, char* argv[])
{ 
  std::cout << "Starting BN254 MSM" << std::endl;
  msm_bn254(argc, argv);
  std::cout << "Starting BLS12-377 MSM" << std::endl;
  msm_bls12_377(argc, argv);
  return 0;
}
