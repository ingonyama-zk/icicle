#include <chrono>
#include <fstream>
#include <iostream>

// include MSM template
#include "appUtils/msm/msm.cu"
// select the curve
#include "curves/bn254/curve_config.cuh"

using namespace BN254;

int main(int argc, char* argv[])
{
  std::cout << "Icicle example: Muli-Scalar Multiplication (MSM)" << std::endl;

  std::cout << "Example parameters" << std::endl;
  unsigned msm_size = 1048576;
  std::cout << "msm_size: " << msm_size << std::endl;
  unsigned bucket_factor = 10;
  std::cout << "bucket_factor: " << bucket_factor << std::endl;
  std::cout << "Generating random inputs on-host" << std::endl;
  scalar_t* scalars = new scalar_t[msm_size];
  affine_t* points = new affine_t[msm_size];
  projective_t result;
  for (unsigned i = 0; i < msm_size; i++) {
    points[i] = (i % msm_size < 10) ? projective_t::to_affine(projective_t::rand_host()) : points[i - 10];
    scalars[i] = scalar_t::rand_host();
  }

  std::cout << "Preparing inputs on-device" << std::endl;
  scalar_t* scalars_d;
  affine_t* points_d;
  projective_t* result_d;
  cudaMalloc(&scalars_d, sizeof(scalar_t) * msm_size);
  cudaMalloc(&points_d, sizeof(affine_t) * msm_size);
  cudaMalloc(&result_d, sizeof(projective_t));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * msm_size, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(affine_t) * msm_size, cudaMemcpyHostToDevice);

  std::cout << "Running MSM on-device" << std::endl;
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  auto begin = std::chrono::high_resolution_clock::now();
  large_msm<scalar_t, projective_t, affine_t>(
    scalars_d, points_d, msm_size, result_d, true, false, bucket_factor, stream1);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("On-device runtime: %.3f seconds.\n", elapsed.count() * 1e-9);
  cudaStreamSynchronize(stream1);
  cudaStreamDestroy(stream1);
  cudaMemcpy(&result, result_d, sizeof(projective_t), cudaMemcpyDeviceToHost);
  std::cout << projective_t::to_affine(result) << std::endl;

  cudaFree(scalars_d);
  cudaFree(points_d);
  cudaFree(result_d);

  return 0;
}
