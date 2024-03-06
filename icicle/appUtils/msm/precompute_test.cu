#define CURVE_ID BN254

#include <chrono>
#include <iostream>
#include <vector>

#include "primitives/field.cu"
#include "curves/curve_config.cuh"
#include "primitives/field.cuh"
#include "primitives/projective.cuh"
#include "utils/device_context.cuh"
#include "utils/mont.cuh"

#include "msm.cu"

using namespace curve_config;

#include <iostream>
#include <fstream>

template <typename T>
void dumpArrayToFile(const T* array, std::size_t size, const std::string& filename)
{
  // Open the file in binary mode
}

int main(int argc, char* argv[])
{
  int batch_size = 1;
  unsigned log_size = argc > 2 ? atoi(argv[2]) : 15;
  unsigned msm_size = 1 << log_size;
  unsigned precompute_factor = argc > 1 ? atoi(argv[1]) : 2;
  int N = batch_size * msm_size;

  device_context::DeviceContext ctx = device_context::get_default_device_context();
  cudaStream_t& stream = ctx.stream;

  scalar_t* scalars = new scalar_t[N];
  affine_t* points = new affine_t[N];

  scalar_t::RandHostMany(scalars, N);
  projective_t::RandHostManyAffine(points, N);

  std::cout << "finished generating" << std::endl;
  std::cout << "Scalars" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << scalars[i] << std::endl;
    std::cout << scalars[msm_size - i - 1] << std::endl;
  }
  std::cout << "Points" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << points[i].x << std::endl;
    std::cout << points[msm_size - i - 1].x << std::endl;
  }

  projective_t large_res[batch_size];

  scalar_t* scalars_d;
  affine_t* points_d;

  cudaMalloc(&scalars_d, sizeof(scalar_t) * msm_size);
  cudaMalloc(&points_d, sizeof(affine_t) * msm_size * precompute_factor);
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * msm_size, cudaMemcpyHostToDevice);
  std::cout << "finished copying" << std::endl;

  cudaMemcpy(points_d, points, sizeof(affine_t) * msm_size, cudaMemcpyHostToDevice);
  mont::ToMontgomery(points_d, msm_size, stream, points_d);
  affine_t* points_mont = new affine_t[N];
  cudaMemcpy(points_mont, points_d, sizeof(affine_t) * msm_size, cudaMemcpyDeviceToHost);
  mont::FromMontgomery(points_d, msm_size, stream, points_d);

  std::ofstream s_file("scalars.bin", std::ios::binary);
  std::ofstream p_file("points.bin", std::ios::binary);
  for (int i = 0; i < msm_size; i++) {
    s_file.write(reinterpret_cast<const char*>(&(scalars[i])), sizeof(scalar_t));
  }
  for (int i = 0; i < msm_size; i++) {
    p_file.write(reinterpret_cast<const char*>(&(points[i].x)), sizeof(point_field_t));
  }
  s_file.close();
  p_file.close();
  dumpArrayToFile(points, msm_size, "points.bin");

  std::cout << "PointsMont" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << points_mont[i].x << std::endl;
  }
  std::cout << "..." << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << points_mont[msm_size - i - 1].x << std::endl;
  }

  msm::PrecomputeMSMBases<affine_t, projective_t>(points, msm_size, precompute_factor, false, points_d, ctx);

  affine_t* precomputed = new affine_t[msm_size * precompute_factor];
  cudaMemcpy(precomputed, points_d, sizeof(affine_t) * msm_size * precompute_factor, cudaMemcpyDeviceToHost);
  for (int i = 0; i < precompute_factor; i++) {
    std::cout << "Precompute part " << i << std::endl;
    for (int j = 0; j < 5; j++) {
      std::cout << precomputed[i * msm_size + j].x << std::endl;
    }
  }

  std::cout << "finished precomputation" << std::endl;

  msm::MSMConfig config = {
    ctx,               // DeviceContext
    0,                 // points_size
    precompute_factor, // precompute_factor
    0,                 // c
    0,                 // bitsize
    10,                // large_bucket_factor
    1,                 // batch_size
    true,              // are_scalars_on_device
    false,             // are_scalars_montgomery_form
    true,              // are_points_on_device
    false,             // are_points_montgomery_form
    false,             // are_results_on_device
    false,             // is_big_triangle
    false,             // is_async
  };

  auto begin1 = std::chrono::high_resolution_clock::now();
  msm::MSM<scalar_t, affine_t, projective_t>(scalars_d, points_d, msm_size, config, large_res);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);
  printf("No Big Triangle : %.3f seconds.\n", elapsed1.count() * 1e-9);
  std::cout << projective_t::to_affine(large_res[0]) << std::endl;

  config.precompute_factor = 1;
  msm::MSM<scalar_t, affine_t, projective_t>(scalars_d, points_d, msm_size, config, large_res);
  printf("No Big Triangle : %.3f seconds.\n", elapsed1.count() * 1e-9);
  std::cout << projective_t::to_affine(large_res[0]) << std::endl;

  config.precompute_factor = precompute_factor;
  config.is_big_triangle = true;
  config.are_results_on_device = false;
  auto begin = std::chrono::high_resolution_clock::now();
  msm::MSM<scalar_t, affine_t, projective_t>(scalars_d, points_d, msm_size, config, large_res);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Big Triangle: %.3f seconds.\n", elapsed.count() * 1e-9);

  std::cout << projective_t::to_affine(large_res[0]) << std::endl;

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return 0;
}
