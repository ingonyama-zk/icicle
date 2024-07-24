#include <fstream>
#include <iostream>
#include <iomanip>

#include "icicle/runtime.h"

#include "icicle/api/bn254.h"
using namespace bn254;

int main(int argc, char* argv[])
{
  std::cout << "Icicle example: Muli-Scalar Multiplication (MSM)" << std::endl;
  std::cout << "Example parameters" << std::endl;

  int batch_size = 1;
  unsigned msm_size = 1 << 5;
  int N = batch_size * msm_size;
  std::cout << "Batch size: " << batch_size << std::endl;
  std::cout << "MSM size: " << msm_size << std::endl;

  std::cout << "Part I: use G1 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  auto scalars = std::make_unique<scalar_t[]>(N);
  auto points = std::make_unique<affine_t[]>(N);
  projective_t result;
  scalar_t::rand_host_many(scalars.get(), N);
  projective_t::rand_host_many(points.get(), N);

  std::cout << "Using default MSM configuration with on-host inputs" << std::endl;

  Device device = "CPU";
  std::cout << "Setting device: " << device << std::endl;
  ICICLE_CHECK(icicle_set_device(device));

  icicleStreamHandle stream;
  ICICLE_CHECK(icicle_create_stream(&stream));
  auto config = default_msm_config();
  config.batch_size = batch_size;
  config.stream = stream;

  std::cout << "Running MSM kernel with on-host inputs" << std::endl;
  // Execute the MSM kernel
  ICICLE_CHECK(bn254_msm(scalars.get(), points.get(), msm_size, config, &result));
  std::cout << projective_t::to_affine(result) << std::endl;

  DeviceProperties device_props;
  ICICLE_CHECK(icicle_get_device_properties(device_props));
  // If device does not share memory with host, copy inputs explicitly and execute msm with device pointers
  if (!device_props.using_host_memory) {
    std::cout << "Copying inputs to-device" << std::endl;
    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* result_d;

    ICICLE_CHECK(icicle_malloc((void**)&scalars_d, sizeof(scalar_t) * N));
    ICICLE_CHECK(icicle_malloc((void**)&points_d, sizeof(affine_t) * N));
    ICICLE_CHECK(icicle_malloc((void**)&result_d, sizeof(projective_t)));
    ICICLE_CHECK(icicle_copy(scalars_d, scalars.get(), sizeof(scalar_t) * N));
    ICICLE_CHECK(icicle_copy(points_d, points.get(), sizeof(affine_t) * N));

    std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
    config.are_results_on_device = true;
    config.are_scalars_on_device = true;
    config.are_points_on_device = true;

    std::cout << "Running MSM kernel with on-device inputs" << std::endl;
    // Execute the MSM kernel
    ICICLE_CHECK(msm(scalars_d, points_d, msm_size, config, result_d));

    // Copy the result back to the host
    icicle_copy(&result, result_d, sizeof(projective_t));
    // Print the result
    std::cout << projective_t::to_affine(result) << std::endl;
    // Free the device memory
    icicle_free(scalars_d);
    icicle_free(points_d);
    icicle_free(result_d);
  }

  std::cout << "Part II: use G2 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  // use the same scalars
  auto g2_points = std::make_unique<g2_affine_t[]>(N);
  g2_projective_t::rand_host_many(g2_points.get(), N);

  std::cout << "Reconfiguring MSM to use on-host inputs" << std::endl;
  config.are_results_on_device = false;
  config.are_scalars_on_device = false;
  config.are_points_on_device = false;
  g2_projective_t g2_result;
  // WHY isn't it linking with msm ?
  // msm(scalars.get(), g2_points.get(), msm_size, config, &g2_result);
  ICICLE_CHECK(bn254_g2_msm(scalars.get(), g2_points.get(), msm_size, config, &g2_result));
  std::cout << g2_projective_t::to_affine(g2_result) << std::endl;

  // Similar to G1 MSM, can explicitly copy to device and execute the G2 MSM using device pointers

  return 0;
}
