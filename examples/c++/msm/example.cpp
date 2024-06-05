#include <fstream>
#include <iostream>
#include <iomanip>

#include "icicle/runtime.h"

#include "icicle/api/bn254.h"
using namespace bn254;

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

void try_load_and_set_cuda_backend(const char* selected_device = nullptr)
{
  std::cout << "Trying to load and select backend device" << std::endl;
#ifdef BACKEND_DIR
  ICICLE_CHECK(icicle_load_backend(BACKEND_DIR, true));
#endif

  if (selected_device) {
    std::cout << "selecting " << selected_device << " device" << std::endl;
    ICICLE_CHECK(icicle_set_device(selected_device));
    return;
  }

  // trying to choose CUDA if available, or fallback to CPU otherwise (default device)
  const bool is_cuda_device_available = (eIcicleError::SUCCESS == icicle_is_device_avialable("CUDA"));
  if (is_cuda_device_available) {
    Device device = {"CUDA", 0}; // GPU-0
    std::cout << "setting " << device << std::endl;
    ICICLE_CHECK(icicle_set_device(device));
    return;
  }

  std::cout << "CUDA device not available, falling back to CPU" << std::endl;
}

int main(int argc, char* argv[])
{
  bool is_device_passed_as_param = (argc > 1);
  try_load_and_set_cuda_backend(is_device_passed_as_param ? argv[1] : nullptr);

  std::cout << "\nIcicle example: Muli-Scalar Multiplication (MSM)" << std::endl;
  std::cout << "Example parameters" << std::endl;

  int batch_size = 1;
  unsigned msm_size = 1 << 10;
  int N = batch_size * msm_size;
  std::cout << "Batch size: " << batch_size << std::endl;
  std::cout << "MSM size: " << msm_size << std::endl;

  std::cout << "\nPart I: use G1 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  auto scalars = std::make_unique<scalar_t[]>(N);
  auto points = std::make_unique<affine_t[]>(N);
  projective_t result;
  scalar_t::rand_host_many(scalars.get(), N);
  projective_t::rand_host_many(points.get(), N);

  std::cout << "Using default MSM configuration with on-host inputs" << std::endl;

  icicleStreamHandle stream;
  ICICLE_CHECK(icicle_create_stream(&stream));
  auto config = default_msm_config();
  config.batch_size = batch_size;
  config.stream = stream;

  std::cout << "\nRunning MSM kernel with on-host inputs" << std::endl;
  // Execute the MSM kernel
  START_TIMER(MSM_host_mem);
  ICICLE_CHECK(bn254_msm(scalars.get(), points.get(), msm_size, config, &result));
  END_TIMER(MSM_host_mem, "MSM from host-memory took");
  std::cout << projective_t::to_affine(result) << std::endl;

  DeviceProperties device_props;
  ICICLE_CHECK(icicle_get_device_properties(device_props));
  // If device does not share memory with host, copy inputs explicitly and execute msm with device pointers
  if (!device_props.using_host_memory) {
    std::cout << "\nReconfiguring MSM to use on-device inputs" << std::endl;
    config.are_results_on_device = true;
    config.are_scalars_on_device = true;
    config.are_points_on_device = true;

    std::cout << "Copying inputs to-device" << std::endl;
    scalar_t* scalars_d;
    affine_t* points_d;
    projective_t* result_d;

    ICICLE_CHECK(icicle_malloc((void**)&scalars_d, sizeof(scalar_t) * N));
    ICICLE_CHECK(icicle_malloc((void**)&points_d, sizeof(affine_t) * N));
    ICICLE_CHECK(icicle_malloc((void**)&result_d, sizeof(projective_t)));
    ICICLE_CHECK(icicle_copy(scalars_d, scalars.get(), sizeof(scalar_t) * N));
    ICICLE_CHECK(icicle_copy(points_d, points.get(), sizeof(affine_t) * N));

    std::cout << "Running MSM kernel with on-device inputs" << std::endl;
    // Execute the MSM kernel
    START_TIMER(MSM_device_mem);
    ICICLE_CHECK(msm(scalars_d, points_d, msm_size, config, result_d));
    END_TIMER(MSM_device_mem, "MSM from device-memory took");

    // Copy the result back to the host
    icicle_copy(&result, result_d, sizeof(projective_t));
    // Print the result
    std::cout << projective_t::to_affine(result) << std::endl;
    // Free the device memory
    icicle_free(scalars_d);
    icicle_free(points_d);
    icicle_free(result_d);
  }

  std::cout << "\nPart II: use G2 points" << std::endl;

  std::cout << "Generating random inputs on-host" << std::endl;
  // use the same scalars
  auto g2_points = std::make_unique<g2_affine_t[]>(N);
  g2_projective_t::rand_host_many(g2_points.get(), N);

  std::cout << "Reconfiguring MSM to use on-host inputs" << std::endl;
  config.are_results_on_device = false;
  config.are_scalars_on_device = false;
  config.are_points_on_device = false;
  g2_projective_t g2_result;
  START_TIMER(MSM_g2);
  ICICLE_CHECK(bn254_g2_msm(scalars.get(), g2_points.get(), msm_size, config, &g2_result));
  END_TIMER(MSM_g2, "MSM G2 from host-memory took");
  std::cout << g2_projective_t::to_affine(g2_result) << std::endl;

  // Similar to G1 MSM, can explicitly copy to device and execute the G2 MSM using device pointers

  return 0;
}