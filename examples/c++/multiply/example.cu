#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>

#include "api/bn254.h"
#include "api/bls12_377.h"
#include "vec_ops/vec_ops.cuh"

using namespace vec_ops;
// using namespace bn254;
typedef bn254::scalar_t T;

typedef  bls12_377::scalar_t T_bls;


int vector_mult_bn254(T* vec_b, T* vec_a, T* vec_result, size_t n_elments, device_context::DeviceContext ctx)
{
  vec_ops::VecOpsConfig config = vec_ops::DefaultVecOpsConfig();
  config.is_a_on_device = true;
  config.is_b_on_device = true;
  config.is_result_on_device = true;
  cudaError_t err = bn254_mul_cuda(vec_a, vec_b, n_elments, config, vec_result);
  if (err != cudaSuccess) {
    std::cerr << "Failed to multiply vectors - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  return 0;
}

int vector_mult_bls12377(T_bls* vec_b, T_bls* vec_a, T_bls* vec_result, size_t n_elments, device_context::DeviceContext ctx)
{
  vec_ops::VecOpsConfig config = vec_ops::DefaultVecOpsConfig();
  config.is_a_on_device = true;
  config.is_b_on_device = true;
  config.is_result_on_device = true;
  cudaError_t err = bls12_377_mul_cuda(vec_a, vec_b, n_elments, config, vec_result);
  if (err != cudaSuccess) {
    std::cerr << "Failed to multiply vectors - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  return 0;
}

int main(int argc, char** argv)
{
  const unsigned vector_size = 1 << 15;
  const unsigned repetitions = 1 ;

  cudaError_t err;
  nvmlInit();
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(0, &device); // for GPU 0
  std::cout << "Icicle-Examples: vector multiplications" << std::endl;
  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  if (nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE) == NVML_SUCCESS) {
    std::cout << "GPU Model: " << name << std::endl;
  } else {
    std::cerr << "Failed to get GPU model name." << std::endl;
  }
  unsigned power_limit;
  nvmlDeviceGetPowerManagementLimit(device, &power_limit);

  std::cout << "Vector size: " << vector_size << std::endl;
  std::cout << "Repetitions: " << repetitions << std::endl;
  std::cout << "Power limit: " << std::fixed << std::setprecision(3) << 1.0e-3 * power_limit << " W" << std::endl;

  unsigned int baseline_power;
  nvmlDeviceGetPowerUsage(device, &baseline_power);
  std::cout << "Baseline power: " << std::fixed << std::setprecision(3) << 1.0e-3 * baseline_power << " W" << std::endl;
  unsigned baseline_temperature;
  if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &baseline_temperature) == NVML_SUCCESS) {
    std::cout << "Baseline GPU Temperature: " << baseline_temperature << " C" << std::endl;
  } else {
    std::cerr << "Failed to get GPU temperature." << std::endl;
  }

  // host data
  T* host_in1 = (T*)malloc(vector_size * sizeof(T));
  T* host_in2 = (T*)malloc(vector_size * sizeof(T));
  T_bls* host_in1_bls12377 = (T_bls*)malloc(vector_size * sizeof(T_bls));
  T_bls* host_in2_bls12377 = (T_bls*)malloc(vector_size * sizeof(T_bls));
  std::cout << "Initializing vectors with random data" << std::endl;
  T::rand_host_many(host_in1, vector_size);
  T::rand_host_many(host_in2, vector_size);
  T_bls::rand_host_many(host_in1_bls12377, vector_size);
  T_bls::rand_host_many(host_in2_bls12377, vector_size);
  // device data
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  T* device_in1_bn254;
  T* device_in2_bn254;
  T* device_out_bn254;
  T_bls* device_in1_bls12377;
  T_bls* device_in2_bls12377;
  T_bls* device_out_bls12377;

  err = cudaMalloc((void**)&device_in1_bn254, vector_size * sizeof(T));
  err = cudaMalloc((void**)&device_in1_bls12377, vector_size * sizeof(T_bls));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  err = cudaMalloc((void**)&device_in2_bn254, vector_size * sizeof(T));
  err = cudaMalloc((void**)&device_in2_bls12377, vector_size * sizeof(T_bls));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  err = cudaMalloc((void**)&device_out_bn254, vector_size * sizeof(T));
  err = cudaMalloc((void**)&device_out_bls12377, vector_size * sizeof(T_bls));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  // copy from host to device
  err = cudaMemcpy(device_in1_bn254, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice);
  err = cudaMemcpy(device_in1_bls12377, host_in1_bls12377, vector_size * sizeof(T_bls), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  err = cudaMemcpy(device_in2_bn254, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice);
  err = cudaMemcpy(device_in2_bls12377, host_in2_bls12377, vector_size * sizeof(T_bls), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  std::cout << "Starting warm-up" << std::endl;
  // Warm-up loop
  for (int i = 0; i < repetitions; i++) {
    std::cout << "bn254 mult" << std::endl;
    vector_mult_bn254(device_in1_bn254, device_in2_bn254, device_out_bn254, vector_size, ctx);
    std::cout << "bls12-377 mult" << std::endl;
    vector_mult_bls12377(device_in1_bls12377, device_in2_bls12377, device_out_bls12377, vector_size, ctx);
  }

  std::cout << "Starting benchmarking" << std::endl;
  unsigned power_before;
  nvmlDeviceGetPowerUsage(device, &power_before);
  std::cout << "Power before: " << std::fixed << std::setprecision(3) << 1.0e-3 * power_before << " W" << std::endl;
  std::cout << "Power utilization: " << std::fixed << std::setprecision(1) << (float)100.0 * power_before / power_limit
            << " %" << std::endl;
  unsigned temperature_before;
  if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_before) == NVML_SUCCESS) {
    std::cout << "GPU Temperature before: " << temperature_before << " C" << std::endl;
  } else {
    std::cerr << "Failed to get GPU temperature." << std::endl;
  }
  auto start_time = std::chrono::high_resolution_clock::now();
  // Benchmark loop
  for (int i = 0; i < repetitions; i++) {
    vector_mult_bn254(device_in1_bn254, device_in2_bn254, device_out_bn254, vector_size, ctx);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  unsigned power_after;
  nvmlDeviceGetPowerUsage(device, &power_after);
  std::cout << "Power after: " << std::fixed << std::setprecision(3) << 1.0e-3 * power_after << " W" << std::endl;
  std::cout << "Power utilization: " << std::fixed << std::setprecision(1) << (float)100.0 * power_after / power_limit
            << " %" << std::endl;
  unsigned temperature_after;
  if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_after) == NVML_SUCCESS) {
    std::cout << "GPU Temperature after: " << temperature_after << " C" << std::endl;
  } else {
    std::cerr << "Failed to get GPU temperature." << std::endl;
  }

  // Report performance in GMPS: Giga Multiplications Per Second
  double GMPS = 1.0e-9 * repetitions * vector_size / (1.0e-6 * duration.count());
  std::cout << "Performance: " << GMPS << " Giga Multiplications Per Second" << std::endl;

  // Optional: validate multiplication
  T* host_out = (T*)malloc(vector_size * sizeof(T));

  cudaMemcpy(host_out, device_out_bn254, vector_size * sizeof(T), cudaMemcpyDeviceToHost);

  // validate multiplication here...

  // clean up and exit
  free(host_in1);
  free(host_in2);
  free(host_out);
  cudaFree(device_in1_bn254);
  cudaFree(device_in2_bn254);
  cudaFree(device_out_bn254);
  nvmlShutdown();
  return 0;
}