#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>

#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "utils/device_context.cuh"
#include "utils/vec_ops.cu"

using namespace curve_config;

// select scalar or point field
//typedef scalar_t T;
typedef point_field_t T;

int vector_mult(T* vec_b, T* vec_a, T* vec_result, size_t n_elments, device_context::DeviceContext ctx)
{
  const bool is_on_device = true;
  const bool is_montgomery = false;
  cudaError_t err =  vec_ops::Mul<T,T>(vec_a, vec_b, n_elments, is_on_device, is_montgomery, ctx, vec_result);
  if (err != cudaSuccess) {
    std::cerr << "Failed to multiply vectors - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  return 0;
}

int main(int argc, char** argv)
{
  const unsigned vector_size = 1 << 15;
  const unsigned repetitions = 1 << 15;

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
  std::cout << "Initializing vectors with random data" << std::endl;
  T::RandHostMany(host_in1, vector_size);
  T::RandHostMany(host_in2, vector_size);
  // device data
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  T* device_in1;
  T* device_in2;
  T* device_out;

  err = cudaMalloc((void**)&device_in1, vector_size * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  err = cudaMalloc((void**)&device_in2, vector_size * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  err = cudaMalloc((void**)&device_out, vector_size * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  // copy from host to device
  err = cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  err = cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  
  std::cout << "Starting warm-up" << std::endl;
  // Warm-up loop
  for (int i = 0; i < repetitions; i++) {
    vector_mult(device_in1, device_in2, device_out, vector_size, ctx);
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
    vector_mult(device_in1, device_in2, device_out, vector_size, ctx);
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

  cudaMemcpy(host_out, device_out, vector_size * sizeof(T), cudaMemcpyDeviceToHost);

  // validate multiplication here...

  // clean up and exit
  free(host_in1); 
  free(host_in2);
  free(host_out);
  cudaFree(device_in1);
  cudaFree(device_in2);
  cudaFree(device_out);
  nvmlShutdown();
  return 0;
}