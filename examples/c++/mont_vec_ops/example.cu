#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>

#include "api/bn254.h"
#include "vec_ops/vec_ops.cuh"
#include <vec_ops/../../include/utils/mont.cuh>

using namespace vec_ops;
using namespace bn254;

typedef scalar_t T;

enum Op { MUL, ADD, SUB, LAST };

// bn254 p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

int vector_op(
  T* vec_a,
  T* vec_b,
  T* vec_result,
  size_t n_elements,
  device_context::DeviceContext ctx,
  vec_ops::VecOpsConfig config,
  Op op)
{
  cudaError_t err;
  switch (op) {
  case MUL:
    err = bn254_mul_cuda(vec_a, vec_b, n_elements, config, vec_result);
    break;
  case ADD:
    err = bn254_add_cuda(vec_a, vec_b, n_elements, config, vec_result);
    break;
  case SUB:
    err = bn254_sub_cuda(vec_a, vec_b, n_elements, config, vec_result);
    break;
  }
  // cudaError_t err = bn254_mul_cuda(vec_a, vec_b, n_elements, config, vec_result);
  if (err != cudaSuccess) {
    std::cerr << "Failed to multiply vectors - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  return 0;
}

int main(int argc, char** argv)
{
  const unsigned vector_size              = 1 << 11;
  const unsigned not_in_place_repetitions = 1 << 0;    // Repetitions are used only for the non in-place tests.
  const unsigned in_place_repetitions     = 1;    // Repetitions for in-place tests should be 1. Don't check it.

  cudaError_t err;
  nvmlInit();
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(0, &device); // for GPU 0
  std::cout << "Icicle-Examples: vector mul / add / sub operations." << std::endl;
  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  if (nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE) == NVML_SUCCESS) {
    std::cout << "GPU Model: " << name << std::endl;
  } else {
    std::cerr << "Failed to get GPU model name." << std::endl;
  }
  unsigned power_limit;
  nvmlDeviceGetPowerManagementLimit(device, &power_limit);

  std::cout << "Vector size:              " << vector_size << std::endl;
  std::cout << "not_in_place_repetitions: " << not_in_place_repetitions << std::endl;
  std::cout << "in_place_repetitions:     " << in_place_repetitions << std::endl;
  std::cout << "Power limit:              " << std::fixed << std::setprecision(3) << 1.0e-3 * power_limit << " W" << std::endl;

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
  std::cout << "Allocate memory for the input vectors (both normal and Montgomery presentation)" << std::endl;
  T* host_in1_init = (T*)malloc(vector_size * sizeof(T));
  std::cout << "example malloc host_in1_init" << std::endl;
  T* host_in2_init = (T*)malloc(vector_size * sizeof(T));
  std::cout << "example malloc host_in2_init" << std::endl;
  std::cout << "Initializing vectors with normal presentation random data" << std::endl;
  T::rand_host_many(host_in1_init, vector_size);
  T::rand_host_many(host_in2_init, vector_size);
  std::cout << "Allocate memory for the output vectors" << std::endl;
  T* host_out = (T*)malloc(vector_size * sizeof(T)); // This memory will be used for the test output.
  std::cout << "example malloc host_out" << std::endl;
  T* host_out_ref_mul = (T*)malloc(
    vector_size *
    sizeof(T)); // This memory will be used as a reference result for mul (will be compared to host_out content).
  std::cout << "example malloc host_out_ref_mul" << std::endl;
  T* host_out_ref_add = (T*)malloc(
    vector_size *
    sizeof(T)); // This memory will be used as a reference result for add (will be compared to host_out content).
  std::cout << "example malloc host_out_ref_add" << std::endl;    
  T* host_out_ref_sub = (T*)malloc(
    vector_size *
    sizeof(T)); // This memory will be used as a reference result for sub (will be compared to host_out content).
  std::cout << "malloc host_out_ref_sub" << std::endl;
  std::cout << "Initializing output vectors with random data" << std::endl;
  T::rand_host_many(host_out, vector_size);
  T::rand_host_many(host_out_ref_mul, vector_size);
  T::rand_host_many(host_out_ref_add, vector_size);
  T::rand_host_many(host_out_ref_sub, vector_size);
  // device data
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  T* device_in1;
  T* device_in2;
  T* device_out;

  err = cudaMalloc((void**)&device_in1, vector_size * sizeof(T));
  std::cout << "example cudaMalloc device_in1" << std::endl;
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  err = cudaMalloc((void**)&device_in2, vector_size * sizeof(T));
  std::cout << "example cudaMalloc device_in2" << std::endl;
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  err = cudaMalloc((void**)&device_out, vector_size * sizeof(T));
  std::cout << "example cudaMalloc device_out" << std::endl;
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  vec_ops::VecOpsConfig config = vec_ops::DefaultVecOpsConfig();
  int nof_of_configs_for_test = 5;
  int nof_of_storage_configs = 3;   // 2 inputs, 1 result.

  //****************************************
  // Test warn-up and reference output config. Reference output to be used to check if test passed or not.
  //****************************************
  // copy from host to device
  err = cudaMemcpy(device_in1, host_in1_init, vector_size * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  err = cudaMemcpy(device_in2, host_in2_init, vector_size * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  std::cout << "Starting warm-up run" << std::endl;
  // Warm-up loop
  for (int op = MUL; op != LAST; op++) {
    for (int i = 0; i < not_in_place_repetitions; i++) {
      vector_op(device_in1, device_in2, device_out, vector_size, ctx, config, (Op)op);
      switch (op) {
        case MUL:
          err = cudaMemcpy(host_out_ref_mul, device_out, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          break;
        case ADD:
          err = cudaMemcpy(host_out_ref_add, device_out, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          break;
        case SUB:
          err = cudaMemcpy(host_out_ref_sub, device_out, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          break;
      }
      if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from device_out to host - " << cudaGetErrorString(err) << std::endl;
        return 0;
      }
    }
  }
  //****************************************
  // End of test warn-up and reference output config.
  //****************************************

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

  //*******************************************************
  // Benchmark test:
  // Loop for (mul, add, sub):
  //   Loop (is_a_on_device, is_b_on_device, is_result_on_device, is_input_in_montgomery_form):
  //*******************************************************
  T* host_in1 =
    (T*)malloc(vector_size * sizeof(T)); // This buffer is used to load the data from host_in1_init for the benchmark.
  std::cout << "example malloc host_in1" << std::endl;
  T* host_in2 =
    (T*)malloc(vector_size * sizeof(T)); // This buffer is used to load the data from host_in2_init for the benchmark.
  std::cout << "example malloc host_in1" << std::endl;
  // Test when the result is not in-place
  std::cout << "*****************************************" << std::endl;
  std::cout << "*** Start not in-place benchmark loop ***" << std::endl;
  std::cout << "*****************************************" << std::endl;
  for (int op = MUL; op != LAST; op++) {
    for (int config_idx = 0; config_idx < 32; config_idx++) {
      // // DEBUG
      // int num_gpus;
      // size_t free, total;
      // cudaGetDeviceCount( &num_gpus );
      // std::cout << "num_gpus " << num_gpus << std::endl;
      // for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
      //     cudaSetDevice( gpu_id );
      //     int id;
      //     cudaGetDevice( &id );
      //     cudaMemGetInfo( &free, &total );
      //     std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
      // }
      // // DEBUG
      switch (op) {
        case MUL: std::cout << "Start benchmark loop for op MUL config_idx " << config_idx << " not in-place" << std::endl; break;
        case ADD: std::cout << "Start benchmark loop for op ADD config_idx " << config_idx << " not in-place" << std::endl; break;
        case SUB: std::cout << "Start benchmark loop for op SUB config_idx " << config_idx << " not in-place" << std::endl; break;
      }
      std::cout << "Start benchmark loop for config_idx " << config_idx << std::endl;
      // Destroy the result of the prev loop.
      T::rand_host_many(host_out, vector_size);   // Randomize host_out in order to randomize device_out.
      err = cudaMemcpy(
        device_out, host_out, vector_size * sizeof(T),
        cudaMemcpyHostToDevice); // Copy random data to device_out.
      if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from host_out to device_out - " << cudaGetErrorString(err) << std::endl;
        return 0;
      }
      T::rand_host_many(host_out, vector_size);   // Make hist_out != device_out. 
      // Initializa inputs with the known data
      for (int i = 0; i < vector_size; i++) {
        host_in1[i] = host_in1_init[i];
        host_in2[i] = host_in2_init[i];
      }
      config.is_a_on_device               = (config_idx >> 4) & 0x1;
      config.is_b_on_device               = (config_idx >> 3) & 0x1;
      config.is_result_on_device          = (config_idx >> 2) & 0x1;
      config.is_input_in_montgomery_form  = (config_idx >> 1) & 0x1;
      config.is_result_in_montgomery_form = (config_idx >> 0) & 0x1;

      // Copy from host to device (copy again in order to be used later in the loop and device_inX was already
      // overwritten by warmup.
      if (config.is_a_on_device) {
        if (config.is_input_in_montgomery_form) {
          err =
            cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(
            mont::to_montgomery(device_in1, vector_size, config.ctx.stream, device_in1)); // Convert in-place.
          // Destroy host_in1 value with values of host_in2.
          for (int i = 0; i < vector_size; i++) {
            host_in1[i] = host_in2_init[i];
          }
        } else {                                                                          // Normal presentation.
          err =
            cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          err =
            cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(mont::to_montgomery(device_in1, vector_size, config.ctx.stream, device_in1));
          err = cudaMemcpy(host_in1, device_in1, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from device_in1 to host_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          // Destroy device_in1 value with values of host_in2.
          err =
            cudaMemcpy(device_in1, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      }
      if (config.is_b_on_device) {
        if (config.is_input_in_montgomery_form) {
          err =
            cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(
            mont::to_montgomery(device_in2, vector_size, config.ctx.stream, device_in2)); // Convert in-place.
          // Destroy host_in2 value with values of host_in1.
          for (int i = 0; i < vector_size; i++) {
            host_in2[i] = host_in1_init[i];
          }
        } else {
          // Normal presentation.
          err =
            cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          err =
            cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(mont::to_montgomery(device_in2, vector_size, config.ctx.stream, device_in2));
          err = cudaMemcpy(host_in2, device_in2, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from device_in2 to host_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          // Destroy device_in2 valuewith values of host_in1.
          err =
            cudaMemcpy(device_in2, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      }
      CHK_IF_RETURN(cudaPeekAtLastError());

      auto start_time = std::chrono::high_resolution_clock::now();
      // Benchmark loop
      for (int i = 0; i < not_in_place_repetitions; i++) {
        switch (config_idx >> (nof_of_configs_for_test - nof_of_storage_configs)) { // {is_a_on_device, is_b_on_device, is_result_on_device}
          case 0b000:
            vector_op(host_in1, host_in2, host_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b001:
            vector_op(host_in1, host_in2, device_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b010:
            vector_op(host_in1, device_in2, host_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b011:
            vector_op(host_in1, device_in2, device_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b100:
            vector_op(device_in1, host_in2, host_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b101:
            vector_op(device_in1, host_in2, device_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b110:
            vector_op(device_in1, device_in2, host_out, vector_size, ctx, config, (Op)op);
            break;
          case 0b111:
            vector_op(device_in1, device_in2, device_out, vector_size, ctx, config, (Op)op);
            break;
        }
        CHK_IF_RETURN(cudaPeekAtLastError());
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      switch (op) {
      case MUL:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation MUL for config_idx "
                  << config_idx << " and result not in-place" << std::endl;
        break;
      case ADD:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation ADD for config_idx "
                  << config_idx << " and result not in-place" << std::endl;
        break;
      case SUB:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation SUB for config_idx "
                  << config_idx << " and result not in-place" << std::endl;
        break;
      }

      if (config.is_result_on_device) { // Copy the data to host_out in order to compare it vs. host_out_ref_XXX value.
        if (config.is_result_in_montgomery_form) { // Convert to normal from montgomery if needed.
          CHK_IF_RETURN(mont::from_montgomery(
            device_out, vector_size, config.ctx.stream,
            device_out)); // Convert to normal in order to check vs. host_out_ref_XXX.
        }
        err = cudaMemcpy(
          host_out, device_out, vector_size * sizeof(T),
          cudaMemcpyDeviceToHost); // Copy to host_out in order to check vs. host_out_ref_XXX.
        if (err != cudaSuccess) {
          std::cerr << "Failed to copy data from device_out to host - " << cudaGetErrorString(err) << std::endl;
          return 0;
        }
      } else {                              // Data is not on device but it is in host_out.
        if (config.is_result_in_montgomery_form) { // host_out should be written to device, converted to mmontgomery and
                                                   // written back to host. Then compared vs. host_out_ref_XXX.
          err = cudaMemcpy(
            device_out, host_out, vector_size * sizeof(T),
            cudaMemcpyHostToDevice); // Copy to device_out in order to check later vs. host_out_ref_XXX.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_out to device_out - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(mont::from_montgomery(
            device_out, vector_size, config.ctx.stream,
            device_out)); // Convert to normal in order to check vs. host_out_ref_XXX.
          err = cudaMemcpy(
            host_out, device_out, vector_size * sizeof(T),
            cudaMemcpyDeviceToHost); // Copy to host_out in order to check vs. host_out_ref_XXX.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from device_out to host_out - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        } else { // host_out could be compared vs. host_out_ref_XXX as is.   
        }
      }
      //****************************************
      // End of benchmark test.
      //****************************************

      //***********************************************
      // Test result check (not in-place)
      // Check is performed by executing the operation in a normal presentation
      //   (located in in host_out_ref_XXX) and comparing it with the
      //   benchmark test result.
      //***********************************************
      int test_failed = 0;
      switch (op) {
      case MUL:
        for (int i = 0; i < vector_size; i++) {
          if (host_out_ref_mul[i] != host_out[i]) {
            std::cout << "===>>> ERROR!!! MUL: Test failed for vector index " << i
                      << ", config is printed below:" << std::endl;
            test_failed = 1;
          }
        }
        break;
      case ADD:
        for (int i = 0; i < vector_size; i++) {
          if (host_out_ref_add[i] != host_out[i]) {
            std::cout << "===>>> ERROR!!! ADD: Test failed for vector index " << i
                      << ", config is printed below:" << std::endl;
            test_failed = 1;
          }
        }
        break;
      case SUB:
        for (int i = 0; i < vector_size; i++) {
          if (host_out_ref_sub[i] != host_out[i]) {
            std::cout << "===>>> ERROR!!! SUB: Test failed for vector index " << i
                      << ", config is printed below:" << std::endl;
            test_failed = 1;
          }
        }
        break;
      }
      if (test_failed) {
        std::cout << "===>>> result is in-place:                " << std::endl;
        std::cout << "===>>> is_a_on_device:                    " << config.is_a_on_device << std::endl;
        std::cout << "===>>> is_b_on_device:                    " << config.is_b_on_device << std::endl;
        std::cout << "===>>> is_result_on_device:               " << config.is_result_on_device << std::endl;
        std::cout << "===>>> is_input_in_montgomery_form:       " << config.is_input_in_montgomery_form << std::endl;
        std::cout << "===>>> is_input_resultin_montgomery_form: " << config.is_result_in_montgomery_form << std::endl;
        std::cout << "===>>> host_in1_init[0]                      = " << host_in1_init[0] << std::endl;
        std::cout << "===>>> host_in2_init[0]                      = " << host_in2_init[0] << std::endl;
        std::cout << "===>>> host_out[0]                           = " << host_out[0] << std::endl;
        std::cout << "===>>> warm-up: normal host_out_ref[0] (MUL) = " << host_out_ref_mul[0] << std::endl;
        std::cout << "===>>> warm-up: normal host_out_ref[0] (ADD) = " << host_out_ref_add[0] << std::endl;
        std::cout << "===>>> warm-up: normal host_out_ref[0] (SUB) = " << host_out_ref_sub[0] << std::endl;
        exit(2);
      }

      unsigned power_after;
      nvmlDeviceGetPowerUsage(device, &power_after);
      std::cout << "Power after: " << std::fixed << std::setprecision(3) << 1.0e-3 * power_after << " W" << std::endl;
      std::cout << "Power utilization: " << std::fixed << std::setprecision(1)
                << (float)100.0 * power_after / power_limit << " %" << std::endl;
      unsigned temperature_after;
      if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_after) == NVML_SUCCESS) {
        std::cout << "GPU Temperature after: " << temperature_after << " C" << std::endl;
      } else {
        std::cerr << "Failed to get GPU temperature." << std::endl;
      }

      // Report performance in GMPS: Giga Multiplications Per Second
      double GMPS = 1.0e-9 * not_in_place_repetitions * vector_size / (1.0e-6 * duration.count());
      std::cout << "Performance: " << GMPS << " Giga Multiplications Per Second" << std::endl;
    }
  }

  // Test when the result is in-place
  std::cout << "*************************************" << std::endl;
  std::cout << "*** Start in-place benchmark loop ***" << std::endl;
  std::cout << "*************************************" << std::endl;
  for (int op = MUL; op != LAST; op++) {
    for (int config_idx = 0; config_idx < 32; config_idx++) {
      switch (op) {
        case MUL: std::cout << "Start benchmark loop for op MUL config_idx " << config_idx << " in-place" << std::endl; break;
        case ADD: std::cout << "Start benchmark loop for op ADD config_idx " << config_idx << " in-place" << std::endl; break;
        case SUB: std::cout << "Start benchmark loop for op SUB config_idx " << config_idx << " in-place" << std::endl; break;
      }
      std::cout << "Start benchmark loop for config_idx " << config_idx << std::endl;
      // Destroy the result of the prev loop.
      T::rand_host_many(host_out, vector_size);   // Randomize host_out in order to randomize device_out.
      err = cudaMemcpy(
        device_out, host_out, vector_size * sizeof(T),
        cudaMemcpyHostToDevice); // Copy random data to device_out.
      if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from host_out to device_out - " << cudaGetErrorString(err) << std::endl;
        return 0;
      }
      T::rand_host_many(host_out, vector_size);   // Make hist_out != device_out. 
      // Initializa inputs with the known data. For in-place tests host_in1 is going to be used as a result. So, it should be initialized later in the repetions loop.
      for (int i = 0; i < vector_size; i++) {
        host_in1[i] = host_in1_init[i];
        host_in2[i] = host_in2_init[i];
      }
      config.is_a_on_device               = (config_idx >> 4) & 0x1;
      config.is_b_on_device               = (config_idx >> 3) & 0x1;
      config.is_result_on_device          = (config_idx >> 2) & 0x1;
      config.is_input_in_montgomery_form  = (config_idx >> 1) & 0x1;
      config.is_result_in_montgomery_form = (config_idx >> 1) & 0x1;
      if (config.is_a_on_device ^ config.is_result_on_device == 1) { continue; }    // Illegal case for this loop.
      if (config.is_input_in_montgomery_form ^ config.is_result_in_montgomery_form == 1) { continue; }    // Illegal case for this loop.

      // Copy from host to device (copy again in order to be used later in the loop and device_inX was already
      // overwritten by warmup.
      if (config.is_a_on_device) {
        if (config.is_input_in_montgomery_form) {
          err =
            cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(
            mont::to_montgomery(device_in1, vector_size, config.ctx.stream, device_in1)); // Convert in-place.
          // Destroy host_in1 value with values of host_in2.
          for (int i = 0; i < vector_size; i++) {
            host_in1[i] = host_in2_init[i];
          }
        } else {                                                                          // Normal presentation.
          err =
            cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          err =
            cudaMemcpy(device_in1, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(mont::to_montgomery(device_in1, vector_size, config.ctx.stream, device_in1));
          err = cudaMemcpy(host_in1, device_in1, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from device_in1 to host_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          // Destroy device_in1 value with values of host_in2.
          err =
            cudaMemcpy(device_in1, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      }
      if (config.is_b_on_device) {
        if (config.is_input_in_montgomery_form) {
          err =
            cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in1 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(
            mont::to_montgomery(device_in2, vector_size, config.ctx.stream, device_in2)); // Convert in-place.
          // Destroy host_in2 value with values of host_in1.
          for (int i = 0; i < vector_size; i++) {
            host_in2[i] = host_in1_init[i];
          }
        } else {
          // Normal presentation.
          err =
            cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          err =
            cudaMemcpy(device_in2, host_in2, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in2 to device_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(mont::to_montgomery(device_in2, vector_size, config.ctx.stream, device_in2));
          err = cudaMemcpy(host_in2, device_in2, vector_size * sizeof(T), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from device_in2 to host_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          // Destroy device_in2 valuewith values of host_in1.
          err =
            cudaMemcpy(device_in2, host_in1, vector_size * sizeof(T), cudaMemcpyHostToDevice); // Copy data to device.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_in2 - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        }
      }
      CHK_IF_RETURN(cudaPeekAtLastError());

      auto start_time = std::chrono::high_resolution_clock::now();
      // Benchmark loop
      for (int i = 0; i < in_place_repetitions; i++) {
        switch (config_idx >> (nof_of_configs_for_test - nof_of_storage_configs)) { // {is_a_on_device, is_b_on_device, is_result_on_device}
          case 0b000:
            vector_op(host_in1, host_in2, host_in1, vector_size, ctx, config, (Op)op);
            break;
          case 0b001:
            break;
          case 0b010:
            vector_op(host_in1, device_in2, host_in1, vector_size, ctx, config, (Op)op);
            break;
          case 0b011:
            break;
          case 0b100:
            break;
          case 0b101:
            vector_op(device_in1, host_in2, device_in1, vector_size, ctx, config, (Op)op);
            std::cout << "===>>> COMMAND: vector_op(device_in1, host_in2, device_in1, vector_size, ctx, config, (Op)op);" << std::endl;
            break;
          case 0b110:
            break;
          case 0b111:
            vector_op(device_in1, device_in2, device_in1, vector_size, ctx, config, (Op)op);
            break;
        }
        CHK_IF_RETURN(cudaPeekAtLastError());
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      switch (op) {
      case MUL:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation MUL for config_idx "
                  << config_idx << " and result in-place" << std::endl;
        break;
      case ADD:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation ADD for config_idx "
                  << config_idx << " and result in-place" << std::endl;
        break;
      case SUB:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation SUB for config_idx "
                  << config_idx << " and result in-place" << std::endl;
        break;
      }

      if (config.is_result_on_device) { // Copy the data to host_out in order to compare it vs. host_out_ref_XXX value.
        if (config.is_result_in_montgomery_form) { // Convert to normal from montgomery if needed.
          CHK_IF_RETURN(mont::from_montgomery(
            device_in1, vector_size, config.ctx.stream,
            device_in1)); // Convert to normal in order to check vs. host_out_ref_XXX.
        }
        err = cudaMemcpy(
          host_out, device_in1, vector_size * sizeof(T),
          cudaMemcpyDeviceToHost); // Copy to host_out in order to check vs. host_out_ref_XXX.
        if (err != cudaSuccess) {
          std::cerr << "Failed to copy data from device_in1 to host_out - " << cudaGetErrorString(err) << std::endl;
          return 0;
        }
      } else { // Data is not on device but it is in host_in1. It should be moved to host_out for test pass/fail check.
        if (config.is_result_in_montgomery_form) { // host_out should be written to device, converted to mmontgomery and
                                                   // written back to host. Then compared vs. host_out_ref_XXX.
          err = cudaMemcpy(
            device_out, host_in1, vector_size * sizeof(T),
            cudaMemcpyHostToDevice); // Copy to device_out in order to check later vs. host_out_ref_XXX.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from host_in1 to device_out - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
          CHK_IF_RETURN(mont::from_montgomery(
            device_out, vector_size, config.ctx.stream,
            device_out)); // Convert to normal in order to check vs. host_out_ref_XXX.
          err = cudaMemcpy(
            host_out, device_out, vector_size * sizeof(T),
            cudaMemcpyDeviceToHost); // Copy to host_out in order to check vs. host_out_ref_XXX.
          if (err != cudaSuccess) {
            std::cerr << "Failed to copy data from device_out to host_out - " << cudaGetErrorString(err) << std::endl;
            return 0;
          }
        } else { // host_out could be compared vs. host_out_ref_XXX as is.
          for (int i = 0; i < vector_size; i++) {   // Copy from host_in1 (result) to host_out to compare later vs. host_out_ref_XXX.
            host_out[i] = host_in1[i];
          }
        }
      }
      //****************************************
      // End of benchmark test.
      //****************************************

      //***********************************************
      // Test result check (in-place)
      // Check is performed by executing the operation in a normal presentation
      //   (located in in host_out_ref_XXX) and comparing it with the
      //   benchmark test result.
      //***********************************************
      int test_failed = 0;
      switch (op) {
        case MUL:
          for (int i = 0; i < vector_size; i++) {
            if (host_out_ref_mul[i] != host_out[i]) {
              std::cout << "===>>> ERROR!!! MUL: Test failed for vector index " << i
                        << ", config is printed below:" << std::endl;
              test_failed = 1;
            }
          }
          break;
        case ADD:
          for (int i = 0; i < vector_size; i++) {
            if (host_out_ref_add[i] != host_out[i]) {
              std::cout << "===>>> ERROR!!! ADD: Test failed for vector index " << i
                        << ", config is printed below:" << std::endl;
              test_failed = 1;
            }
          }
          break;
        case SUB:
          for (int i = 0; i < vector_size; i++) {
            if (host_out_ref_sub[i] != host_out[i]) {
              std::cout << "===>>> ERROR!!! SUB: Test failed for vector index " << i
                        << ", config is printed below:" << std::endl;
              test_failed = 1;
            }
          }
          break;
      }
      if (test_failed) {
        std::cout << "===>>> result is in-place:                " << std::endl;
        std::cout << "===>>> is_a_on_device:                    " << config.is_a_on_device << std::endl;
        std::cout << "===>>> is_b_on_device:                    " << config.is_b_on_device << std::endl;
        std::cout << "===>>> is_result_on_device:               " << config.is_result_on_device << std::endl;
        std::cout << "===>>> is_input_in_montgomery_form:       " << config.is_input_in_montgomery_form << std::endl;
        std::cout << "===>>> is_input_resultin_montgomery_form: " << config.is_result_in_montgomery_form << std::endl;
        std::cout << "===>>> host_in1_init[0]                      = " << host_in1_init[0] << std::endl;
        std::cout << "===>>> host_in2_init[0]                      = " << host_in2_init[0] << std::endl;
        std::cout << "===>>> host_out[0]                           = " << host_out[0] << std::endl;
        std::cout << "===>>> warm-up: normal host_out_ref[0] (MUL) = " << host_out_ref_mul[0] << std::endl;
        std::cout << "===>>> warm-up: normal host_out_ref[0] (ADD) = " << host_out_ref_add[0] << std::endl;
        std::cout << "===>>> warm-up: normal host_out_ref[0] (SUB) = " << host_out_ref_sub[0] << std::endl;
        exit(2);
      }

      unsigned power_after;
      nvmlDeviceGetPowerUsage(device, &power_after);
      std::cout << "Power after: " << std::fixed << std::setprecision(3) << 1.0e-3 * power_after << " W" << std::endl;
      std::cout << "Power utilization: " << std::fixed << std::setprecision(1)
                << (float)100.0 * power_after / power_limit << " %" << std::endl;
      unsigned temperature_after;
      if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_after) == NVML_SUCCESS) {
        std::cout << "GPU Temperature after: " << temperature_after << " C" << std::endl;
      } else {
        std::cerr << "Failed to get GPU temperature." << std::endl;
      }

      // Report performance in GMPS: Giga Multiplications Per Second
      double GMPS = 1.0e-9 * in_place_repetitions * vector_size / (1.0e-6 * duration.count());
      std::cout << "Performance: " << GMPS << " Giga Multiplications Per Second" << std::endl;
    }
  }

  // clean up and exit
  free(host_in1_init);
  std::cout << "example free host_in1_init" << std::endl;
  free(host_in2_init);
  std::cout << "example free host_in2_init" << std::endl;
  free(host_in1);
  std::cout << "example free host_in1" << std::endl;
  free(host_in2);
  std::cout << "example free host_in2" << std::endl;
  free(host_out);
  std::cout << "example free host_out" << std::endl;
  free(host_out_ref_mul);
  std::cout << "example free host_out_ref_mul" << std::endl;
  free(host_out_ref_add);
  std::cout << "example free host_out_ref_add" << std::endl;
  free(host_out_ref_sub);
  std::cout << "example free host_out_ref_sub" << std::endl;
  cudaFree(device_in1);
  std::cout << "example cudaFree device_in1" << std::endl;
  cudaFree(device_in2);
  std::cout << "example cudaFree device_in2" << std::endl;
  cudaFree(device_out);
  std::cout << "example cudaFree device_out" << std::endl;
  nvmlShutdown();
  return 0;
}
