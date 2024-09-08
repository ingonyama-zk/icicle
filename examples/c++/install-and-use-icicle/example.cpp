#include <iostream>
#include <cassert>
#include "icicle/runtime.h"
#include "icicle/api/bn254.h"

using namespace bn254; // This makes scalar_t a bn254 scalar instead of bn254::scalar_t

// Utility function to print arrays
template <typename T>
void print_array(const T* arr, int size)
{
  for (int i = 0; i < size; ++i) {
    std::cout << "\t" << i << ": " << arr[i] << std::endl;
  }
}

int main(int argc, char* argv[])
{
  // Load installed backends
  icicle_load_backend_from_env_or_default();

  // Check if GPU is available
  Device device_cpu = {"CPU", 0};
  const bool is_cuda_device_available = (eIcicleError::SUCCESS == icicle_is_device_available("CUDA"));
  Device device_gpu = {"CUDA", 0};
  if (is_cuda_device_available) {
    ICICLE_LOG_INFO << "GPU is available";
  } else {
    ICICLE_LOG_INFO << "GPU is not available, falling back to CPU only";
    device_gpu = device_cpu;
  }

  // Example input (on host memory) for NTT
  const unsigned log_ntt_size = 2;
  const unsigned ntt_size = 1 << log_ntt_size;
  auto input_cpu = std::make_unique<scalar_t[]>(ntt_size);
  scalar_t::rand_host_many(input_cpu.get(), ntt_size);

  // Allocate output on host memory
  auto output_cpu = std::make_unique<scalar_t[]>(ntt_size);
  scalar_t root_of_unity = scalar_t::omega(log_ntt_size);
  auto ntt_config = default_ntt_config<scalar_t>();

  // Part 1: Running NTT on CPU
  std::cout << "Part 1: compute on CPU: " << std::endl;
  icicle_set_device(device_cpu);
  ntt_init_domain(root_of_unity, default_ntt_init_domain_config()); // Initialize NTT domain for CPU
  ntt(input_cpu.get(), ntt_size, NTTDir::kForward, default_ntt_config<scalar_t>(), output_cpu.get());
  print_array(output_cpu.get(), ntt_size);

  // Part 2: Running NTT on GPU
  std::cout << "Part 2: compute on GPU (from/to CPU memory): " << std::endl;
  icicle_set_device(device_gpu);
  ntt_init_domain(root_of_unity, default_ntt_init_domain_config()); // Initialize NTT domain for GPU
  ntt(input_cpu.get(), ntt_size, NTTDir::kForward, ntt_config, output_cpu.get());
  print_array(output_cpu.get(), ntt_size);

  // Allocate, copy data to GPU and compute on GPU memory
  std::cout << "Part 2: compute on GPU (from/to GPU memory): " << std::endl;
  scalar_t* input_gpu = nullptr;
  scalar_t* output_gpu = nullptr;
  icicle_malloc((void**)&input_gpu, ntt_size * sizeof(scalar_t));
  icicle_malloc((void**)&output_gpu, ntt_size * sizeof(scalar_t));
  icicle_copy(input_gpu, input_cpu.get(), ntt_size * sizeof(scalar_t));
  ntt_config.are_inputs_on_device = true;
  ntt_config.are_outputs_on_device = true;
  ntt(input_gpu, ntt_size, NTTDir::kForward, ntt_config, output_gpu);
  icicle_copy(output_cpu.get(), output_gpu, ntt_size * sizeof(scalar_t));
  print_array(output_cpu.get(), ntt_size);

  // Part 3: Using both CPU and GPU to compute NTT (GPU) and inverse INTT (CPU)
  auto output_intt_cpu = std::make_unique<scalar_t[]>(ntt_size);

  // Step 1: Compute NTT on GPU
  std::cout << "Part 3: compute NTT on GPU (NTT input): " << std::endl;
  icicle_set_device(device_gpu);
  ntt_config.are_inputs_on_device = false; // using host memory now
  ntt_config.are_outputs_on_device = false;
  ntt(input_cpu.get(), ntt_size, NTTDir::kForward, ntt_config, output_cpu.get());
  print_array(input_cpu.get(), ntt_size);

  // Step 2: Compute INTT on CPU
  std::cout << "Part 3: compute INTT on CPU (INTT output): " << std::endl;
  icicle_set_device(device_cpu);
  ntt(output_cpu.get(), ntt_size, NTTDir::kInverse, ntt_config, output_intt_cpu.get());
  print_array(output_intt_cpu.get(), ntt_size);

  // Assert that INTT output is the same as NTT input
  assert(0 == memcmp(input_cpu.get(), output_intt_cpu.get(), ntt_size * sizeof(scalar_t)));

  return 0;
}