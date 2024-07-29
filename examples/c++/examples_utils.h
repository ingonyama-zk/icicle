#pragma once

#include <chrono>
#include "icicle/runtime.h"

// Timer
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

// Load and choose backend
void try_load_and_set_backend_device(int argc = 0, char** argv = nullptr)
{
  if (argc > 2 && 0 != strcmp(argv[2], "")) {
    const char* backend_install_dir = argv[2];
    std::cout << "Trying to load and backend device from " << backend_install_dir << std::endl;
    ICICLE_CHECK(icicle_load_backend(backend_install_dir, true));
  }

  const char* selected_device = argc > 1 ? argv[1] : nullptr;
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