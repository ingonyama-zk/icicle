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
  const char* selected_device = argc > 1 ? argv[1] : nullptr;

  if (nullptr == selected_device) {
    ICICLE_LOG_INFO << "Defaulting to CPU device";
    return;
  }

  icicle_load_backend_from_env_or_default();

  // trying to choose device if available, or fallback to CPU otherwise (default device)
  const bool is_device_available = (eIcicleError::SUCCESS == icicle_is_device_available(selected_device));
  if (is_device_available) {
    ICICLE_LOG_INFO << "selecting " << selected_device << " device";
    ICICLE_CHECK(icicle_set_device(selected_device));
    return;
  }

  ICICLE_LOG_INFO << "Device " << selected_device << " is not available, falling back to CPU";
}