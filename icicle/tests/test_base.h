#pragma once
#include <time.h>
#include <random>

#include <gtest/gtest.h>
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
using namespace field_config; // To have access to the Field rand-gen

using FpMiliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf("%s: %.3f ms\n", msg, FpMiliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());
#define END_TIMER_AVERAGE(timer, msg, enable, iters)                                                                   \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMiliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / iters);

#define UNKOWN_DEVICE "UNKNOWN"

class IcicleTestBase : public ::testing::Test
{
public:
  static inline std::vector<std::string> s_registered_devices;
  static inline std::string s_main_device = "CPU";
  static inline std::string s_ref_device = "CPU"; // assuming always present
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    unsigned seed = time(NULL);
    srand(seed);
    ICICLE_LOG_INFO << "Seed for tests is: " << seed;
    scalar_t::seed_rand_generator(seed);
#ifdef BACKEND_BUILD_DIR
    setenv("ICICLE_BACKEND_INSTALL_DIR", BACKEND_BUILD_DIR, 0 /*=replace*/);
#endif
    icicle_load_backend_from_env_or_default();
    s_registered_devices = get_registered_devices_list();
    ASSERT_GT(s_registered_devices.size(), 0);
    ASSERT_LE(s_registered_devices.size(), 2); // assuming we have a single device except for CPU
    if (s_registered_devices.size() > 1) {
      for (auto& device : s_registered_devices) {
        // looking for first device that is not CPU and use that as main device
        if (device != s_ref_device) {
          s_main_device = device;
          break;
        }
      }
    }
    ICICLE_LOG_INFO << "Main-device=" << s_main_device << ", Reference-device=" << s_ref_device;
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}

  bool is_main_device_available() const { return s_main_device != UNKOWN_DEVICE && s_main_device != "CPU"; }
  const std::string& main_device() const { return s_main_device; }
  const std::string& reference_device() const { return s_ref_device; }
};