
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"

#include "icicle/runtime.h"
#include "icicle/curves/curve_config.h"

#include "dlfcn.h"

// using namespace curve_config;
using namespace icicle;

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

static bool VERBOSE = true;

class CurveApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    dlopen(
      "/home/administrator/users/yuvals/icicle/icicle_v3/build/backend/cpu/libicicle_cpu_device.so",
      RTLD_LAZY | RTLD_NOW);
    dlopen(
      "/home/administrator/users/yuvals/icicle/icicle_v3/build/backend/cpu/libicicle_cpu_curve_bn254.so",
      RTLD_LAZY | RTLD_NOW);

    dlopen(
      "/home/administrator/users/yuvals/icicle/icicle_v3/build/backend/cuda/libicicle_cuda_device.so",
      RTLD_LAZY | RTLD_NOW);
    dlopen(
      "/home/administrator/users/yuvals/icicle/icicle_v3/build/backend/cuda/libicicle_cuda_curve_bn254.so",
      RTLD_LAZY | RTLD_NOW);

    s_regsitered_devices = get_registered_devices();
    ASSERT_GT(s_regsitered_devices.size(), 0);
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CurveApiTest, MSM)
{
  // TODO Yuval
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}