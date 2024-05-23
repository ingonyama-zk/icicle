
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/device.h"
#include "icicle/device_api.h"
#include "icicle/vec_ops/vec_ops.h"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

class FieldApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    s_regsitered_devices = getRegisteredDevices();
    // ASSERT_EQ(s_regsitered_devices.size(), 2); // "CPU" and "CUDA"
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(FieldApiTest, vectorAdd)
{
  Device dev = {"CPU", 0};
  scalar_t in_a[2] = {1, 2};
  scalar_t in_b[2] = {5, 7};
  scalar_t out[2] = {0, 0};

  auto config = DefaultVecOpsConfig();
  VectorAdd(dev, in_a, in_b, 2, config, out);

  scalar_t expected[2] = {6, 9};
  ASSERT_EQ(0, memcmp(expected, out, sizeof(out)));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}