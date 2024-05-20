
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/device.h"
#include "icicle/device_api.h"

class IcicleTest : public ::testing::Test
{
public:
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite() {}
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(IcicleTest, deviceAPI)
{
  std::cout << "deviceAPI test" << std::endl;
  icicle::Device dev = {"CPU", 0};
  auto cpu_device_api = getDeviceAPI(&dev);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}