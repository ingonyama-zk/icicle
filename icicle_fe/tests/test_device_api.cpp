
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/device.h"
#include "icicle/device_api.h"

class DeviceApiTest : public ::testing::Test
{
public:
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite() {}
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};


TEST_F(DeviceApiTest, MemoryCopySync)
{
  int input[2] = {1, 2};
  int output[2] = {0, 0};

  icicle::Device dev = {"CPU", 0};
  auto device_api = icicle::getDeviceAPI(&dev);
  void* dev_mem = nullptr;
  ICICLE_CHECK(device_api->allocateMemory(dev, &dev_mem, sizeof(input)));
  ICICLE_CHECK(device_api->copyToDevice(dev, dev_mem, input, sizeof(input)));
  ICICLE_CHECK(device_api->copyToHost(dev, output, dev_mem, sizeof(input)));
  ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
}

// TEST_F(DeviceApiTest, MemoryCopyAsync)
// {
//   int input[2] = {1, 2};
//   int output[2] = {0, 0};

//   icicle::Device dev = {"CPU", 0};
//   auto device_api = getDeviceAPI(&dev);
//   void* dev_mem = nullptr;

//   IcicleStreamHandle stream;
//   device_api->createStream()

//    auto rv = device_api->allocateMemoryAsy(dev, &dev_mem, sizeof(input));
//   device_api->copyToDevice(dev, dev_mem, input, sizeof(input));
//   device_api->copyToHost(dev, output, dev_mem, sizeof(input));
//   ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
// }

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}