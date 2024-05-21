
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/device.h"
#include "icicle/device_api.h"

using namespace icicle;

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


TEST_F(DeviceApiTest, UnregisteredDeviceError)
{
  icicle::Device dev = {"INVALID_DEVICE", 2};
  EXPECT_ANY_THROW(getDeviceAPI(&dev));
}

TEST_F(DeviceApiTest, MemoryCopySync)
{
  int input[2] = {1, 2};
  int output[2] = {0, 0};

  icicle::Device dev = {"CPU", 0};
  auto device_api = getDeviceAPI(&dev);
  void* dev_mem = nullptr;
  ICICLE_CHECK(device_api->allocateMemory(dev, &dev_mem, sizeof(input)));
  ICICLE_CHECK(device_api->copyToDevice(dev, dev_mem, input, sizeof(input)));
  ICICLE_CHECK(device_api->copyToHost(dev, output, dev_mem, sizeof(input)));

  ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
}

TEST_F(DeviceApiTest, MemoryCopyAsync)
{
  int input[2] = {1, 2};
  int output[2] = {0, 0};

  icicle::Device dev = {"CPU", 0};
  auto device_api = getDeviceAPI(&dev);
  void* dev_mem = nullptr;

  IcicleStreamHandle stream;
  ICICLE_CHECK(device_api->createStream(dev, &stream));
  ICICLE_CHECK(device_api->allocateMemoryAsync(dev, &dev_mem, sizeof(input), stream));
  ICICLE_CHECK(device_api->copyToDeviceAsync(dev, dev_mem, input, sizeof(input), stream));
  ICICLE_CHECK(device_api->copyToHostAsync(dev, output, dev_mem, sizeof(input), stream));
  ICICLE_CHECK(device_api->synchronize(dev, stream));

  ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
}

TEST_F(DeviceApiTest, ApiError) {
  icicle::Device dev = {"CPU", 0};
  auto device_api = getDeviceAPI(&dev);
  void* dev_mem = nullptr;
  EXPECT_ANY_THROW(ICICLE_CHECK(device_api->allocateMemory(dev, &dev_mem, -1)));
}

TEST_F(DeviceApiTest, AvailableMemory) {
  icicle::Device dev = {"CPU", 0};
  auto device_api = getDeviceAPI(&dev);
  size_t total, free;
  // TODO Yuval:fix when implemented
  ASSERT_EQ(IcicleError::API_NOT_IMPLEMENTED, device_api->getAvailableMemory(dev, total, free));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}