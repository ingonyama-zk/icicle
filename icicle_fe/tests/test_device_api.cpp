
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/device.h"
#include "icicle/device_api.h"

using namespace icicle;

class DeviceApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite() {
    s_regsitered_devices = getRegisteredDevices();
    ASSERT_EQ(s_regsitered_devices.size(), 2); // "CPU" and "CUDA"
  }
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

  for (const auto& device_type : s_regsitered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type.c_str(), 0};
    auto device_api = getDeviceAPI(&dev);
    void* dev_mem = nullptr;
    ICICLE_CHECK(device_api->allocateMemory(dev, &dev_mem, sizeof(input)));
    ICICLE_CHECK(device_api->copyToDevice(dev, dev_mem, input, sizeof(input)));
    ICICLE_CHECK(device_api->copyToHost(dev, output, dev_mem, sizeof(input)));
    ICICLE_CHECK(device_api->freeMemory(dev, dev_mem));

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, MemoryCopyAsync)
{
  int input[2] = {1, 2};
  for (const auto& device_type : s_regsitered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type.c_str(), 0};
    auto device_api = getDeviceAPI(&dev);
    void* dev_mem = nullptr;

    IcicleStreamHandle stream;
    ICICLE_CHECK(device_api->createStream(dev, &stream));
    ICICLE_CHECK(device_api->allocateMemoryAsync(dev, &dev_mem, sizeof(input), stream));
    ICICLE_CHECK(device_api->copyToDeviceAsync(dev, dev_mem, input, sizeof(input), stream));
    ICICLE_CHECK(device_api->copyToHostAsync(dev, output, dev_mem, sizeof(input), stream));
    ICICLE_CHECK(device_api->freeMemoryAsync(dev, dev_mem, stream));
    ICICLE_CHECK(device_api->synchronize(dev, stream));

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, ApiError) {
  for (const auto& device_type : s_regsitered_devices) {    
    icicle::Device dev = {device_type.c_str(), 0};
    auto device_api = getDeviceAPI(&dev);
    void* dev_mem = nullptr;
    EXPECT_ANY_THROW(ICICLE_CHECK(device_api->allocateMemory(dev, &dev_mem, -1)));
  }
}

TEST_F(DeviceApiTest, AvailableMemory) {
  icicle::Device dev = {"CUDA", 0}; // TODO Yuval: implement for CPU too
  auto device_api = getDeviceAPI(&dev);
  size_t total, free;
  ASSERT_EQ(IcicleError::SUCCESS, device_api->getAvailableMemory(dev, total, free));

  double total_GB = double(total) / (1<<30);
  double free_GB = double(free) / (1<<30);
  std::cout << std::fixed << std::setprecision(2) << "total=" << total_GB << "[GB], free=" << free_GB << "[GB]" << std::endl;
}

TEST_F(DeviceApiTest, InvalidDevice) {
  for (const auto& device_type : s_regsitered_devices) {
    icicle::Device dev = {device_type.c_str(), 10}; // no such device-id thus expecting an error
    auto device_api = getDeviceAPI(&dev);
    void* dev_mem = nullptr;
    ASSERT_EQ(IcicleError::INVALID_DEVICE, device_api->allocateMemory(dev, &dev_mem, 128));    
  }
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}