
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/runtime.h"

using namespace icicle;

class DeviceApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    s_regsitered_devices = getRegisteredDevices();
    ASSERT_GT(s_regsitered_devices.size(), 0);
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(DeviceApiTest, UnregisteredDeviceError)
{
  icicle::Device dev = {"INVALID_DEVICE", 2};
  EXPECT_ANY_THROW(getDeviceAPI(dev));
}

TEST_F(DeviceApiTest, MemoryCopySync)
{
  int input[2] = {1, 2};

  for (const auto& device_type : s_regsitered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type.c_str(), 0};
    icicleSetDevice(dev);

    void* dev_mem = nullptr;
    ICICLE_CHECK(icicleMalloc(&dev_mem, sizeof(input)));
    ICICLE_CHECK(icicleCopyToDevice(dev_mem, input, sizeof(input)));
    ICICLE_CHECK(icicleCopyToHost(output, dev_mem, sizeof(input)));
    ICICLE_CHECK(icicleFree(dev_mem));

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, MemoryCopyAsync)
{
  int input[2] = {1, 2};
  for (const auto& device_type : s_regsitered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type.c_str(), 0};
    icicleSetDevice(dev);
    void* dev_mem = nullptr;

    icicleStreamHandle stream;
    ICICLE_CHECK(icicleCreateStream(&stream));
    ICICLE_CHECK(icicleMallocAsync(&dev_mem, sizeof(input), stream));
    ICICLE_CHECK(icicleCopyToDeviceAsync(dev_mem, input, sizeof(input), stream));
    ICICLE_CHECK(icicleCopyToHostAsync(output, dev_mem, sizeof(input), stream));
    ICICLE_CHECK(icicleFreeAsync(dev_mem, stream));
    ICICLE_CHECK(icicleStreamSynchronize(stream));

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, ApiError)
{
  for (const auto& device_type : s_regsitered_devices) {
    icicle::Device dev = {device_type.c_str(), 0};
    icicleSetDevice(dev);
    void* dev_mem = nullptr;
    EXPECT_ANY_THROW(ICICLE_CHECK(icicleMalloc(&dev_mem, -1)));
  }
}

TEST_F(DeviceApiTest, AvailableMemory)
{
  icicle::Device dev = {"CUDA", 0}; // TODO Yuval: implement for CPU too
  icicleSetDevice(dev);
  size_t total, free;
  ASSERT_EQ(eIcicleError::SUCCESS, icicleGetAvailableMemory(total, free));

  double total_GB = double(total) / (1 << 30);
  double free_GB = double(free) / (1 << 30);
  std::cout << std::fixed << std::setprecision(2) << "total=" << total_GB << "[GB], free=" << free_GB << "[GB]"
            << std::endl;
}

TEST_F(DeviceApiTest, InvalidDevice)
{
  for (const auto& device_type : s_regsitered_devices) {
    icicle::Device dev = {device_type.c_str(), 10}; // no such device-id thus expecting an error
    ASSERT_EQ(eIcicleError::INVALID_DEVICE, icicleSetDevice(dev));
  }
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}