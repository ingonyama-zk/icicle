
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/runtime.h"
#include "dlfcn.h"

using namespace icicle;

class DeviceApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    icicle_load_backend(BACKEND_BUILD_DIR, true);
    s_regsitered_devices = get_registered_devices_list();
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
  EXPECT_ANY_THROW(get_deviceAPI(dev));
}

TEST_F(DeviceApiTest, MemoryCopySync)
{
  int input[2] = {1, 2};

  for (const auto& device_type : s_regsitered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type, 0};
    icicle_set_device(dev);

    void* dev_mem = nullptr;
    ICICLE_CHECK(icicle_malloc(&dev_mem, sizeof(input)));
    ICICLE_CHECK(icicle_copy_to_device(dev_mem, input, sizeof(input)));
    ICICLE_CHECK(icicle_copy_to_host(output, dev_mem, sizeof(input)));
    ICICLE_CHECK(icicle_free(dev_mem));

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, MemoryCopyAsync)
{
  int input[2] = {1, 2};
  for (const auto& device_type : s_regsitered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type, 0};
    icicle_set_device(dev);
    void* dev_mem = nullptr;

    icicleStreamHandle stream;
    ICICLE_CHECK(icicle_create_stream(&stream));
    ICICLE_CHECK(icicle_malloc_async(&dev_mem, sizeof(input), stream));
    ICICLE_CHECK(icicle_copy_to_device_async(dev_mem, input, sizeof(input), stream));
    ICICLE_CHECK(icicle_copy_to_host_async(output, dev_mem, sizeof(input), stream));
    ICICLE_CHECK(icicle_free_async(dev_mem, stream));
    ICICLE_CHECK(icicle_stream_synchronize(stream));

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, ApiError)
{
  for (const auto& device_type : s_regsitered_devices) {
    icicle::Device dev = {device_type, 0};
    icicle_set_device(dev);
    void* dev_mem = nullptr;
    EXPECT_ANY_THROW(ICICLE_CHECK(icicle_malloc(&dev_mem, -1)));
  }
}

TEST_F(DeviceApiTest, AvailableMemory)
{
  icicle::Device dev = {"CUDA", 0};
  const bool is_cuda_registered = eIcicleError::SUCCESS == icicle_is_device_avialable(dev);
  if (!is_cuda_registered) { return; } // TODO implement for CPU too

  icicle_set_device(dev);
  size_t total, free;
  ASSERT_EQ(eIcicleError::SUCCESS, icicle_get_available_memory(total, free));

  double total_GB = double(total) / (1 << 30);
  double free_GB = double(free) / (1 << 30);
  std::cout << std::fixed << std::setprecision(2) << "total=" << total_GB << "[GB], free=" << free_GB << "[GB]"
            << std::endl;
}

TEST_F(DeviceApiTest, InvalidDevice)
{
  for (const auto& device_type : s_regsitered_devices) {
    icicle::Device dev = {device_type, 10}; // no such device-id thus expecting an error
    ASSERT_EQ(eIcicleError::INVALID_DEVICE, icicle_set_device(dev));
  }
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}