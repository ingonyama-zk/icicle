
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/runtime.h"
#include "icicle/memory_tracker.h"
#include "test_base.h"
#include "icicle/utils/rand_gen.h"

using namespace icicle;

class DeviceApiTest : public IcicleTestBase
{
};

TEST_F(DeviceApiTest, UnregisteredDeviceError)
{
  icicle::Device dev = {"INVALID_DEVICE", 2};
  EXPECT_ANY_THROW(get_deviceAPI(dev));
}

TEST_F(DeviceApiTest, MemoryCopySync)
{
  int input[2] = {1, 2};

  for (const auto& device_type : s_registered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type, 0};
    ICICLE_CHECK(icicle_set_device(dev));

    void* dev_memA = nullptr;
    void* dev_memB = nullptr;
    // test copy host->device->device->host
    ICICLE_CHECK(icicle_malloc(&dev_memA, sizeof(input)));
    ICICLE_CHECK(icicle_malloc(&dev_memB, sizeof(input)));
    ICICLE_CHECK(icicle_copy_to_device(dev_memA, input, sizeof(input)));
    ICICLE_CHECK(icicle_copy(dev_memB, dev_memA, sizeof(input)));
    ICICLE_CHECK(icicle_copy_to_host(output, dev_memB, sizeof(input)));
    ICICLE_CHECK(icicle_free(dev_memB));
    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, MemoryCopySyncWithOffset)
{
  int input[4] = {1, 2, 3, 4};
  int expected[4] = {3, 4, 1, 2};

  for (const auto& device_type : s_registered_devices) {
    int output[4] = {0, 0, 0, 0};

    icicle::Device dev = {device_type, 0};
    ICICLE_CHECK(icicle_set_device(dev));

    int* dev_mem = nullptr;
    ICICLE_CHECK(icicle_malloc((void**)&dev_mem, 4 * sizeof(int)));
    ICICLE_CHECK(icicle_copy_to_device(dev_mem, input + 2, 2 * sizeof(int)));
    ICICLE_CHECK(icicle_copy_to_device(dev_mem + 2, input, 2 * sizeof(int)));
    ICICLE_CHECK(icicle_copy_to_host(output, dev_mem, 4 * sizeof(int)));
    ICICLE_CHECK(icicle_free(dev_mem));

    ASSERT_EQ(0, memcmp(expected, output, 4 * sizeof(int)));
  }
}

TEST_F(DeviceApiTest, MemoryCopyAsync)
{
  int input[2] = {1, 2};
  for (const auto& device_type : s_registered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type, 0};
    ICICLE_CHECK(icicle_set_device(dev));
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

TEST_F(DeviceApiTest, CopyDeviceInference)
{
  int input[2] = {1, 2};
  for (const auto& device_type : s_registered_devices) {
    int output[2] = {0, 0};

    icicle::Device dev = {device_type, 0};
    ICICLE_CHECK(icicle_set_device(dev));
    void* dev_mem = nullptr;

    ICICLE_CHECK(icicle_malloc(&dev_mem, sizeof(input)));
    ICICLE_CHECK(icicle_copy(dev_mem, input, sizeof(input)));  // implicit host to device
    ICICLE_CHECK(icicle_copy(output, dev_mem, sizeof(input))); // implicit device to host

    ASSERT_EQ(0, memcmp(input, output, sizeof(input)));
  }
}

TEST_F(DeviceApiTest, Memset)
{
  char expected[2] = {1, 2};
  for (const auto& device_type : s_registered_devices) {
    char host_mem[2] = {0, 0};

    icicle::Device dev = {device_type, 0};
    ICICLE_CHECK(icicle_set_device(dev));
    char* dev_mem = nullptr;

    ICICLE_CHECK(icicle_malloc((void**)&dev_mem, sizeof(host_mem)));
    ICICLE_CHECK(icicle_memset(dev_mem, 1, 1));
    ICICLE_CHECK(icicle_memset(dev_mem + 1, 2, 1));                 // memset with offset
    ICICLE_CHECK(icicle_copy(host_mem, dev_mem, sizeof(host_mem))); // implicit device to host

    ASSERT_EQ(0, memcmp(expected, host_mem, sizeof(host_mem)));
  }
}

TEST_F(DeviceApiTest, ApiError)
{
  for (const auto& device_type : s_registered_devices) {
    icicle::Device dev = {device_type, 0};
    ICICLE_CHECK(icicle_set_device(dev));
    void* dev_mem = nullptr;
    EXPECT_ANY_THROW(ICICLE_CHECK(icicle_malloc(&dev_mem, -1)));
  }
}

TEST_F(DeviceApiTest, AvailableMemory)
{
  icicle::Device dev = {"CUDA", 0};
  const bool is_cuda_registered = eIcicleError::SUCCESS == icicle_is_device_available(dev);
  if (!is_cuda_registered) { GTEST_SKIP(); } // most devices do not support this

  ICICLE_CHECK(icicle_set_device(dev));
  size_t total, free;
  ASSERT_EQ(eIcicleError::SUCCESS, icicle_get_available_memory(total, free));

  double total_GB = double(total) / (1 << 30);
  double free_GB = double(free) / (1 << 30);
  std::cout << std::fixed << std::setprecision(2) << "total=" << total_GB << "[GB], free=" << free_GB << "[GB]"
            << std::endl;
}

TEST_F(DeviceApiTest, InvalidDevice)
{
  for (const auto& device_type : s_registered_devices) {
    icicle::Device dev = {device_type, 10}; // no such device-id thus expecting an error
    ASSERT_EQ(eIcicleError::INVALID_DEVICE, icicle_set_device(dev));
  }
}

TEST_F(DeviceApiTest, memoryTracker)
{
  // need two devices for this test
  if (s_registered_devices.size() == 1) { return; }
  const int NOF_ALLOCS = 200; // Note that some backends have a bound (typically 256) on allocations
  const int ALLOC_SIZE = 1 << 20;

  MemoryTracker<Device> tracker{};
  ICICLE_ASSERT(s_main_device != UNKOWN_DEVICE) << "memoryTracker test assumes more than one device";
  Device main_device = {s_main_device, 0};
  ICICLE_CHECK(icicle_set_device(main_device));

  std::vector<void*> allocated_addresses(NOF_ALLOCS, nullptr);

  START_TIMER(allocation);
  for (auto& it : allocated_addresses) {
    icicle_malloc(&it, ALLOC_SIZE);
  }
  END_TIMER_AVERAGE(allocation, "memory-tracker: malloc average", true, NOF_ALLOCS);

  START_TIMER(insertion);
  for (auto& it : allocated_addresses) {
    tracker.add_allocation(it, ALLOC_SIZE, main_device);
  }
  END_TIMER_AVERAGE(insertion, "memory-tracker: insert average", true, NOF_ALLOCS);

  START_TIMER(lookup);
  for (auto& it : allocated_addresses) {
    // identify addresses identified correctly (to active device)
    const void* addr = (void*)((size_t)it + rand_uint_32b(0, RAND_MAX) % ALLOC_SIZE);
    ICICLE_CHECK(icicle_is_active_device_memory(addr));
  }
  END_TIMER_AVERAGE(lookup, "memory-tracker: lookup (and compare) average", true, NOF_ALLOCS);

  // test host pointers are identified as host memory
  auto host_mem = std::make_unique<int>();
  ICICLE_CHECK(icicle_is_host_memory(host_mem.get()));
  ASSERT_EQ(eIcicleError::INVALID_POINTER, icicle_is_active_device_memory(host_mem.get()));

  // test that we still identify correctly after switching device
  ICICLE_CHECK(icicle_set_device({"CPU", 0}));
  const void* addr = (void*)((size_t)*allocated_addresses.begin() + rand_uint_32b(0, RAND_MAX) % ALLOC_SIZE);
  ASSERT_EQ(eIcicleError::INVALID_POINTER, icicle_is_active_device_memory(addr));
  ASSERT_EQ(eIcicleError::INVALID_POINTER, icicle_is_active_device_memory(host_mem.get()));
  auto it = tracker.identify(addr);
  ASSERT_EQ(*it->first, main_device);

  ICICLE_CHECK(icicle_set_device(main_device));
  START_TIMER(remove);
  for (auto& it : allocated_addresses) {
    tracker.remove_allocation(it);
  }
  END_TIMER_AVERAGE(remove, "memory-tracker: remove average", true, NOF_ALLOCS);

  START_TIMER(free);
  for (auto& it : allocated_addresses) {
    icicle_free(it);
  }
  END_TIMER_AVERAGE(free, "memory-tracker: free average", true, NOF_ALLOCS);

  void* mem;
  ICICLE_CHECK(icicle_malloc(&mem, ALLOC_SIZE));
  ICICLE_CHECK(icicle_free(mem));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}