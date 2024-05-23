
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/runtime.h"
#include "icicle/vec_ops/vec_ops.h"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

class FieldApiTest : public ::testing::Test
{
public:
  static inline bool VERBOSE = true;
  static inline std::list<std::string> s_regsitered_devices;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    s_regsitered_devices = getRegisteredDevices();
    ASSERT_EQ(s_regsitered_devices.size(), 2); // "CPU" and "CUDA"
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(FieldApiTest, vectorAddSync)
{
  const int N = 1 << 15;
  auto in_a = std::make_unique<scalar_t[]>(N);
  auto in_b = std::make_unique<scalar_t[]>(N);
  scalar_t::rand_host_many(in_a.get(), N);
  scalar_t::rand_host_many(in_b.get(), N);

  auto out_cpu = std::make_unique<scalar_t[]>(N);
  auto out_cuda = std::make_unique<scalar_t[]>(N);

  auto run = [&](const char* dev_type, scalar_t* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicleSetDevice(dev);
    auto config = DefaultVecOpsConfig();

    START_TIMER(VECADD_sync)
    for (int i = 0; i < iters; ++i)
      VectorAdd(in_a.get(), in_b.get(), N, config, out);
    END_TIMER(VECADD_sync, msg, measure);
  };

  run("CUDA", out_cuda.get(), "CUDA vector add", false /*=measure*/, 1 /*=iters*/); // warmup

  run("CPU", out_cpu.get(), "CPU vector add", VERBOSE /*=measure*/, 16 /*=iters*/);
  run("CUDA", out_cuda.get(), "CUDA vector add (host mem)", VERBOSE /*=measure*/, 16 /*=iters*/);

  ASSERT_EQ(0, memcmp(out_cpu.get(), out_cuda.get(), N * sizeof(scalar_t)));
}

TEST_F(FieldApiTest, vectorAddAsync)
{
  const int N = 1 << 15;
  auto in_a = std::make_unique<scalar_t[]>(N);
  auto in_b = std::make_unique<scalar_t[]>(N);
  scalar_t::rand_host_many(in_a.get(), N);
  scalar_t::rand_host_many(in_b.get(), N);

  auto out_cpu = std::make_unique<scalar_t[]>(N);
  auto out_cuda = std::make_unique<scalar_t[]>(N);

  auto run = [&](const char* dev_type, scalar_t* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicleSetDevice(dev);
    // const bool is_cpu = std::string("CPU") == dev.type;

    scalar_t *d_in_a, *d_in_b, *d_out;
    icicleStreamHandle stream;
    icicleCreateStream(&stream);
    icicleMallocAsync((void**)&d_in_a, N * sizeof(scalar_t), stream);
    icicleMallocAsync((void**)&d_in_b, N * sizeof(scalar_t), stream);
    icicleMallocAsync((void**)&d_out, N * sizeof(scalar_t), stream);
    icicleCopyToDeviceAsync(d_in_a, in_a.get(), N * sizeof(scalar_t), stream);

    auto config = DefaultVecOpsConfig();
    config.is_a_on_device = true;
    config.is_b_on_device = true;
    config.is_result_on_device = true;
    config.is_async = true;
    config.stream = stream;

    START_TIMER(VECADD_async);
    for (int i = 0; i < iters; ++i) {
      VectorAdd(d_in_a, d_in_b, N, config, d_out);
    }
    END_TIMER(VECADD_async, msg, measure);

    icicleCopyToHostAsync(out, d_out, N * sizeof(scalar_t), stream);
    icicleStreamSynchronize(stream);

    icicleFreeAsync(d_in_a, stream);
    icicleFreeAsync(d_in_b, stream);
    icicleFreeAsync(d_out, stream);
  };

  run("CPU", out_cpu.get(), "CPU vector add", VERBOSE /*=measure*/, 16 /*=iters*/);
  run("CUDA", out_cuda.get(), "CUDA vector add (device mem)", VERBOSE /*=measure*/, 16 /*=iters*/);

  ASSERT_EQ(0, memcmp(out_cpu.get(), out_cuda.get(), N * sizeof(scalar_t)));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}