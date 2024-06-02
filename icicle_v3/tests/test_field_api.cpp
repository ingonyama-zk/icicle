
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"

#include "icicle/runtime.h"
#include "icicle/vec_ops.h"
#include "icicle/ntt.h"
#include "icicle/matrix_ops.h"

#include "icicle/fields/field_config.h"

using namespace field_config;
using namespace icicle;

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

static bool VERBOSE = true;
template <typename T>
class FieldApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    icicle_load_backend(BACKEND_BUILD_DIR);
    s_regsitered_devices = get_registered_devices();
    ASSERT_GT(s_regsitered_devices.size(), 0);
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

#ifdef EXT_FIELD
typedef testing::Types<scalar_t, extension_t> FTImplementations;
#else
typedef testing::Types<scalar_t> FTImplementations;
#endif

TYPED_TEST_SUITE(FieldApiTest, FTImplementations);

TYPED_TEST(FieldApiTest, vectorAddSync)
{
  const int N = 1 << 15;
  auto in_a = std::make_unique<TypeParam[]>(N);
  auto in_b = std::make_unique<TypeParam[]>(N);
  generate_scalars(in_a.get(), N);
  generate_scalars(in_b.get(), N);

  auto out_cpu = std::make_unique<TypeParam[]>(N);
  auto out_cuda = std::make_unique<TypeParam[]>(N);

  auto run = [&](const char* dev_type, TypeParam* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();

    START_TIMER(VECADD_sync)
    for (int i = 0; i < iters; ++i)
      vector_add(in_a.get(), in_b.get(), N, config, out);
    END_TIMER(VECADD_sync, msg, measure);
  };

  // run("CUDA", out_cuda.get(), "CUDA vector add", false /*=measure*/, 1 /*=iters*/); // warmup

  run("CPU", out_cpu.get(), "CPU vector add", VERBOSE /*=measure*/, 16 /*=iters*/);
  // run("CUDA", out_cuda.get(), "CUDA vector add (host mem)", VERBOSE /*=measure*/, 16 /*=iters*/);

  // ASSERT_EQ(0, memcmp(out_cpu.get(), out_cuda.get(), N * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, vectorAddAsync)
{
  const int N = 1 << 15;
  auto in_a = std::make_unique<TypeParam[]>(N);
  auto in_b = std::make_unique<TypeParam[]>(N);
  generate_scalars(in_a.get(), N);
  generate_scalars(in_b.get(), N);

  auto out_cpu = std::make_unique<TypeParam[]>(N);
  auto out_cuda = std::make_unique<TypeParam[]>(N);

  auto run = [&](const char* dev_type, TypeParam* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    // const bool is_cpu = std::string("CPU") == dev.type;

    TypeParam *d_in_a, *d_in_b, *d_out;
    icicleStreamHandle stream;
    icicle_create_stream(&stream);
    icicle_malloc_async((void**)&d_in_a, N * sizeof(TypeParam), stream);
    icicle_malloc_async((void**)&d_in_b, N * sizeof(TypeParam), stream);
    icicle_malloc_async((void**)&d_out, N * sizeof(TypeParam), stream);
    icicle_copy_to_device_async(d_in_a, in_a.get(), N * sizeof(TypeParam), stream);

    auto config = default_vec_ops_config();
    config.is_a_on_device = true;
    config.is_b_on_device = true;
    config.is_result_on_device = true;
    config.is_async = true;
    config.stream = stream;

    START_TIMER(VECADD_async);
    for (int i = 0; i < iters; ++i) {
      vector_add(d_in_a, d_in_b, N, config, d_out);
    }
    END_TIMER(VECADD_async, msg, measure);

    icicle_copy_to_host_async(out, d_out, N * sizeof(TypeParam), stream);
    icicle_stream_synchronize(stream);

    icicle_free_async(d_in_a, stream);
    icicle_free_async(d_in_b, stream);
    icicle_free_async(d_out, stream);
  };

  run("CPU", out_cpu.get(), "CPU vector add", VERBOSE /*=measure*/, 16 /*=iters*/);
  // run("CUDA", out_cuda.get(), "CUDA vector add (device mem)", VERBOSE /*=measure*/, 16 /*=iters*/);

  // ASSERT_EQ(0, memcmp(out_cpu.get(), out_cuda.get(), N * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, Ntt)
{
  const int logn = 15;
  const int N = 1 << logn;
  auto scalars = std::make_unique<TypeParam[]>(N);
  generate_scalars(scalars.get(), N);

  auto out_cpu = std::make_unique<TypeParam[]>(N);
  auto out_cuda = std::make_unique<TypeParam[]>(N);

  auto run = [&](const char* dev_type, TypeParam* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    ntt_init_domain(scalar_t::omega(logn), ConfigExtension());

    auto config = default_ntt_config<scalar_t>();

    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i)
      ntt(scalars.get(), N, NTTDir::kForward, config, out);
    END_TIMER(NTT_sync, msg, measure);

    ntt_release_domain<scalar_t>();
  };

  run("CPU", out_cpu.get(), "CPU ntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  // run("CUDA", out_cuda.get(), "CUDA ntt (host mem)", VERBOSE /*=measure*/, 1 /*=iters*/);

  // ASSERT_EQ(0, memcmp(out_cpu.get(), out_cuda.get(), N * sizeof(scalar_t)));
}

TYPED_TEST(FieldApiTest, CpuVecAPIs)
{
  const int N = 1 << 15;
  auto in_a = std::make_unique<TypeParam[]>(N);
  auto in_b = std::make_unique<TypeParam[]>(N);
  generate_scalars(in_a.get(), N);
  generate_scalars(in_b.get(), N);

  auto out_cpu_add = std::make_unique<TypeParam[]>(N);
  auto out_cpu_sub = std::make_unique<TypeParam[]>(N);
  auto out_cpu_mul = std::make_unique<TypeParam[]>(N);

  Device dev = {"CPU", 0};
  icicle_set_device(dev);
  auto config = default_vec_ops_config();

  START_TIMER(VEC_OPS)
  scalar_convert_montgomery(in_a.get(), N, true /*into montgomery*/, config);
  scalar_convert_montgomery(in_b.get(), N, true /*into montgomery*/, config);
  vector_add(in_a.get(), in_b.get(), N, config, out_cpu_add.get());
  vector_sub(in_a.get(), in_b.get(), N, config, out_cpu_sub.get());
  vector_mul(in_a.get(), in_b.get(), N, config, out_cpu_mul.get());
  END_TIMER(VEC_OPS, "CPU vec ops took", VERBOSE);

  // TODO real test
  const int test_idx = N >> 1;
  ASSERT_EQ(out_cpu_add[test_idx], in_a[test_idx] + in_b[test_idx]);
  ASSERT_EQ(out_cpu_sub[test_idx], in_a[test_idx] - in_b[test_idx]);
  ASSERT_EQ(out_cpu_mul[test_idx], in_a[test_idx] * in_b[test_idx]);
}

TYPED_TEST(FieldApiTest, CpuMatrixAPIs)
{
  const int R = 1 << 10, C = 1 << 6;
  auto in = std::make_unique<scalar_t[]>(R * C);
  generate_scalars(in.get(), R * C);

  auto out_cpu_transpose = std::make_unique<scalar_t[]>(R * C);

  Device dev = {"CPU", 0};
  icicle_set_device(dev);
  auto config = default_matrix_ops_config();

  START_TIMER(MATRIX_OPS)
  ICICLE_CHECK(matrix_transpose(in.get(), R, C, config, out_cpu_transpose.get()));
  END_TIMER(MATRIX_OPS, "CPU matrix ops took", VERBOSE);

  // TODO verify
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}