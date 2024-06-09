
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
static int ITERS = 16;
static inline std::string s_main_target;
static inline std::string s_reference_target;
template <typename T>
class FieldApiTest : public ::testing::Test
{
public:
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    icicle_load_backend(BACKEND_BUILD_DIR);

    // check targets are loaded and choose main and reference targets
    auto regsitered_devices = get_registered_devices();
    ASSERT_GE(regsitered_devices.size(), 2);

    const bool is_cuda_registered = is_device_registered("CUDA");
    const bool is_cpu_registered = is_device_registered("CPU");
    const bool is_cpu_ref_registered = is_device_registered("CPU_REF");
    // if cuda is available, want main="CUDA", ref="CPU", otherwise main="CPU", ref="CPU_REF".
    s_main_target = is_cuda_registered ? "CUDA" : "CPU";
    s_reference_target = is_cuda_registered ? "CPU" : "CPU_REF";
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

TYPED_TEST(FieldApiTest, vectorOps)
{
  const int N = 1 << 15;
  auto in_a = std::make_unique<TypeParam[]>(N);
  auto in_b = std::make_unique<TypeParam[]>(N);
  TypeParam::rand_host_many(in_a.get(), N);
  TypeParam::rand_host_many(in_b.get(), N);

  auto out_main = std::make_unique<TypeParam[]>(N);
  auto out_ref = std::make_unique<TypeParam[]>(N);

  auto run =
    [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
      Device dev = {dev_type.c_str(), 0};
      icicle_set_device(dev);
      auto config = default_vec_ops_config();

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(in_a.get(), in_b.get(), N, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // warmup
  // run(s_reference_target, out_ref.get(), false /*=measure*/, 16 /*=iters*/);
  // run(s_main_target, out_main.get(), false /*=measure*/, 1 /*=iters*/);

  // add
  run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), N * sizeof(TypeParam)));

  // sub
  run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), N * sizeof(TypeParam)));

  // mul
  run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), N * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, matrixAPIsAsync)
{
  const int R = 1 << 10, C = 1 << 8;
  auto in = std::make_unique<TypeParam[]>(R * C);
  TypeParam::rand_host_many(in.get(), R * C);

  auto out_main = std::make_unique<TypeParam[]>(R * C);
  auto out_ref = std::make_unique<TypeParam[]>(R * C);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type.c_str(), 0};
    icicle_set_device(dev);
    auto config = default_matrix_ops_config();

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    TypeParam *d_in, *d_out;
    icicle_create_stream(&config.stream);
    icicle_malloc_async((void**)&d_in, R * C * sizeof(TypeParam), config.stream);
    icicle_malloc_async((void**)&d_out, R * C * sizeof(TypeParam), config.stream);
    icicle_copy_to_device_async(d_in, in.get(), R * C * sizeof(TypeParam), config.stream);

    config.is_input_on_device = true;
    config.is_output_on_device = true;
    config.is_async = true;

    START_TIMER(TRANSPOSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(matrix_transpose(d_in, R, C, config, d_out));
    }
    END_TIMER(TRANSPOSE, oss.str().c_str(), measure);

    icicle_copy_to_host_async(out, d_out, R * C * sizeof(TypeParam), config.stream);
    icicle_stream_synchronize(config.stream);
    icicle_free_async(d_in, config.stream);
    icicle_free_async(d_out, config.stream);
  };

  run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, "transpose", ITERS);
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, "transpose", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), R * C * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, montgomeryConversion)
{
  const int N = 1 << 18;
  auto elements_main = std::make_unique<TypeParam[]>(N);
  auto elements_ref = std::make_unique<TypeParam[]>(N);
  TypeParam::rand_host_many(elements_main.get(), N);
  memcpy(elements_ref.get(), elements_main.get(), N * sizeof(TypeParam));

  auto run = [&](const std::string& dev_type, TypeParam* inout, bool measure, const char* msg, int iters) {
    Device dev = {dev_type.c_str(), 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(MONTGOMERY)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(scalar_convert_montgomery(inout, N, true /*into montgomery*/, config));
    }
    END_TIMER(MONTGOMERY, oss.str().c_str(), measure);
  };

  run(s_reference_target, elements_main.get(), VERBOSE /*=measure*/, "montgomery", 1);
  run(s_main_target, elements_ref.get(), VERBOSE /*=measure*/, "montgomery", 1);
  ASSERT_EQ(0, memcmp(elements_main.get(), elements_ref.get(), N * sizeof(TypeParam)));
}

#ifndef NTT_DISABLED
TYPED_TEST(FieldApiTest, Ntt)
{
  const int logn = 15;
  const int N = 1 << logn;
  auto scalars = std::make_unique<TypeParam[]>(N);
  TypeParam::rand_host_many(scalars.get(), N);

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
#endif // NTT_DISABLED

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}