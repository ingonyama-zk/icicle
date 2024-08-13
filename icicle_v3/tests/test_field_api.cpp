
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"
#include <random>

#include "icicle/runtime.h"
#include "icicle/vec_ops.h"
#include "icicle/ntt.h"

#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"
#include "icicle/backend/ntt_config.h"

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
#ifdef BACKEND_BUILD_DIR
    setenv("ICICLE_BACKEND_INSTALL_DIR", BACKEND_BUILD_DIR, 0 /*=replace*/);
#endif
    icicle_load_backend_from_env_or_default();

    // check targets are loaded and choose main and reference targets
    auto regsitered_devices = get_registered_devices_list();
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

  void random_samples(T* arr, uint64_t count)
  {
    for (uint64_t i = 0; i < count; i++)
      arr[i] = i < 1000 ? T::rand_host() : arr[i - 1000];
  }
};

#ifdef EXT_FIELD
typedef testing::Types<scalar_t, extension_t> FTImplementations;
#else
typedef testing::Types<scalar_t> FTImplementations;
#endif

TYPED_TEST_SUITE(FieldApiTest, FTImplementations);

// Note: this is testing host arithmetic. Other tests against CPU backend should guarantee correct device arithmetic too
TYPED_TEST(FieldApiTest, FieldSanityTest)
{
  auto a = TypeParam::rand_host();
  auto b = TypeParam::rand_host();
  auto b_inv = TypeParam::inverse(b);
  auto a_neg = TypeParam::neg(a);
  ASSERT_EQ(a + TypeParam::zero(), a);
  ASSERT_EQ(a + b - a, b);
  ASSERT_EQ(b * a * b_inv, a);
  ASSERT_EQ(a + a_neg, TypeParam::zero());
  ASSERT_EQ(a * TypeParam::zero(), TypeParam::zero());
  ASSERT_EQ(b * b_inv, TypeParam::one());
  ASSERT_EQ(a * scalar_t::from(2), a + a);
}

TYPED_TEST(FieldApiTest, vectorOps)
{
  const uint64_t N = 1 << 22;
  auto in_a = std::make_unique<TypeParam[]>(N);
  auto in_b = std::make_unique<TypeParam[]>(N);
  FieldApiTest<TypeParam>::random_samples(in_a.get(), N);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), N);

  auto out_main = std::make_unique<TypeParam[]>(N);
  auto out_ref = std::make_unique<TypeParam[]>(N);

  auto vector_accumulate_wrapper = [](TypeParam* a, const TypeParam* b, uint64_t size, const VecOpsConfig& config, TypeParam* /*out*/) {
      return vector_accumulate(a, b, size, config);
  };

  auto run =
    [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
      Device dev = {dev_type, 0};
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

  //accumulate
  auto temp_result = std::make_unique<TypeParam[]>(N);
  auto initial_in_a = std::make_unique<TypeParam[]>(N);

  std::memcpy(initial_in_a.get(), in_a.get(), N * sizeof(TypeParam));
  run(s_reference_target, out_main.get(), VERBOSE /*=measure*/, vector_accumulate_wrapper, "vector accumulate", ITERS);
  std::memcpy(temp_result.get(), in_a.get(), N * sizeof(TypeParam));
  std::memcpy(in_a.get(), initial_in_a.get(), N * sizeof(TypeParam));
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_accumulate_wrapper, "vector accumulate", ITERS);
  ASSERT_EQ(0, memcmp(in_a.get(), temp_result.get(), N * sizeof(TypeParam)));

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
  auto h_in = std::make_unique<TypeParam[]>(R * C);
  FieldApiTest<TypeParam>::random_samples(h_in.get(), R * C);

  auto h_out_main = std::make_unique<TypeParam[]>(R * C);
  auto h_out_ref = std::make_unique<TypeParam[]>(R * C);

  auto run = [&](const std::string& dev_type, TypeParam* h_out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    DeviceProperties device_props;
    icicle_get_device_properties(device_props);
    auto config = default_vec_ops_config();

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    // Note: if the device uses host memory, do not allocate device memory and copy

    TypeParam *d_in, *d_out;
    if (!device_props.using_host_memory) {
      icicle_create_stream(&config.stream);
      icicle_malloc_async((void**)&d_in, R * C * sizeof(TypeParam), config.stream);
      icicle_malloc_async((void**)&d_out, R * C * sizeof(TypeParam), config.stream);
      icicle_copy_to_device_async(d_in, h_in.get(), R * C * sizeof(TypeParam), config.stream);

      config.is_a_on_device = true;
      config.is_result_on_device = true;
      config.is_async = false;
    }

    TypeParam* in = device_props.using_host_memory ? h_in.get() : d_in;
    TypeParam* out = device_props.using_host_memory ? h_out : d_out;

    START_TIMER(TRANSPOSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(matrix_transpose(in, R, C, config, out));
    }
    END_TIMER(TRANSPOSE, oss.str().c_str(), measure);

    if (!device_props.using_host_memory) {
      icicle_copy_to_host_async(h_out, d_out, R * C * sizeof(TypeParam), config.stream);
      icicle_stream_synchronize(config.stream);
      icicle_free_async(d_in, config.stream);
      icicle_free_async(d_out, config.stream);
    }
  };

  run(s_reference_target, h_out_ref.get(), VERBOSE /*=measure*/, "transpose", ITERS);
  run(s_main_target, h_out_main.get(), VERBOSE /*=measure*/, "transpose", ITERS);
  ASSERT_EQ(0, memcmp(h_out_main.get(), h_out_ref.get(), R * C * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, montgomeryConversion)
{
  const uint64_t N = 1 << 18;
  auto elements_main = std::make_unique<TypeParam[]>(N);
  auto elements_ref = std::make_unique<TypeParam[]>(N);
  FieldApiTest<TypeParam>::random_samples(elements_main.get(), N);
  memcpy(elements_ref.get(), elements_main.get(), N * sizeof(TypeParam));

  auto run = [&](const std::string& dev_type, TypeParam* inout, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(MONTGOMERY)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(convert_montgomery(inout, N, true /*into montgomery*/, config, inout));
    }
    END_TIMER(MONTGOMERY, oss.str().c_str(), measure);
  };

  run(s_reference_target, elements_main.get(), VERBOSE /*=measure*/, "montgomery", 1);
  run(s_main_target, elements_ref.get(), VERBOSE /*=measure*/, "montgomery", 1);
  ASSERT_EQ(0, memcmp(elements_main.get(), elements_ref.get(), N * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, bitReverse)
{
  const uint64_t N = 1 << 18;
  auto elements_main = std::make_unique<TypeParam[]>(N);
  auto elements_ref = std::make_unique<TypeParam[]>(N);
  FieldApiTest<TypeParam>::random_samples(elements_main.get(), N);
  memcpy(elements_ref.get(), elements_main.get(), N * sizeof(TypeParam));

  auto run = [&](const std::string& dev_type, TypeParam* inout, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(BIT_REVERSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(bit_reverse(inout, N, config, inout));
    }
    END_TIMER(BIT_REVERSE, oss.str().c_str(), measure);
  };

  run(s_reference_target, elements_main.get(), VERBOSE /*=measure*/, "bit-reverse", 1);
  run(s_main_target, elements_ref.get(), VERBOSE /*=measure*/, "bit-reverse", 1);
  ASSERT_EQ(0, memcmp(elements_main.get(), elements_ref.get(), N * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, Slice)
{
  const uint64_t N = 1 << 18;
  const uint64_t offset = 2;
  const uint64_t stride = 3;
  const uint64_t size = 4;

  auto elements_main = std::make_unique<TypeParam[]>(N);
  auto elements_ref = std::make_unique<TypeParam[]>(size);
  auto elements_out = std::make_unique<TypeParam[]>(size);

  FieldApiTest<TypeParam>::random_samples(elements_main.get(), N);

  auto run =
    [&](const std::string& dev_type, const TypeParam* in, TypeParam* out, bool measure, const char* msg, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);
      auto config = VecOpsConfig(); // Adjust configuration as needed

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(SLICE)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(slice(in, offset, stride, size, config, out));
      }
      END_TIMER(SLICE, oss.str().c_str(), measure);
    };

  run(s_reference_target, elements_main.get(), elements_ref.get(), VERBOSE /*=measure*/, "slice", 1);
  run(s_main_target, elements_main.get(), elements_out.get(), VERBOSE /*=measure*/, "slice", 1);
  ASSERT_EQ(0, memcmp(elements_ref.get(), elements_out.get(), size * sizeof(TypeParam)));
}

#ifdef NTT_ENABLED
TYPED_TEST(FieldApiTest, ntt)
{
  srand(time(0));

  // Randomize config
  const int logn = rand() % 10 + 3;
  const uint64_t N = 1 << logn;
  const int log_ntt_domain_size = 17;
  const int log_batch_size = rand() % 10 + 3;
  const int batch_size = 1 << log_batch_size;
  const Ordering ordering = static_cast<Ordering>(rand() % 4);
  bool columns_batch;
  if (logn == 7 || logn < 4) {
    columns_batch = false; // currently not supported (icicle_v3/backend/cuda/src/ntt/ntt.cuh line 578)
  } else {
    columns_batch = rand() % 2;
  }

  const NTTDir dir = static_cast<NTTDir>(rand() % 2); // 0: forward, 1: inverse
  const int log_coset_stride = rand() % 4;
  scalar_t coset_gen;
  if (log_coset_stride) {
    coset_gen = scalar_t::omega(logn + log_coset_stride);
  } else {
    coset_gen = scalar_t::one();
  }

  const int total_size = N * batch_size;
  auto scalars = std::make_unique<TypeParam[]>(total_size);
  FieldApiTest<TypeParam>::random_samples(scalars.get(), total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run = [&](const std::string& dev_type, TypeParam* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    icicleStreamHandle stream = nullptr;
    ICICLE_CHECK(icicle_create_stream(&stream));

    auto init_domain_config = default_ntt_init_domain_config();
    init_domain_config.stream = stream;
    init_domain_config.is_async = false;
    ConfigExtension ext;
    ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
    init_domain_config.ext = &ext;

    auto config = default_ntt_config<scalar_t>();
    config.stream = stream;
    config.coset_gen = coset_gen;
    config.batch_size = batch_size;       // default: 1
    config.columns_batch = columns_batch; // default: false
    config.ordering = ordering;           // default: kNN
    config.are_inputs_on_device = true;
    config.are_outputs_on_device = true;
    config.is_async = false;
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log_ntt_domain_size), init_domain_config));

    TypeParam *d_in, *d_out;
    ICICLE_CHECK(icicle_malloc_async((void**)&d_in, total_size * sizeof(TypeParam), config.stream));
    ICICLE_CHECK(icicle_malloc_async((void**)&d_out, total_size * sizeof(TypeParam), config.stream));
    ICICLE_CHECK(icicle_copy_to_device_async(d_in, scalars.get(), total_size * sizeof(TypeParam), config.stream));

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(ntt(d_in, N, dir, config, d_out));
    }
    END_TIMER(NTT_sync, oss.str().c_str(), measure);

    ICICLE_CHECK(icicle_copy_to_host_async(out, d_out, total_size * sizeof(TypeParam), config.stream));
    ICICLE_CHECK(icicle_free_async(d_in, config.stream));
    ICICLE_CHECK(icicle_free_async(d_out, config.stream));
    ICICLE_CHECK(icicle_stream_synchronize(config.stream));
    ICICLE_CHECK(icicle_destroy_stream(stream));

    ICICLE_CHECK(ntt_release_domain<scalar_t>());
  };

  run(s_main_target, out_main.get(), "ntt", false /*=measure*/, 1 /*=iters*/); // warmup

  run(s_reference_target, out_ref.get(), "ntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  run(s_main_target, out_main.get(), "ntt", VERBOSE /*=measure*/, 1 /*=iters*/);

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}
#endif // NTT_ENABLED

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}