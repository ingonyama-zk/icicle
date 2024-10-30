#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"
#include <new>
#include <random>
#include <cstdlib> // For system

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
static int ITERS = 1;
static inline std::string s_main_target;
static inline std::string s_reference_target;
// static const bool s_is_cuda_registered = is_device_registered("CUDA");
bool s_is_cuda_registered;

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

    s_is_cuda_registered = is_device_registered("CUDA");
    if (!s_is_cuda_registered) { ICICLE_LOG_ERROR << "CUDA device not found. Testing CPU vs reference (on cpu)"; }
    s_main_target = s_is_cuda_registered ? "CUDA" : "CPU";
    s_reference_target = "CPU";
  }
  static void TearDownTestSuite()
  {
    // make sure to fail in CI if only have one device
    ICICLE_ASSERT(is_device_registered("CUDA")) << "missing CUDA backend";
  }

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

TYPED_TEST(FieldApiTest, vectorVectorOps)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << (rand() % 15 + 3);
  // const uint64_t N = 1 << (3);
  const int batch_size = 1 << (rand() % 5);
  // const int batch_size = 2;
  const bool columns_batch = rand() % 2;
  const int total_size = N * batch_size;
  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto in_b = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);
  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  auto vector_accumulate_wrapper =
    [](TypeParam* a, const TypeParam* b, uint64_t size, const VecOpsConfig& config, TypeParam* /*out*/) {
      return vector_accumulate(a, b, size, config);
    };

  auto run =
    [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);
      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;

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

  // warmup
  // run(s_reference_target, out_ref.get(), false /*=measure*/, 16 /*=iters*/);
  // run(s_main_target, out_main.get(), false /*=measure*/, 1 /*=iters*/);

  // Element-wise vector operations
  // If config.batch_size>1, (columns_batch=true or false) the operation is done element-wise anyway, so it doesn't
  // affect the test

  // // add
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!s_is_cuda_registered) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] + in_b[i];
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // // accumulate
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  // if (!s_is_cuda_registered) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] + in_b[i];
    }
  // } else {
    // run(s_reference_target, nullptr, VERBOSE /*=measure*/, vector_accumulate_wrapper, "vector accumulate", ITERS);
  // }
  run(s_main_target, nullptr, VERBOSE /*=measure*/, vector_accumulate_wrapper, "vector accumulate", ITERS);

  // for (int i = 0; i < total_size; i++) {
  //   ICICLE_LOG_DEBUG << i << ", " << in_a[i] << ", " << in_b[i] << ", " << out_ref[i];
  // }

  ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // // sub
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!s_is_cuda_registered) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] - in_b[i];
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // // mul
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!s_is_cuda_registered) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] * in_b[i];
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // // div
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  // reference
  if (!s_is_cuda_registered) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] * TypeParam::inverse(in_b[i]);
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_div<TypeParam>, "vector div", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_div<TypeParam>, "vector div", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, montgomeryConversion)
{
  int seed = time(0);
  srand(seed);
  // ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << (rand() % 15 + 3);
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;
  const bool is_to_montgomery = rand() % 2;
  const int total_size = N * batch_size;
  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(MONTGOMERY)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(convert_montgomery(in_a.get(), N, is_to_montgomery, config, out));
    }
    END_TIMER(MONTGOMERY, oss.str().c_str(), measure);
  };

  // Element-wise operation
  // If config.batch_size>1, (columns_batch=true or false) the addition is done element-wise anyway, so it doesn't
  // affect the test

  // convert_montgomery
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // reference
  if (!s_is_cuda_registered) {
    if (is_to_montgomery) {
      for (int i = 0; i < total_size; i++) {
        out_ref[i] = TypeParam::to_montgomery(in_a[i]);
      }
    } else {
      for (int i = 0; i < total_size; i++) {
        out_ref[i] = TypeParam::from_montgomery(in_a[i]);
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, "montgomery", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, "montgomery", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, VectorReduceOps)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << (rand() % 15 + 3);
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;
  const int total_size = N * batch_size;

  // const uint64_t N = 1 << (20);
  // const int batch_size = 1 << 4;
  // const bool columns_batch = 1;
  // const int total_size = N * batch_size;

  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(batch_size);
  auto out_ref = std::make_unique<TypeParam[]>(batch_size);

  auto vector_accumulate_wrapper =
    [](TypeParam* a, const TypeParam* b, uint64_t size, const VecOpsConfig& config, TypeParam* /*out*/) {
      return vector_accumulate(a, b, size, config);
    };

  auto run =
    [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);
      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(in_a.get(), N, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // // sum
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // reference
  for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
    out_ref[idx_in_batch] = TypeParam::from(0);
  }
  if (!s_is_cuda_registered) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_a = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_in_batch] = out_ref[idx_in_batch] + in_a[idx_a];
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_sum<TypeParam>, "vector sum", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_sum<TypeParam>, "vector sum", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(TypeParam)));

  // // product
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  if (!s_is_cuda_registered) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      out_ref[idx_in_batch] = TypeParam::from(1);
    }
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_a = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_in_batch] = out_ref[idx_in_batch] * in_a[idx_a];
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_product<TypeParam>, "vector product", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_product<TypeParam>, "vector product", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, scalarVectorOps)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << (rand() % 15 + 3);
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;
  const bool use_single_scalar = rand() % 2;

  // const uint64_t N = 1 << (4);
  // const int batch_size = 7;
  // const bool columns_batch = 1;
  // const bool use_single_scalar = 0;

  const int total_size = N * batch_size;
  auto scalar_a = std::make_unique<TypeParam[]>(use_single_scalar ? 1 : batch_size);
  auto in_b = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);
  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "use_single_scalar = " << use_single_scalar;

  auto vector_accumulate_wrapper =
    [](TypeParam* a, const TypeParam* b, uint64_t size, const VecOpsConfig& config, TypeParam* /*out*/) {
      return vector_accumulate(a, b, size, config);
    };

  auto run =
    [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);
      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(scalar_a.get(), in_b.get(), N, use_single_scalar, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // // scalar add vec
  FieldApiTest<TypeParam>::random_samples(scalar_a.get(), (use_single_scalar ? 1 : batch_size));
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);

  // reference
  if (!s_is_cuda_registered) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_b = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_b] = (use_single_scalar ? scalar_a[0] : scalar_a[idx_in_batch]) + in_b[idx_b];
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, scalar_add_vec<TypeParam>, "scalar add vec", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, scalar_add_vec<TypeParam>, "scalar add vec", ITERS);

  
  // ICICLE_LOG_DEBUG << scalar_a[0] << ", ";
  // ICICLE_LOG_DEBUG << scalar_a[1] << ", ";
  // for (int i = 0; i < total_size; i++) {
  //   ICICLE_LOG_DEBUG << i << ", " << in_b[i] << ", " << out_main[i] << ", " << out_ref[i];
  // }
  
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // scalar sub vec
  FieldApiTest<TypeParam>::random_samples(scalar_a.get(), (use_single_scalar ? 1 : batch_size));
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);

  if (!s_is_cuda_registered) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_b = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_b] = (use_single_scalar ? scalar_a[0] : scalar_a[idx_in_batch]) - in_b[idx_b];
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, scalar_sub_vec<TypeParam>, "scalar sub vec", ITERS);
  }

  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, scalar_sub_vec<TypeParam>, "scalar sub vec", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // // scalar mul vec
  FieldApiTest<TypeParam>::random_samples(scalar_a.get(), (use_single_scalar ? 1 : batch_size));
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);

  if (!s_is_cuda_registered) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_b = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_b] = (use_single_scalar ? scalar_a[0] : scalar_a[idx_in_batch]) * in_b[idx_b];
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, scalar_mul_vec<TypeParam>, "scalar mul vec", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, scalar_mul_vec<TypeParam>, "scalar mul vec", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, matrixAPIsAsync)
{
  int seed = time(0);
  srand(seed);
  // ICICLE_LOG_DEBUG << "seed = " << seed;
  const int R =
    1 << (rand() % 8 + 2); // cpu implementation for out of place trancpose also supports sizes wich are not powers of 2
  const int C =
    1 << (rand() % 8 + 2); // cpu implementation for out of place trancpose also supports sizes wich are not powers of 2
  const int batch_size = 1 << (rand() % 4);
  const bool columns_batch = rand() % 2;
  const bool is_in_place = s_is_cuda_registered? 0 : rand() % 2; //TODO - fix inplace (Hadar: I'm not sure we should support it)

  // const int R = 4; // cpu implementation for out of place trancpose also supports sizes wich are not powers of 2
  // const int C = 3;
  // const int batch_size = 1 << (1);
  // const bool columns_batch = 1;
  // const bool is_in_place = 1;

  // ICICLE_LOG_DEBUG << "R = " << R << ", C = " << C << ", batch_size = " << batch_size << ", columns_batch = " <<
  // columns_batch << ", is_in_place = " << is_in_place; //TODO SHANIE - remove this
  const int total_size = R * C * batch_size;
  auto h_inout = std::make_unique<TypeParam[]>(total_size);
  auto h_out_main = std::make_unique<TypeParam[]>(total_size);
  auto h_out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run = [&](const std::string& dev_type, TypeParam* h_out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    DeviceProperties device_props;
    icicle_get_device_properties(device_props);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    // Note: if the device uses host memory, do not allocate device memory and copy

    TypeParam *d_in, *d_out;
    if (!device_props.using_host_memory) {
      icicle_create_stream(&config.stream);
      icicle_malloc_async((void**)&d_in, total_size * sizeof(TypeParam), config.stream);
      icicle_malloc_async((void**)&d_out, total_size * sizeof(TypeParam), config.stream);
      icicle_copy_to_device_async(d_in, h_inout.get(), total_size * sizeof(TypeParam), config.stream);

      config.is_a_on_device = true;
      config.is_result_on_device = true;
      config.is_async = false;
    }

    TypeParam* in = device_props.using_host_memory ? h_inout.get() : d_in;
    TypeParam* out = device_props.using_host_memory ? h_out : d_out;

    START_TIMER(TRANSPOSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(matrix_transpose(in, R, C, config, out));
    }
    END_TIMER(TRANSPOSE, oss.str().c_str(), measure);

    if (!device_props.using_host_memory) {
      icicle_copy_to_host_async(h_out, d_out, total_size * sizeof(TypeParam), config.stream);
      icicle_stream_synchronize(config.stream);
      icicle_free_async(d_in, config.stream);
      icicle_free_async(d_out, config.stream);
    }
  };

  // // Option 1: Initialize each input matrix in the batch with the same ascending values
  // for (uint32_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
  //   for (uint32_t i = 0; i < R * C; i++) {
  //     if(columns_batch){
  //       h_inout[idx_in_batch + batch_size * i] = TypeParam::from(i);
  //     } else {
  //       h_inout[idx_in_batch * R * C + i] = TypeParam::from(i);
  //     }
  //   }
  // }

  // // Option 2: Initialize the entire input array with ascending values
  // for (int i = 0; i < total_size; i++) {
  //   h_inout[i] = TypeParam::from(i);
  // }

  // Option 3: Initialize the entire input array with random values
  FieldApiTest<TypeParam>::random_samples(h_inout.get(), total_size);

  // Reference implementation
  if (!s_is_cuda_registered) {
    const TypeParam* cur_mat_in = h_inout.get();
    TypeParam* cur_mat_out = h_out_ref.get();
    uint32_t stride = columns_batch ? batch_size : 1;
    const uint64_t total_elements_one_mat = static_cast<uint64_t>(R) * C;
    for (uint32_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      // Perform the matrix transpose
      for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = 0; j < C; ++j) {
          cur_mat_out[stride * (j * R + i)] = cur_mat_in[stride * (i * C + j)];
        }
      }
      cur_mat_in += (columns_batch ? 1 : total_elements_one_mat);
      cur_mat_out += (columns_batch ? 1 : total_elements_one_mat);
    }
  } else {
    run(s_reference_target, (is_in_place ? h_inout.get() : h_out_ref.get()), VERBOSE /*=measure*/, "transpose", ITERS);
  }

  run(s_main_target, (is_in_place ? h_inout.get() : h_out_main.get()), VERBOSE /*=measure*/, "transpose", ITERS);

   // ICICLE_LOG_DEBUG << scalar_a[0] << ", ";
  // for (int i = 0; i < total_size; i++) {
  //   ICICLE_LOG_DEBUG << i << ", " << h_inout[i] << ", " << h_out_main[i] << ", " << h_out_ref[i];
  // }

  if (is_in_place) {
    ASSERT_EQ(0, memcmp(h_inout.get(), h_out_ref.get(), total_size * sizeof(TypeParam)));
  } else {
    // std::cout << "h_out_main:\t["; for (int i = 0; i < total_size-1; i++) { std::cout << h_out_main[i] << ", "; }
    // std::cout <<h_out_main[total_size-1]<<"]"<< std::endl; std::cout << " h_out_ref:\t["; for (int i = 0; i <
    // total_size-1; i++) { std::cout <<  h_out_ref[i] << ", "; } std::cout << h_out_ref[total_size-1]<<"]"<< std::endl;
    ASSERT_EQ(0, memcmp(h_out_main.get(), h_out_ref.get(), total_size * sizeof(TypeParam)));
    // }}//for loop TODO SHANIE - remove this
  }
}

TYPED_TEST(FieldApiTest, bitReverse)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << (rand() % 15 + 3);
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;
  const bool is_in_place = rand() % 2;
  const int total_size = N * batch_size;

  // const uint64_t N = 1 << (3);
  // const int batch_size = 1 << (1);
  // const bool columns_batch = 1;
  // const bool is_in_place = 0;
  // const int total_size = N * batch_size;

  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(BIT_REVERSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(bit_reverse(in_a.get(), N, config, out));
    }
    END_TIMER(BIT_REVERSE, oss.str().c_str(), measure);
  };

  // // Option 1: Initialize each input vector in the batch with the same ascending values
  // for (uint32_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
  //   for (uint32_t i = 0; i < N; i++) {
  //     if(columns_batch){
  //       in_a[idx_in_batch + batch_size * i] = TypeParam::from(i);
  //     } else {
  //       in_a[idx_in_batch * N + i] = TypeParam::from(i);
  //     }
  //   }
  // }

  // // Option 2: Initialize the entire input array with ascending values
  // for (int i = 0; i < total_size; i++) {
  //   in_a[i] = TypeParam::from(i);
  // }

  // Option 3: Initialize the entire input array with random values
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);

  // Reference implementation
  if (!s_is_cuda_registered || is_in_place) {
    uint64_t logn = 0;
    uint64_t temp = N;
    while (temp > 1) {
      temp >>= 1;
      logn++;
    }
    // BIT REVERSE FUNCTION
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t i = 0; i < N; i++) {
        int rev = 0;
        for (int j = 0; j < logn; ++j) {
          if (i & (1 << j)) { rev |= 1 << (logn - 1 - j); }
        }
        if (columns_batch) {
          out_ref[idx_in_batch + batch_size * i] = in_a[idx_in_batch + batch_size * rev];
          // ICICLE_LOG_DEBUG << "out_ref[" << idx_in_batch + batch_size * i << "] = in_a[" << idx_in_batch + batch_size
          // * rev << "]";
        } else {
          out_ref[idx_in_batch * N + i] = in_a[idx_in_batch * N + rev];
          // ICICLE_LOG_DEBUG << "out_ref[" << idx_in_batch * N + i << "] = in_a[" << idx_in_batch * N + rev << "]";
        }
      }
    }
  } else {
    run(s_reference_target, (is_in_place ? in_a.get() : out_ref.get()), VERBOSE /*=measure*/, "bit-reverse", 1);
  }
  run(s_main_target, (is_in_place ? in_a.get() : out_main.get()), VERBOSE /*=measure*/, "bit-reverse", 1);

  //   for (int i = 0; i < total_size; i++) {
  //   ICICLE_LOG_DEBUG << i << ", " << in_a[i] << ", " << out_main[i] << ", " << out_ref[i];
  // }

  if (is_in_place) {
    ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), N * sizeof(TypeParam)));
  } else {
    // std::cout << "out_main:\t["; for (int i = 0; i < total_size-1; i++) { std::cout << out_main[i] << ", "; }
    // std::cout <<out_main[total_size-1]<<"]"<< std::endl; std::cout << "out_ref:\t["; for (int i = 0; i <
    // total_size-1; i++) { std::cout <<  out_ref[i] << ", "; } std::cout << out_ref[total_size-1]<<"]"<< std::endl;
    ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
  }
}

TYPED_TEST(FieldApiTest, Slice)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t size_in = 1 << (rand() % 15 + 5);
  const uint64_t offset = rand() % 15;
  const uint64_t stride = rand() % 4 + 1;
  const uint64_t size_out = rand() % (((size_in - offset) / stride) - 1) + 1;
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;

  // const uint64_t size_in = 1 << (20);
  // const uint64_t offset = 97;
  // const uint64_t stride = 6;
  // const uint64_t size_out = (((size_in - offset) / stride) - 1) - 100;

  // ICICLE_LOG_DEBUG << size_in <<", "<< offset<<", "<<stride<<", "<<size_out;

  // const int batch_size = 50;
  // const bool columns_batch = 1;


  const int total_size_in = size_in * batch_size;
  const int total_size_out = size_out * batch_size;
  // ICICLE_LOG_DEBUG << "size_in = " << size_in << ", offset = " << offset << ", stride = " << stride << ", size_out =
  // " << size_out << ", batch_size = " << batch_size << ", columns_batch = " << columns_batch;

  auto in_a = std::make_unique<TypeParam[]>(total_size_in);
  auto out_main = std::make_unique<TypeParam[]>(total_size_out);
  auto out_ref = std::make_unique<TypeParam[]>(total_size_out);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(SLICE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(slice(in_a.get(), offset, stride, size_in, size_out, config, out));
    }
    END_TIMER(SLICE, oss.str().c_str(), measure);
  };

  // // Option 1: Initialize each input vector in the batch with the same ascending values
  // for (uint32_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
  //   for (uint32_t i = 0; i < size_in; i++) {
  //     if(columns_batch){
  //       in_a[idx_in_batch + batch_size * i] = TypeParam::from(i);
  //     } else {
  //       in_a[idx_in_batch * size_in + i] = TypeParam::from(i);
  //     }
  //   }
  // }

  // // Option 2: Initialize the entire input array with ascending values
  // for (int i = 0; i < total_size_in; i++) {
  //   in_a[i] = TypeParam::from(i);
  // }

  // Option 3: Initialize the entire input array with random values
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size_in);

  // Reference implementation
  if (!s_is_cuda_registered) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t i = 0; i < size_out; i++) {
        if (columns_batch) {
          out_ref[idx_in_batch + batch_size * i] = in_a[idx_in_batch + batch_size * (offset + i * stride)];
        } else {
          out_ref[idx_in_batch * size_out + i] = in_a[idx_in_batch * size_in + (offset + i * stride)];
        }
      }
    }
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, "slice", 1);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, "slice", 1);
  // std::cout << "out_main\t["; for (int i = 0; i < total_size_out-1; i++) { std::cout << out_main[i] << ", "; }
  // std::cout <<out_main[total_size_out-1]<<"]"<< std::endl; std::cout << "out_ref:\t["; for (int i = 0; i <
  // total_size_out-1; i++) { std::cout <<  out_ref[i] << ", "; } std::cout << out_ref[total_size_out-1]<<"]"<<
  // std::endl;

  //   for (int i = 0; i < total_size_in; i++) {
  //   ICICLE_LOG_DEBUG << i << ", " << in_a[i] << ", " << out_main[i] << ", " << out_ref[i];
  // }

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size_out * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, highestNonZeroIdx)
{
  int seed = time(0);
  srand(seed);
  // ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << (rand() % 15 + 3);
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;
  // const uint64_t N = 1 << (3);
  // const int batch_size = 1 << (1);
  // const bool columns_batch = true;
  const int total_size = N * batch_size;

  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<int64_t[]>(batch_size);
  auto out_ref = std::make_unique<int64_t[]>(batch_size);

  auto run = [&](const std::string& dev_type, int64_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(highestNonZeroIdx)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(highest_non_zero_idx(in_a.get(), N, config, out));
    }
    END_TIMER(highestNonZeroIdx, oss.str().c_str(), measure);
  };

  // Initialize each entire vector with 1 at a random index. The highest non-zero index is the index with 1
  for (uint32_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
    if (!s_is_cuda_registered) { out_ref[idx_in_batch] = rand() % N; } // highest_non_zero_idx
    for (uint32_t i = 0; i < N; i++) {
      if (columns_batch) {
        in_a[idx_in_batch + batch_size * i] = TypeParam::from(i == out_ref[idx_in_batch] ? 1 : 0);
      } else {
        in_a[idx_in_batch * N + i] = TypeParam::from(i == out_ref[idx_in_batch] ? 1 : 0);
      }
    }
  }
  if (s_is_cuda_registered) { run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, "highest_non_zero_idx", 1); }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, "highest_non_zero_idx", 1);
  // std::cout << "out_main:\t["; for (int i = 0; i < batch_size-1; i++) { std::cout << out_main[i] << ", "; } std::cout
  // <<out_main[batch_size-1]<<"]"<< std::endl; std::cout << "out_ref:\t["; for (int i = 0; i < batch_size-1; i++) {
  // std::cout <<  out_ref[i] << ", "; } std::cout << out_ref[batch_size-1]<<"]"<< std::endl;
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, polynomialEval)
{
  int seed = time(0);
  srand(seed);
  // ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t coeffs_size = 1 << (rand() % 10 + 4);
  const uint64_t domain_size = 1 << (rand() % 8 + 2);
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;
  const int total_coeffs_size = coeffs_size * batch_size;

  auto in_coeffs = std::make_unique<TypeParam[]>(total_coeffs_size);
  auto in_domain = std::make_unique<TypeParam[]>(domain_size);
  auto out_main = std::make_unique<TypeParam[]>(total_coeffs_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_coeffs_size);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(polynomialEval)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(polynomial_eval(in_coeffs.get(), coeffs_size, in_domain.get(), domain_size, config, out));
    }
    END_TIMER(polynomialEval, oss.str().c_str(), measure);
  };

  FieldApiTest<TypeParam>::random_samples(in_coeffs.get(), total_coeffs_size);
  FieldApiTest<TypeParam>::random_samples(in_domain.get(), domain_size);

  // Reference implementation
  // TODO - Check in comperison with GPU implementation

  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, "polynomial_eval", 1);
  if (s_is_cuda_registered) {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, "polynomial_eval", 1);
    // std::cout << "out_main:\t["; for (int i = 0; i < total_coeffs_size-1; i++) { std::cout << out_main[i] << ", "; }
    // std::cout <<out_main[total_coeffs_size-1]<<"]"<< std::endl; std::cout << "out_ref:\t["; for (int i = 0; i <
    // total_coeffs_size-1; i++) { std::cout <<  out_ref[i] << ", "; } std::cout << out_ref[total_coeffs_size-1]<<"]"<<
    // std::endl;
    ASSERT_EQ(
      0, memcmp(
           out_main.get(), out_ref.get(),
           total_coeffs_size * sizeof(TypeParam))); // TODO - Check in comperison with GPU implementation
  }
}

TYPED_TEST(FieldApiTest, polynomialDivision)
{
  int seed = time(0);
  srand(seed);
  // ICICLE_LOG_DEBUG << "seed = " << seed;
  // const int64_t numerator_deg = 1 << 4;
  // const int64_t denumerator_deg = 1 << 2;
  // const uint64_t q_size = numerator_deg - denumerator_deg + 1;
  // const uint64_t r_size = numerator_deg + 1;
  const int64_t numerator_deg = 3;
  const int64_t denumerator_deg = 2;
  const uint64_t q_size = 2;
  const uint64_t r_size = 4;
  const int batch_size = 1 << (rand() % 5);
  const bool columns_batch = rand() % 2;

  const int64_t total_numerator_size = (numerator_deg + 1) * batch_size;
  const int64_t total_denumerator_size = (denumerator_deg + 1) * batch_size;
  const uint64_t total_q_size = q_size * batch_size;
  const uint64_t total_r_size = r_size * batch_size;

  auto numerator = std::make_unique<TypeParam[]>(total_numerator_size);
  auto denumerator = std::make_unique<TypeParam[]>(total_denumerator_size);
  auto q_out_main = std::make_unique<TypeParam[]>(total_q_size);
  auto r_out_main = std::make_unique<TypeParam[]>(total_r_size);
  auto q_out_ref = std::make_unique<TypeParam[]>(total_q_size);
  auto r_out_ref = std::make_unique<TypeParam[]>(total_r_size);

  auto run =
    [&](const std::string& dev_type, TypeParam* q_out, TypeParam* r_out, bool measure, const char* msg, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);
      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(polynomialDivision)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(polynomial_division(
          numerator.get(), numerator_deg, denumerator.get(), denumerator_deg, q_size, r_size, config, q_out, r_out));
      }
      END_TIMER(polynomialDivision, oss.str().c_str(), measure);
    };

  // // Option 1: Initialize input vectors with random values
  // FieldApiTest<TypeParam>::random_samples(numerator.get(), total_numerator_size);
  // FieldApiTest<TypeParam>::random_samples(denumerator.get(), total_denumerator_size);
  // // Reference implementation
  // TODO - Check in comperison with GPU implementation or implement a general reference implementation

  // Option 2: Initialize the numerator and denumerator with chosen example
  //           And the reference implementation for the example

  for (uint32_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
    if (columns_batch) {
      // numerator = 3x^3+4x^2+5
      numerator[idx_in_batch + 0 * batch_size] = TypeParam::from(5);
      numerator[idx_in_batch + 1 * batch_size] = TypeParam::from(0);
      numerator[idx_in_batch + 2 * batch_size] = TypeParam::from(4);
      numerator[idx_in_batch + 3 * batch_size] = TypeParam::from(3);
      // denumerator = x^2-1
      denumerator[idx_in_batch + 0 * batch_size] = TypeParam::from(0) - TypeParam::from(1);
      denumerator[idx_in_batch + 1 * batch_size] = TypeParam::from(0);
      denumerator[idx_in_batch + 2 * batch_size] = TypeParam::from(1);
      if (!s_is_cuda_registered) {
        // q_out_ref = 3x+4
        q_out_ref[idx_in_batch + 0 * batch_size] = TypeParam::from(4);
        q_out_ref[idx_in_batch + 1 * batch_size] = TypeParam::from(3);
        // r_out_ref = 3x+9
        r_out_ref[idx_in_batch + 0 * batch_size] = TypeParam::from(9);
        r_out_ref[idx_in_batch + 1 * batch_size] = TypeParam::from(3);
      }
    } else {
      // numerator = 3x^3+4x^2+5
      numerator[idx_in_batch * (numerator_deg + 1) + 0] = TypeParam::from(5);
      numerator[idx_in_batch * (numerator_deg + 1) + 1] = TypeParam::from(0);
      numerator[idx_in_batch * (numerator_deg + 1) + 2] = TypeParam::from(4);
      numerator[idx_in_batch * (numerator_deg + 1) + 3] = TypeParam::from(3);
      // denumerator = x^2-1
      denumerator[idx_in_batch * (denumerator_deg + 1) + 0] = TypeParam::from(0) - TypeParam::from(1);
      denumerator[idx_in_batch * (denumerator_deg + 1) + 1] = TypeParam::from(0);
      denumerator[idx_in_batch * (denumerator_deg + 1) + 2] = TypeParam::from(1);
      if (!s_is_cuda_registered) {
        // q_out_ref = 3x+4
        q_out_ref[idx_in_batch * q_size + 0] = TypeParam::from(4);
        q_out_ref[idx_in_batch * q_size + 1] = TypeParam::from(3);
        // r_out_ref = 3x+9
        r_out_ref[idx_in_batch * r_size + 0] = TypeParam::from(9);
        r_out_ref[idx_in_batch * r_size + 1] = TypeParam::from(3);
      }
    }
  }

  if (s_is_cuda_registered) {
    run(s_reference_target, q_out_ref.get(), r_out_ref.get(), VERBOSE /*=measure*/, "polynomial_division", 1);
  }
  // std::cout << "numerator:\t["; for (int i = 0; i < total_numerator_size-1; i++) { std::cout << numerator[i] << ", ";
  // } std::cout <<numerator[total_numerator_size-1]<<"]"<< std::endl; std::cout << "denumerator:\t["; for (int i = 0; i
  // < total_denumerator_size-1; i++) { std::cout << denumerator[i] << ", "; } std::cout
  // <<denumerator[total_denumerator_size-1]<<"]"<< std::endl; std::cout << "q_out_ref:\t["; for (int i = 0; i <
  // total_q_size-1; i++) { std::cout <<  q_out_ref[i] << ", "; } std::cout << q_out_ref[total_q_size-1]<<"]"<<
  // std::endl; std::cout << "r_out_ref:\t["; for (int i = 0; i < total_r_size-1; i++) { std::cout <<  r_out_ref[i] <<
  // ", "; } std::cout << r_out_ref[total_r_size-1]<<"]"<< std::endl;
  run(s_main_target, q_out_main.get(), r_out_main.get(), VERBOSE /*=measure*/, "polynomial_division", 1);
  ASSERT_EQ(0, memcmp(q_out_main.get(), q_out_ref.get(), total_q_size * sizeof(TypeParam)));
  ASSERT_EQ(0, memcmp(r_out_main.get(), r_out_ref.get(), total_r_size * sizeof(TypeParam)));
}

// #ifdef NTT
// TYPED_TEST(FieldApiTest, ntt)
// {
//   // ICICLE_LOG_INFO << "Current branch: " << get_current_branch();
//   ICICLE_LOG_DEBUG << "ICICLE_LOG_DEBUG";
//   // for (int i = 3; i < 23; ++i) {
//   // //Randomize configuration

//   // int seed = time(0) + i;
//   // // int seed = 1726493105;
//   // srand(seed);
//   // const bool inplace = rand() % 2;
//   // const int logn = rand() % 17 + 3;
//   // // const int logn = rand() % 14 + 3;
//   // // const int logn = 16;
//   // const uint64_t N = 1 << logn;
//   // const int log_ntt_domain_size = logn + 1;
//   // const int log_batch_size = rand() % 3;
//   // const int batch_size = 1 << log_batch_size;
//   // const Ordering ordering = static_cast<Ordering>(rand() % 4);
//   // bool columns_batch;
//   // if (logn == 7 || logn < 4) {
//   //   columns_batch = false; // currently not supported (icicle_v3/backend/cuda/src/ntt/ntt.cuh line 578)
//   // } else {
//   //   // columns_batch = true;
//   //   columns_batch = rand() % 2;
//   // }
//   // // const NTTDir dir = static_cast<NTTDir>(rand() % 2); // 0: forward, 1: inverse
//   // const NTTDir dir = static_cast<NTTDir>(0); // 0: forward, 1: inverse
//   // const int log_coset_stride = rand() % 3;
//   // scalar_t coset_gen;
//   // if (log_coset_stride) {
//   //   coset_gen = scalar_t::omega(logn + log_coset_stride);
//   // } else {
//   //   coset_gen = scalar_t::one();
//   // }

//   const bool inplace = false;
//   const int logn = 15;
//   const uint64_t N = 1 << logn;
//   const int log_ntt_domain_size = logn;
//   const int log_batch_size = 0;
//   const int batch_size = 1 << log_batch_size;
//   const Ordering ordering = static_cast<Ordering>(0);
//   bool columns_batch = false;
//   const NTTDir dir = static_cast<NTTDir>(0); // 0: forward, 1: inverse
//   const int log_coset_stride = 0;
//   scalar_t coset_gen;
//   if (log_coset_stride) {
//     coset_gen = scalar_t::omega(logn + log_coset_stride);
//   } else {
//     coset_gen = scalar_t::one();
//   }

// // TODO SHANIE : remove
//   // ICICLE_LOG_INFO << "NTT test: seed=" << seed;
//   // ICICLE_LOG_INFO << "NTT test: omega=" << scalar_t::omega(logn);
//   // ICICLE_LOG_INFO << "NTT test:s inplace=" << inplace;
//   ICICLE_LOG_INFO << "NTT test: logn=" << logn;
//   // ICICLE_LOG_INFO << "NTT test: log_ntt_domain_size=" << log_ntt_domain_size;
//   // ICICLE_LOG_INFO << "NTT test: log_batch_size=" << log_batch_size;
//   // ICICLE_LOG_INFO << "NTT test: columns_batch=" << columns_batch;
//   // ICICLE_LOG_INFO << "NTT test: ordering=" << int(ordering);
//   ICICLE_LOG_INFO << "NTT test: dir=" << (dir == NTTDir::kForward ? "forward" : "inverse");
//   ICICLE_LOG_INFO << "NTT test: log_coset_stride=" << log_coset_stride;
//   ICICLE_LOG_INFO << "NTT test: coset_gen=" << coset_gen;

//   const int total_size = N * batch_size;
//   auto scalars = std::make_unique<TypeParam[]>(total_size);
//   FieldApiTest<TypeParam>::random_samples(scalars.get(), total_size);
//   // for (int i = 0; i < total_size; i++) { scalars[i] = scalar_t::from(i); } //FIXME SHANIE: remove
//   auto out_main = std::make_unique<TypeParam[]>(total_size);
//   auto out_ref = std::make_unique<TypeParam[]>(total_size);
//   auto run = [&](const std::string& dev_type, TypeParam* out, const char* msg, bool measure, int iters) {
//     Device dev = {dev_type, 0};
//     icicle_set_device(dev);
//     icicleStreamHandle stream = nullptr;
//     ICICLE_CHECK(icicle_create_stream(&stream));
//     auto init_domain_config = default_ntt_init_domain_config();
//     init_domain_config.stream = stream;
//     init_domain_config.is_async = false;
//     ConfigExtension ext;
//     ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
//     init_domain_config.ext = &ext;
//     auto config = default_ntt_config<scalar_t>();
//     config.stream = stream;
//     config.coset_gen = coset_gen;
//     config.batch_size = batch_size;       // default: 1
//     config.columns_batch = columns_batch; // default: false
//     config.ordering = ordering;           // default: kNN
//     config.are_inputs_on_device = true;
//     config.are_outputs_on_device = true;
//     config.is_async = false;
//     ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log_ntt_domain_size), init_domain_config));
//     TypeParam *d_in, *d_out;
//     ICICLE_CHECK(icicle_malloc_async((void**)&d_in, total_size * sizeof(TypeParam), config.stream));
//     ICICLE_CHECK(icicle_malloc_async((void**)&d_out, total_size * sizeof(TypeParam), config.stream));
//     ICICLE_CHECK(icicle_copy_to_device_async(d_in, scalars.get(), total_size * sizeof(TypeParam), config.stream));
//     std::ostringstream oss;
//     oss << dev_type << " " << msg;
//     START_TIMER(NTT_sync)
//     for (int i = 0; i < iters; ++i) {
//       if (inplace) {
//         ICICLE_CHECK(ntt(d_in, N, dir, config, d_in));
//       } else {
//         ICICLE_CHECK(ntt(d_in, N, dir, config, d_out));
//       }
//     }
//     END_TIMER(NTT_sync, oss.str().c_str(), measure);

//     if (inplace) {
//       ICICLE_CHECK(icicle_copy_to_host_async(out, d_in, total_size * sizeof(TypeParam), config.stream));
//     } else {
//       ICICLE_CHECK(icicle_copy_to_host_async(out, d_out, total_size * sizeof(TypeParam), config.stream));
//     }
//     ICICLE_CHECK(icicle_free_async(d_in, config.stream));
//     ICICLE_CHECK(icicle_free_async(d_out, config.stream));
//     ICICLE_CHECK(icicle_stream_synchronize(config.stream));
//     ICICLE_CHECK(icicle_destroy_stream(stream));
//     ICICLE_CHECK(ntt_release_domain<scalar_t>());
//   };
//   // run(s_main_target, out_main.get(), "ntt", false /*=measure*/, 0 /*=iters*/); // warmup
//   run(s_reference_target, out_ref.get(), "V3ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
//   run(s_main_target, out_main.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
//   // std::cout << "left:\t["; for (int i = 0; i < total_size-1; i++) { std::cout << out_main[i] << ", "; } std::cout
//   <<out_main[total_size-1]<<"]"<< std::endl;
//   // std::cout << "right:\t["; for (int i = 0; i < total_size-1; i++) { std::cout << out_ref[i] << ", "; } std::cout
//   <<out_ref[total_size-1]<<"]"<< std::endl;

//   ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
// }
// #endif // NTT

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}