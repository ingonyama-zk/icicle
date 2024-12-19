#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/vec_ops.h"
#include "icicle/ntt.h"

#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"
#include "icicle/backend/ntt_config.h"

#include "icicle/program/symbol.h"
#include "icicle/program/program.h"
#include "icicle/program/returning_value_program.h"
#include "../../icicle/backend/cpu/include/cpu_program_executor.h"
#include "test_base.h"

using namespace field_config;
using namespace icicle;

// TODO Hadar - add tests that test different configurations of data on device or on host.

static bool VERBOSE = true;
static int ITERS = 1;

class FieldApiTestBase : public IcicleTestBase
{
};
template <typename T>
class FieldApiTest : public FieldApiTestBase
{
public:
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
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_size = N * batch_size;
  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto in_b = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

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

  // add
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!IcicleTestBase::is_main_device_available()) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] + in_b[i];
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add",
      ITERS);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // accumulate
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  for (int i = 0; i < total_size; i++) { // TODO - compare gpu against cpu with inplace operations?
    out_ref[i] = in_a[i] + in_b[i];
  }
  run(
    IcicleTestBase::main_device(), nullptr, VERBOSE /*=measure*/, vector_accumulate_wrapper, "vector accumulate",
    ITERS);

  ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // sub
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!IcicleTestBase::is_main_device_available()) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] - in_b[i];
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub",
      ITERS);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // mul
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!IcicleTestBase::is_main_device_available()) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] * in_b[i];
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul",
      ITERS);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // div
  TypeParam::rand_host_many(in_a.get(), total_size);
  TypeParam::rand_host_many(in_b.get(), total_size);
  // reference
  if (!IcicleTestBase::is_main_device_available()) {
    for (int i = 0; i < total_size; i++) {
      out_ref[i] = in_a[i] * TypeParam::inverse(in_b[i]);
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, vector_div<TypeParam>, "vector div",
      ITERS);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, vector_div<TypeParam>, "vector div", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

TYPED_TEST(FieldApiTest, montgomeryConversion)
{
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);
  const bool is_to_montgomery = rand_uint_32b(0, 1);
  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "is_to_montgomery = " << is_to_montgomery;
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

  // convert_montgomery
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // reference
  if (!IcicleTestBase::is_main_device_available()) {
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
    run(IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, "montgomery", ITERS);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, "montgomery", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

TEST_F(FieldApiTestBase, VectorReduceOps)
{
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);
  const int total_size = N * batch_size;

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  auto in_a = std::make_unique<scalar_t[]>(total_size);
  auto out_main = std::make_unique<scalar_t[]>(batch_size);
  auto out_ref = std::make_unique<scalar_t[]>(batch_size);

  auto vector_accumulate_wrapper =
    [](scalar_t* a, const scalar_t* b, uint64_t size, const VecOpsConfig& config, scalar_t* /*out*/) {
      return vector_accumulate(a, b, size, config);
    };

  auto run =
    [&](const std::string& dev_type, scalar_t* out, bool measure, auto vec_op_func, const char* msg, int iters) {
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

  // sum
  scalar_t::rand_host_many(in_a.get(), total_size);
  // reference
  for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
    out_ref[idx_in_batch] = scalar_t::from(0);
  }
  if (!IcicleTestBase::is_main_device_available()) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_a = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_in_batch] = out_ref[idx_in_batch] + in_a[idx_a];
      }
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, vector_sum<scalar_t>, "vector sum",
      ITERS);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, vector_sum<scalar_t>, "vector sum", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(scalar_t)));

  // product
  scalar_t::rand_host_many(in_a.get(), total_size);
  if (!IcicleTestBase::is_main_device_available()) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      out_ref[idx_in_batch] = scalar_t::from(1);
    }
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_a = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_in_batch] = out_ref[idx_in_batch] * in_a[idx_a];
      }
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, vector_product<scalar_t>,
      "vector product", ITERS);
  }
  run(
    IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, vector_product<scalar_t>, "vector product",
    ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(scalar_t)));
}

TEST_F(FieldApiTestBase, scalarVectorOps)
{
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_size = N * batch_size;
  auto scalar_a = std::make_unique<scalar_t[]>(batch_size);
  auto in_b = std::make_unique<scalar_t[]>(total_size);
  auto out_main = std::make_unique<scalar_t[]>(total_size);
  auto out_ref = std::make_unique<scalar_t[]>(total_size);
  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  auto vector_accumulate_wrapper =
    [](scalar_t* a, const scalar_t* b, uint64_t size, const VecOpsConfig& config, scalar_t* /*out*/) {
      return vector_accumulate(a, b, size, config);
    };

  auto run =
    [&](const std::string& dev_type, scalar_t* out, bool measure, auto vec_op_func, const char* msg, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);
      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(scalar_a.get(), in_b.get(), N, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // scalar add vec
  scalar_t::rand_host_many(scalar_a.get(), batch_size);
  scalar_t::rand_host_many(in_b.get(), total_size);

  // reference
  if (!IcicleTestBase::is_main_device_available()) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_b = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_b] = (scalar_a[idx_in_batch]) + in_b[idx_b];
      }
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, scalar_add_vec<scalar_t>,
      "scalar add vec", ITERS);
  }
  run(
    IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, scalar_add_vec<scalar_t>, "scalar add vec",
    ITERS);

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));

  // scalar sub vec
  scalar_t::rand_host_many(scalar_a.get(), batch_size);
  scalar_t::rand_host_many(in_b.get(), total_size);

  if (!IcicleTestBase::is_main_device_available()) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_b = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_b] = (scalar_a[idx_in_batch]) - in_b[idx_b];
      }
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, scalar_sub_vec<scalar_t>,
      "scalar sub vec", ITERS);
  }

  run(
    IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, scalar_sub_vec<scalar_t>, "scalar sub vec",
    ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));

  // scalar mul vec
  scalar_t::rand_host_many(scalar_a.get(), batch_size);
  scalar_t::rand_host_many(in_b.get(), total_size);

  if (!IcicleTestBase::is_main_device_available()) {
    for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; idx_in_batch++) {
      for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
        uint64_t idx_b = columns_batch ? idx_in_N * batch_size + idx_in_batch : idx_in_batch * N + idx_in_N;
        out_ref[idx_b] = (scalar_a[idx_in_batch]) * in_b[idx_b];
      }
    }
  } else {
    run(
      IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, scalar_mul_vec<scalar_t>,
      "scalar mul vec", ITERS);
  }
  run(
    IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, scalar_mul_vec<scalar_t>, "scalar mul vec",
    ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}

TYPED_TEST(FieldApiTest, matrixAPIsAsync)
{
  const int R = 1 << rand_uint_32b(
                  2, 9); // cpu implementation for out of place transpose also supports sizes which are not powers of 2
  const int C = 1 << rand_uint_32b(
                  2, 9); // cpu implementation for out of place transpose also supports sizes which are not powers of 2
  const int batch_size = 1 << rand_uint_32b(0, 3);
  const bool columns_batch = rand_uint_32b(0, 1);
  const bool is_in_place = IcicleTestBase::is_main_device_available() ? 0 : rand_uint_32b(0, 1);

  ICICLE_LOG_DEBUG << "rows = " << R;
  ICICLE_LOG_DEBUG << "cols = " << C;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

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

  TypeParam::rand_host_many(h_inout.get(), total_size);

  // Reference implementation
  if (!IcicleTestBase::is_main_device_available()) {
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
    run(
      IcicleTestBase::reference_device(), (is_in_place ? h_inout.get() : h_out_ref.get()), VERBOSE /*=measure*/,
      "transpose", ITERS);
  }

  run(
    IcicleTestBase::main_device(), (is_in_place ? h_inout.get() : h_out_main.get()), VERBOSE /*=measure*/, "transpose",
    ITERS);

  if (is_in_place) {
    ASSERT_EQ(0, memcmp(h_inout.get(), h_out_ref.get(), total_size * sizeof(TypeParam)));
  } else {
    ASSERT_EQ(0, memcmp(h_out_main.get(), h_out_ref.get(), total_size * sizeof(TypeParam)));
  }
}

TYPED_TEST(FieldApiTest, bitReverse)
{
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);
  const bool is_in_place = rand_uint_32b(0, 1);
  const int total_size = N * batch_size;

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "is_in_place = " << is_in_place;

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

  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);

  // Reference implementation
  if (!IcicleTestBase::is_main_device_available() || is_in_place) {
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
        } else {
          out_ref[idx_in_batch * N + i] = in_a[idx_in_batch * N + rev];
        }
      }
    }
  } else {
    run(
      IcicleTestBase::reference_device(), (is_in_place ? in_a.get() : out_ref.get()), VERBOSE /*=measure*/,
      "bit-reverse", 1);
  }
  run(
    IcicleTestBase::main_device(), (is_in_place ? in_a.get() : out_main.get()), VERBOSE /*=measure*/, "bit-reverse", 1);

  if (is_in_place) {
    ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), N * sizeof(TypeParam)));
  } else {
    ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
  }
}

TYPED_TEST(FieldApiTest, Slice)
{
  const uint64_t size_in = 1 << rand_uint_32b(4, 17);
  const uint64_t offset = rand_uint_32b(0, 14);
  const uint64_t stride = rand_uint_32b(1, 4);
  const uint64_t size_out = rand_uint_32b(0, (size_in - offset) / stride);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);

  ICICLE_LOG_DEBUG << "size_in = " << size_in;
  ICICLE_LOG_DEBUG << "size_out = " << size_out;
  ICICLE_LOG_DEBUG << "offset = " << offset;
  ICICLE_LOG_DEBUG << "stride = " << stride;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_size_in = size_in * batch_size;
  const int total_size_out = size_out * batch_size;

  auto in_a = std::make_unique<TypeParam[]>(total_size_in);
  auto out_main = std::make_unique<TypeParam[]>(total_size_out);
  auto out_ref = std::make_unique<TypeParam[]>(total_size_out);

  TypeParam::rand_host_many(in_a.get(), total_size_in);

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

  // Reference implementation
  if (!IcicleTestBase::is_main_device_available()) {
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
    run(IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, "slice", 1);
  }
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, "slice", 1);

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size_out * sizeof(TypeParam)));
}

TEST_F(FieldApiTestBase, highestNonZeroIdx)
{
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);
  const int total_size = N * batch_size;

  auto in_a = std::make_unique<scalar_t[]>(total_size);
  for (int i = 0; i < batch_size; ++i) {
    // randomize different rows with zeros in the end
    auto size = std::max(int64_t(N) / 4 - i, int64_t(1));
    scalar_t::rand_host_many(in_a.get() + i * N, size);
  }
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

  run(IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, "highest_non_zero_idx", 1);
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, "highest_non_zero_idx", 1);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(int64_t)));
}

TEST_F(FieldApiTestBase, polynomialEval)
{
  const uint64_t coeffs_size = 1 << rand_uint_32b(4, 13);
  const uint64_t domain_size = 1 << rand_uint_32b(2, 9);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);

  ICICLE_LOG_DEBUG << "coeffs_size = " << coeffs_size;
  ICICLE_LOG_DEBUG << "domain_size = " << domain_size;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_coeffs_size = coeffs_size * batch_size;
  const int total_result_size = domain_size * batch_size;

  auto in_coeffs = std::make_unique<scalar_t[]>(total_coeffs_size);
  auto in_domain = std::make_unique<scalar_t[]>(domain_size);
  auto out_main = std::make_unique<scalar_t[]>(total_result_size);
  auto out_ref = std::make_unique<scalar_t[]>(total_result_size);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
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

  scalar_t::rand_host_many(in_coeffs.get(), total_coeffs_size);
  scalar_t::rand_host_many(in_domain.get(), domain_size);

  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE /*=measure*/, "polynomial_eval", 1);
  run(IcicleTestBase::reference_device(), out_ref.get(), VERBOSE /*=measure*/, "polynomial_eval", 1);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_result_size * sizeof(scalar_t)));
}

TEST_F(FieldApiTestBase, polynomialDivision)
{
  const uint64_t numerator_size = 1 << rand_uint_32b(5, 7);
  const uint64_t denominator_size = 1 << rand_uint_32b(3, 4);
  const uint64_t q_size = numerator_size - denominator_size + 1;
  const uint64_t r_size = numerator_size;
  const int batch_size = rand_uint_32b(10, 19);

  // basically we compute q(x),r(x) for a(x)=q(x)b(x)+r(x) by dividing a(x)/b(x)

  auto numerator = std::make_unique<scalar_t[]>(numerator_size * batch_size);
  auto denominator = std::make_unique<scalar_t[]>(denominator_size * batch_size);

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    for (int columns_batch = 0; columns_batch <= 1; columns_batch++) {
      ICICLE_LOG_INFO << "testing polynomial division on device " << device << " [column_batch=" << columns_batch
                      << "]";

      // randomize matrix with rows/cols as polynomials
      scalar_t::rand_host_many(numerator.get(), numerator_size * batch_size);
      scalar_t::rand_host_many(denominator.get(), denominator_size * batch_size);

      // Add padding to each vector so that the degree is lower than the size
      const int zero_pad_length = 1;
      if (columns_batch) {
        for (int i = 0; i < batch_size * zero_pad_length; i++) {
          numerator[batch_size * numerator_size - batch_size * zero_pad_length + i] = scalar_t::zero();
          denominator[batch_size * denominator_size - batch_size * zero_pad_length + i] = scalar_t::zero();
        }
      } else {
        for (int i = 0; i < batch_size; ++i) {
          for (int j = 0; j < zero_pad_length; ++j) {
            numerator[i * numerator_size + numerator_size - zero_pad_length + j] = scalar_t::zero();
            denominator[i * denominator_size + denominator_size - zero_pad_length + j] = scalar_t::zero();
          }
        }
      }

      auto q = std::make_unique<scalar_t[]>(q_size * batch_size);
      auto r = std::make_unique<scalar_t[]>(r_size * batch_size);

      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;
      // TODO v3.2 support column batch for this API
      if (columns_batch && device == "CUDA") {
        ICICLE_LOG_INFO << "Skipping polynomial division column batch";
        continue;
      }

      ICICLE_CHECK(polynomial_division(
        numerator.get(), numerator_size, denominator.get(), denominator_size, config, q.get(), q_size, r.get(),
        r_size));

      // test a(x)=q(x)b(x)+r(x) in random point
      const auto rand_x = scalar_t::rand_host();
      auto ax = std::make_unique<scalar_t[]>(config.batch_size);
      auto bx = std::make_unique<scalar_t[]>(config.batch_size);
      auto qx = std::make_unique<scalar_t[]>(config.batch_size);
      auto rx = std::make_unique<scalar_t[]>(config.batch_size);
      polynomial_eval(numerator.get(), numerator_size, &rand_x, 1, config, ax.get());
      polynomial_eval(denominator.get(), denominator_size, &rand_x, 1, config, bx.get());
      polynomial_eval(q.get(), q_size, &rand_x, 1, config, qx.get());
      polynomial_eval(r.get(), r_size, &rand_x, 1, config, rx.get());

      for (int i = 0; i < config.batch_size; ++i) {
        ASSERT_EQ(ax[i], qx[i] * bx[i] + rx[i]);
      }
    }
  }
}

#ifdef NTT

TYPED_TEST(FieldApiTest, ntt)
{
  // Randomize configuration
  const bool inplace = rand_uint_32b(0, 1);
  const int logn = rand_uint_32b(3, 17);
  const uint64_t N = 1 << logn;
  const int log_ntt_domain_size = logn + 1;
  const int log_batch_size = rand_uint_32b(0, 2);
  const int batch_size = 1 << log_batch_size;
  const int _ordering = rand_uint_32b(0, 3);
  const Ordering ordering = static_cast<Ordering>(_ordering);
  bool columns_batch;
  if (logn == 7 || logn < 4) {
    columns_batch = false; // currently not supported (icicle_v3/backend/cuda/src/ntt/ntt.cuh line 578)
  } else {
    columns_batch = rand_uint_32b(0, 1);
  }
  const NTTDir dir = static_cast<NTTDir>(rand_uint_32b(0, 1)); // 0: forward, 1: inverse
  const int log_coset_stride = rand_uint_32b(0, 2);
  scalar_t coset_gen;
  if (log_coset_stride) {
    coset_gen = scalar_t::omega(logn + log_coset_stride);
  } else {
    coset_gen = scalar_t::one();
  }

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "inplace = " << inplace;
  ICICLE_LOG_DEBUG << "ordering = " << _ordering;
  ICICLE_LOG_DEBUG << "log_coset_stride = " << log_coset_stride;

  const int total_size = N * batch_size;
  auto scalars = std::make_unique<TypeParam[]>(total_size);
  TypeParam::rand_host_many(scalars.get(), total_size);

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
      if (inplace) {
        ICICLE_CHECK(ntt(d_in, N, dir, config, d_in));
      } else {
        ICICLE_CHECK(ntt(d_in, N, dir, config, d_out));
      }
    }
    END_TIMER(NTT_sync, oss.str().c_str(), measure);

    if (inplace) {
      ICICLE_CHECK(icicle_copy_to_host_async(out, d_in, total_size * sizeof(TypeParam), config.stream));
    } else {
      ICICLE_CHECK(icicle_copy_to_host_async(out, d_out, total_size * sizeof(TypeParam), config.stream));
    }
    ICICLE_CHECK(icicle_free_async(d_in, config.stream));
    ICICLE_CHECK(icicle_free_async(d_out, config.stream));
    ICICLE_CHECK(icicle_stream_synchronize(config.stream));
    ICICLE_CHECK(icicle_destroy_stream(stream));
    ICICLE_CHECK(ntt_release_domain<scalar_t>());
  };
  run(IcicleTestBase::main_device(), out_main.get(), "ntt", false /*=measure*/, 10 /*=iters*/); // warmup
  run(IcicleTestBase::reference_device(), out_ref.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
  run(IcicleTestBase::main_device(), out_main.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}
#endif // NTT

// define program
using MlePoly = Symbol<scalar_t>;

// define program
using MlePoly = Symbol<scalar_t>;
void lambda_multi_result(std::vector<MlePoly>& vars)
{
  const MlePoly& A = vars[0];
  const MlePoly& B = vars[1];
  const MlePoly& C = vars[2];
  const MlePoly& EQ = vars[3];
  vars[4] = EQ * (A * B - C) + scalar_t::from(9);
  vars[5] = A * B - C.inverse();
  vars[6] = vars[5];
}

TEST_F(FieldApiTestBase, CpuProgramExecutorMultiRes)
{
  scalar_t a = scalar_t::rand_host();
  scalar_t b = scalar_t::rand_host();
  scalar_t c = scalar_t::rand_host();
  scalar_t eq = scalar_t::rand_host();
  scalar_t res_0;
  scalar_t res_1;
  scalar_t res_2;

  Program<scalar_t> program(lambda_multi_result, 7);
  CpuProgramExecutor<scalar_t> prog_exe(program);

  // init program
  prog_exe.m_variable_ptrs[0] = &a;
  prog_exe.m_variable_ptrs[1] = &b;
  prog_exe.m_variable_ptrs[2] = &c;
  prog_exe.m_variable_ptrs[3] = &eq;
  prog_exe.m_variable_ptrs[4] = &res_0;
  prog_exe.m_variable_ptrs[5] = &res_1;
  prog_exe.m_variable_ptrs[6] = &res_2;

  // execute
  prog_exe.execute();

  // check correctness
  scalar_t expected_res_0 = eq * (a * b - c) + scalar_t::from(9);
  ASSERT_EQ(res_0, expected_res_0);

  scalar_t expected_res_1 = a * b - scalar_t::inverse(c);
  ASSERT_EQ(res_1, expected_res_1);
  ASSERT_EQ(res_2, res_1);
}

MlePoly returning_value_func(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& EQ = inputs[3];
  return (EQ * (A * B - C));
}

TEST_F(FieldApiTestBase, CpuProgramExecutorReturningVal)
{
  // randomize input vectors
  const int total_size = 100000;
  auto in_a = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_a.get(), total_size);
  auto in_b = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_b.get(), total_size);
  auto in_c = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_c.get(), total_size);
  auto in_eq = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_eq.get(), total_size);

  //----- element wise operation ----------------------
  auto out_element_wise = std::make_unique<scalar_t[]>(total_size);
  START_TIMER(element_wise_op)
  for (int i = 0; i < 100000; ++i) {
    out_element_wise[i] = in_eq[i] * (in_a[i] * in_b[i] - in_c[i]);
  }
  END_TIMER(element_wise_op, "Straight forward function (Element wise) time: ", true);

  //----- explicit program ----------------------
  ReturningValueProgram<scalar_t> program_explicit(returning_value_func, 4);

  CpuProgramExecutor<scalar_t> prog_exe_explicit(program_explicit);
  auto out_explicit_program = std::make_unique<scalar_t[]>(total_size);

  // init program
  prog_exe_explicit.m_variable_ptrs[0] = in_a.get();
  prog_exe_explicit.m_variable_ptrs[1] = in_b.get();
  prog_exe_explicit.m_variable_ptrs[2] = in_c.get();
  prog_exe_explicit.m_variable_ptrs[3] = in_eq.get();
  prog_exe_explicit.m_variable_ptrs[4] = out_explicit_program.get();

  // run on all vectors
  START_TIMER(explicit_program)
  for (int i = 0; i < total_size; ++i) {
    prog_exe_explicit.execute();
    (prog_exe_explicit.m_variable_ptrs[0])++;
    (prog_exe_explicit.m_variable_ptrs[1])++;
    (prog_exe_explicit.m_variable_ptrs[2])++;
    (prog_exe_explicit.m_variable_ptrs[3])++;
    (prog_exe_explicit.m_variable_ptrs[4])++;
  }
  END_TIMER(explicit_program, "Explicit program executor time: ", true);

  // check correctness
  ASSERT_EQ(0, memcmp(out_element_wise.get(), out_explicit_program.get(), total_size * sizeof(scalar_t)));

  //----- predefined program ----------------------
  Program<scalar_t> predef_program(EQ_X_AB_MINUS_C);

  CpuProgramExecutor<scalar_t> prog_exe_predef(predef_program);
  auto out_predef_program = std::make_unique<scalar_t[]>(total_size);

  // init program
  prog_exe_predef.m_variable_ptrs[0] = in_a.get();
  prog_exe_predef.m_variable_ptrs[1] = in_b.get();
  prog_exe_predef.m_variable_ptrs[2] = in_c.get();
  prog_exe_predef.m_variable_ptrs[3] = in_eq.get();
  prog_exe_predef.m_variable_ptrs[4] = out_predef_program.get();

  // run on all vectors
  START_TIMER(predef_program)
  for (int i = 0; i < total_size; ++i) {
    prog_exe_predef.execute();
    (prog_exe_predef.m_variable_ptrs[0])++;
    (prog_exe_predef.m_variable_ptrs[1])++;
    (prog_exe_predef.m_variable_ptrs[2])++;
    (prog_exe_predef.m_variable_ptrs[3])++;
    (prog_exe_predef.m_variable_ptrs[4])++;
  }
  END_TIMER(predef_program, "Program predefined time: ", true);

  // check correctness
  ASSERT_EQ(0, memcmp(out_element_wise.get(), out_predef_program.get(), total_size * sizeof(scalar_t)));

  //----- Vecops operation ----------------------
  auto config = default_vec_ops_config();
  auto out_vec_ops = std::make_unique<scalar_t[]>(total_size);

  START_TIMER(vecop)
  vector_mul(in_a.get(), in_b.get(), total_size, config, out_vec_ops.get());         // A * B
  vector_sub(out_vec_ops.get(), in_c.get(), total_size, config, out_vec_ops.get());  // A * B - C
  vector_mul(out_vec_ops.get(), in_eq.get(), total_size, config, out_vec_ops.get()); // EQ * (A * B - C)
  END_TIMER(vecop, "Vec ops time: ", true);

  // check correctness
  ASSERT_EQ(0, memcmp(out_element_wise.get(), out_vec_ops.get(), total_size * sizeof(scalar_t)));
}

MlePoly ex_x_ab_minus_c_func(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& EQ = inputs[3];
  return EQ * (A * B - C);
}

TEST_F(FieldApiTestBase, ProgramExecutorVecOp)
{
  // randomize input vectors
  const int total_size = 100000;
  const ReturningValueProgram<scalar_t> prog(ex_x_ab_minus_c_func, 4);
  auto in_a = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_a.get(), total_size);
  auto in_b = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_b.get(), total_size);
  auto in_c = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_c.get(), total_size);
  auto in_eq = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_eq.get(), total_size);

  auto run = [&](const std::string& dev_type, std::vector<scalar_t*>& data, const Program<scalar_t>& program, uint64_t size, const char* msg) {
  Device dev = {dev_type, 0};
  icicle_set_device(dev);
  auto config = default_vec_ops_config();

  std::ostringstream oss;
  oss << dev_type << " " << msg;

  START_TIMER(executeProgram)
  ICICLE_CHECK(execute_program(data, program, size, config)); 
  END_TIMER(executeProgram, oss.str().c_str(), true);
  };
  
  // initialize data vector for main device
  auto out_main = std::make_unique<scalar_t[]>(total_size);
  std::vector<scalar_t*> data_main = std::vector<scalar_t*>(5);
  data_main[0] = in_a.get();
  data_main[1] = in_b.get();
  data_main[2] = in_c.get();
  data_main[3] = in_eq.get();
  data_main[4] = out_main.get();

  // initialize data vector for referance device
  auto out_ref = std::make_unique<scalar_t[]>(total_size);
  std::vector<scalar_t*> data_ref = std::vector<scalar_t*>(5);
  data_ref[0] = in_a.get();
  data_ref[1] = in_b.get();
  data_ref[2] = in_c.get();
  data_ref[3] = in_eq.get();
  data_ref[4] = out_ref.get();

  // run on both devices and compare
  run(IcicleTestBase::main_device(), data_main, prog, total_size, "execute_program");
  run(IcicleTestBase::reference_device(), data_ref, prog, total_size, "execute_program");
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}