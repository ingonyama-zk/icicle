#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"
#include <random>

#include "omp.h"

#include "icicle/runtime.h"
#include "icicle/vec_ops.h"
#include "icicle/ntt.h"

#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"
#include "icicle/backend/ntt_config.h"

#include <iostream>
#include <fstream>
#include <memory>

// Function to save the smart pointer to a file
template <typename T>
void saveToFile(const std::string& filename, const std::unique_ptr<T[]>& data, std::size_t size)
{
  std::ofstream outFile(filename, std::ios::binary);
  if (outFile) {
    outFile.write(reinterpret_cast<const char*>(data.get()), size * sizeof(T));
  } else {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
}
// Function to load the smart pointer from a file
template <typename T>
std::unique_ptr<T[]> loadFromFile(const std::string& filename, std::size_t size)
{
  std::unique_ptr<T[]> data = std::make_unique<T[]>(size);
  std::ifstream inFile(filename, std::ios::binary);
  if (inFile) {
    inFile.read(reinterpret_cast<char*>(data.get()), size * sizeof(T));
  } else {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  return data;
}

using namespace field_config;
using namespace icicle;

// TODO Hadar - add tests that test different configurations of data on device or on host.

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
static inline std::vector<std::string> s_registered_devices;
bool s_is_cuda_registered; // TODO Yuval remove this

class FieldApiTestBase : public ::testing::Test
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
    s_registered_devices = get_registered_devices_list();
  }
  static void TearDownTestSuite()
  {
    // make sure to fail in CI if only have one device
    ICICLE_ASSERT(is_device_registered("CUDA")) << "missing CUDA backend";
  }

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
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

/**
 * @brief CPU using OpenMP for vec+vec ops
 * @param vec_a Pointer to the vector/s to be added (a operand)
 * @param vec_b Pointer to the vector/s to be added (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_vector_vector_add(
  const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  #pragma omp parallel for
  for (uint64_t vec_idx = 0; vec_idx < size * config.batch_size; vec_idx++)
  {
    output[vec_idx] = vec_a[vec_idx] + vec_b[vec_idx];
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief CPU using OpenMP for vec-vec ops
 * @param vec_a Pointer to the vector/s to be subbed (a operand)
 * @param vec_b Pointer to the vector/s to be subbed (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_vector_vector_sub(
  const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  #pragma omp parallel for
  for (uint64_t vec_idx = 0; vec_idx < size * config.batch_size; vec_idx++)
  {
    output[vec_idx] = vec_a[vec_idx] - vec_b[vec_idx];
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief CPU using OpenMP for vec*vec ops
 * @param vec_a Pointer to the vector/s to be multiplied (a operand)
 * @param vec_b Pointer to the vector/s to be multiplied (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_vector_vector_mul(
  const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  #pragma omp parallel for schedule(dynamic, 64)
  for (uint64_t vec_idx = 0; vec_idx < size * config.batch_size; vec_idx++)
  {
    output[vec_idx] = vec_a[vec_idx] * vec_b[vec_idx];
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief CPU using OpenMP for vec/vec ops
 * @param vec_a Pointer to the vector/s to be divided (a operand)
 * @param vec_b Pointer to the vector/s to be divided (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_vector_vector_div(
  const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  #pragma omp parallel for schedule(dynamic, 64)
  for (uint64_t vec_idx = 0; vec_idx < size * config.batch_size; vec_idx++)
  {
    output[vec_idx] = vec_a[vec_idx] * T::inverse(vec_b[vec_idx]);
  }
  return eIcicleError::SUCCESS;
}

TYPED_TEST(FieldApiTest, vectorVectorOps)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  // const uint64_t N = 1 << (rand() % 15 + 3);
  const uint64_t N = 1 << 24;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 2;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = true;

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_size = N * batch_size;
  // auto in_a = std::make_unique<TypeParam[]>(total_size);
  // auto in_b = std::make_unique<TypeParam[]>(total_size);
  std::unique_ptr<TypeParam[]> in_a, in_b;
  std::string fname_a = "build/gen_data/in_a_total_size_" + std::to_string(total_size) + ".bin";
  std::string fname_b = "build/gen_data/in_b_total_size_" + std::to_string(total_size) + ".bin";
  try
  {
    in_a = loadFromFile<TypeParam>(fname_a, total_size);
    in_b = loadFromFile<TypeParam>(fname_b, total_size);
    ICICLE_LOG_INFO << "Data loaded";
  }
  catch(const std::exception& e)
  {
    ICICLE_LOG_INFO << "Data not found, generating new data";
    in_a = std::make_unique<TypeParam[]>(total_size);
    in_b = std::make_unique<TypeParam[]>(total_size);
    TypeParam::rand_host_many(in_a.get(), total_size);
    TypeParam::rand_host_many(in_b.get(), total_size);
    saveToFile<TypeParam>(fname_a, in_a, total_size);
    saveToFile<TypeParam>(fname_b, in_b, total_size);
  }

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
      // Warmup
      ICICLE_CHECK(vec_op_func(in_a.get(), in_b.get(), N, config, out));

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(in_a.get(), in_b.get(), N, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // add
  // FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!s_is_cuda_registered) {
    // for (int i = 0; i < total_size; i++) {
    //   out_ref[i] = in_a[i] + in_b[i];
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_vector_vector_add<TypeParam>, "vector add OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_add<TypeParam>, "vector add", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // accumulate
  // FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  // for (int i = 0; i < total_size; i++) {
  //   out_ref[i] = in_a[i] + in_b[i];
  // }
  // run(s_main_target, nullptr, VERBOSE /*=measure*/, vector_accumulate_wrapper, "vector accumulate", ITERS);
  // ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // sub
  // FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!s_is_cuda_registered) {
    // for (int i = 0; i < total_size; i++) {
    //   out_ref[i] = in_a[i] - in_b[i];
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_vector_vector_sub<TypeParam>, "vector sub OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_sub<TypeParam>, "vector sub", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // mul
  // FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // FieldApiTest<TypeParam>::random_samples(in_b.get(), total_size);
  if (!s_is_cuda_registered) {
    // for (int i = 0; i < total_size; i++) {
    //   out_ref[i] = in_a[i] * in_b[i];
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_vector_vector_mul<TypeParam>, "vector mul OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_mul<TypeParam>, "vector mul", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));

  // div
  // TypeParam::rand_host_many(in_a.get(), total_size);
  // TypeParam::rand_host_many(in_b.get(), total_size);
  // reference
  if (!s_is_cuda_registered) {
    // for (int i = 0; i < total_size; i++) {
    //   out_ref[i] = in_a[i] * TypeParam::inverse(in_b[i]);
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_vector_vector_div<TypeParam>, "vector div OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_div<TypeParam>, "vector div", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_div<TypeParam>, "vector div", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

template <typename T>
eIcicleError omp_convert_montgomery(
  const T* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output)
{
  const uint64_t total_size = size * config.batch_size;
  if (is_to_montgomery)
  {
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint64_t i = 0; i < total_size; i++)
    {
      output[i] = T::to_montgomery(input[i]);
    }
  }
  else
  {
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint64_t i = 0; i < total_size; i++)
    {
      output[i] = T::from_montgomery(input[i]);
    }
  }
  return eIcicleError::SUCCESS;
}

TYPED_TEST(FieldApiTest, montgomeryConversion)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  // const uint64_t N = 1 << (rand() % 15 + 3);
  const uint64_t N = 1 << 24;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 2;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = true;
  const bool is_to_montgomery = rand() % 2;
  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "is_to_montgomery = " << is_to_montgomery;
  const int total_size = N * batch_size;
  // auto in_a = std::make_unique<TypeParam[]>(total_size);
  std::unique_ptr<TypeParam[]> in_a;
  std::string fname_a = "build/gen_data/in_a_total_size_" + std::to_string(total_size) + ".bin";
  try
  {
    in_a = loadFromFile<TypeParam>(fname_a, total_size);
    ICICLE_LOG_INFO << "Data loaded";
  }
  catch(const std::exception& e)
  {
    ICICLE_LOG_INFO << "Data not found, generating new data";
    in_a = std::make_unique<TypeParam[]>(total_size);
    TypeParam::rand_host_many(in_a.get(), total_size);
    saveToFile<TypeParam>(fname_a, in_a, total_size);
  }

  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // Warmup
    ICICLE_CHECK(vec_op_func(in_a.get(), N, is_to_montgomery, config, out));

    START_TIMER(MONTGOMERY)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(vec_op_func(in_a.get(), N, is_to_montgomery, config, out));
    }
    END_TIMER(MONTGOMERY, oss.str().c_str(), measure);
  };

  // convert_montgomery
  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  // reference
  if (!s_is_cuda_registered) {
    // if (is_to_montgomery) {
    //   for (int i = 0; i < total_size; i++) {
    //     out_ref[i] = TypeParam::to_montgomery(in_a[i]);
    //   }
    // } else {
    //   for (int i = 0; i < total_size; i++) {
    //     out_ref[i] = TypeParam::from_montgomery(in_a[i]);
    //   }
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/,  omp_convert_montgomery<TypeParam>, "montgomery OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, convert_montgomery<TypeParam>, "montgomery", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, convert_montgomery<TypeParam>, "montgomery", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

/**
 * @brief CPU using OpenMP for sum(vec) ops
 * @param vec_a Pointer to the vector/s to be summed
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size (nof vectors and scalars to be operated) and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_vector_sum(
  const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  #pragma omp declare reduction (+: T : omp_out = omp_out + omp_in) \
          initializer(omp_priv = T::zero())
  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    T sum = T::zero();
    const uint64_t stride =       config.columns_batch? config.batch_size : 1;
    const uint64_t total_size =   config.columns_batch? size * stride : size;
    const uint64_t batch_offset = config.columns_batch? batch_idx : batch_idx * size;
    const T* vec_a_w_batch = vec_a + batch_offset;

    #pragma omp parallel for reduction (+:sum) schedule(dynamic, 64)
    for (uint64_t vec_idx = 0; vec_idx < total_size; vec_idx = vec_idx + stride)
    {
      sum = sum + vec_a_w_batch[vec_idx];
    }
    output[batch_idx] = sum;
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief CPU using OpenMP for multiply(vec) ops
 * @param vec_a Pointer to the vector/s to be summed
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size (nof vectors and scalars to be operated) and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_vector_product(
  const T* vec_a, uint64_t size, const VecOpsConfig& config, T* output)
{
  #pragma omp declare reduction (*: T : omp_out = omp_out * omp_in) \
          initializer(omp_priv = T::one())
  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    T product = T::one();
    const uint64_t stride =       config.columns_batch? config.batch_size : 1;
    const uint64_t total_size =   config.columns_batch? size * stride : size;
    const uint64_t batch_offset = config.columns_batch? batch_idx : batch_idx * size;
    const T* vec_a_w_batch = vec_a + batch_offset;

    #pragma omp parallel for reduction (*:product) schedule(dynamic, 64)
    for (uint64_t vec_idx = 0; vec_idx < total_size; vec_idx +=stride)
    {
      product = product * vec_a_w_batch[vec_idx];
    }
    output[batch_idx] = product;
  }
  return eIcicleError::SUCCESS;
}

TEST_F(FieldApiTestBase, VectorReduceOps)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  // const uint64_t N = 1 << (rand() % 15 + 3);
  const uint64_t N = 1 << 24;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 2;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = true;
  const int total_size = N * batch_size;

  ICICLE_LOG_INFO << "N = " << N;
  ICICLE_LOG_INFO << "batch_size = " << batch_size;
  ICICLE_LOG_INFO << "columns_batch = " << columns_batch;

  // auto in_a = std::make_unique<scalar_t[]>(total_size);
  std::unique_ptr<scalar_t[]> in_a;
  std::string fname = "build/gen_data/in_a_total_size_" + std::to_string(total_size) + ".bin";
  try
  {
    in_a = loadFromFile<scalar_t>(fname, total_size);
    ICICLE_LOG_INFO << "Data loaded";
  }
  catch(const std::exception& e)
  {
    ICICLE_LOG_INFO << "Data not found, generating new data";
    in_a = std::make_unique<scalar_t[]>(total_size);
    scalar_t::rand_host_many(in_a.get(), total_size);
    saveToFile<scalar_t>(fname, in_a, total_size);
  }
  
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
      // Warmup
      ICICLE_CHECK(vec_op_func(in_a.get(), N, config, out));

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(in_a.get(), N, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // sum
  // scalar_t::rand_host_many(in_a.get(), total_size);
  // reference
  for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    out_ref[batch_idx] = scalar_t::from(0);
  }
  if (!s_is_cuda_registered) {
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
    //     uint64_t idx_a = columns_batch ? idx_in_N * batch_size + batch_idx : batch_idx * N + idx_in_N;
    //     out_ref[batch_idx] = out_ref[batch_idx] + in_a[idx_a];
    //   }
    // }
    // Warmup
    // run(s_reference_target, out_ref.get(), false /*=measure*/, omp_vector_sum<scalar_t>, "vector sum OMP", ITERS);

    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_vector_sum<scalar_t>, "vector sum OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_sum<scalar_t>, "vector sum", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_sum<scalar_t>, "vector sum", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(scalar_t)));

  // product
  // scalar_t::rand_host_many(in_a.get(), total_size);
  if (!s_is_cuda_registered) {
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   out_ref[batch_idx] = scalar_t::from(1);
    // }
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
    //     uint64_t idx_a = columns_batch ? idx_in_N * batch_size + batch_idx : batch_idx * N + idx_in_N;
    //     out_ref[batch_idx] = out_ref[batch_idx] * in_a[idx_a];
    //   }
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_vector_product<scalar_t>, "vector product OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, vector_product<scalar_t>, "vector product", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, vector_product<scalar_t>, "vector product", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(scalar_t)));
}

/**
 * @brief CPU using OpenMP for scalar+vec ops
 * @param scalar_a Pointer to the scalar/s to be multiplied (a operand)
 * @param vec_b Pointer to the vector/s to be multiplied (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size (nof vectors and scalars to be operated) and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_scalar_vector_add(
  const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    T scalar = scalar_a[batch_idx];
    if (!config.columns_batch)
    {
      #pragma omp parallel for
      for (uint64_t vec_idx = 0; vec_idx < size; vec_idx++)
      {
        output[vec_idx + batch_idx * size] = scalar + vec_b[vec_idx + batch_idx * size];
      }
    }
    else
    {
      const uint64_t total_size = size * config.batch_size;
      #pragma omp parallel for
      for (uint64_t vec_idx = 0; vec_idx < total_size; vec_idx = vec_idx + config.batch_size)
      {
        output[vec_idx + batch_idx] = scalar + vec_b[vec_idx + batch_idx];
      }
    }
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief CPU using OpenMP for scalar-vec ops
 * @param scalar_a Pointer to the scalar/s to be multiplied (a operand)
 * @param vec_b Pointer to the vector/s to be multiplied (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size (nof vectors and scalars to be operated) and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_scalar_vector_sub(
  const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    T scalar = scalar_a[batch_idx];
    if (!config.columns_batch)
    {
      #pragma omp parallel for
      for (uint64_t vec_idx = 0; vec_idx < size; vec_idx++)
      {
        output[vec_idx + batch_idx * size] = scalar - vec_b[vec_idx + batch_idx * size];
      }
    }
    else
    {
      unsigned total_size = size * config.batch_size;
      #pragma omp parallel for
      for (uint64_t vec_idx = 0; vec_idx < total_size; vec_idx = vec_idx + config.batch_size)
      {
        output[vec_idx + batch_idx] = scalar - vec_b[vec_idx + batch_idx];
      }
    }
  }
  return eIcicleError::SUCCESS;
}

/**
 * @brief CPU using OpenMP for scalar*vec ops
 * @param scalar_a Pouinter to the scalar/s to be multiplied (a operand)
 * @param vec_b Pointer to the vector/s to be multiplied (b operand)
 * @param size Size of a single vector in T limbs
 * @param config Configuration of the VecOP including batch size (nof vectors and scalars to be operated) and columns
 * @param output Result of the operation
 */
template <typename T> eIcicleError omp_scalar_vector_mul(
  const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    T scalar = scalar_a[batch_idx];
    if (!config.columns_batch)
    {
      #pragma omp parallel for schedule(dynamic, 64)
      for (uint64_t vec_idx = 0; vec_idx < size; vec_idx++)
      {
        output[vec_idx + batch_idx * size] = scalar * vec_b[vec_idx + batch_idx * size];
      }
    }
    else
    {
      unsigned total_size = size * config.batch_size;
      #pragma omp parallel for schedule(dynamic, 64)
      for (uint64_t vec_idx = 0; vec_idx < total_size; vec_idx = vec_idx + config.batch_size)
      {
        output[vec_idx + batch_idx] = scalar * vec_b[vec_idx + batch_idx];
      }
    }
  }
  return eIcicleError::SUCCESS;
}

TEST_F(FieldApiTestBase, scalarVectorOps)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  // const uint64_t N = 1 << (rand() % 15 + 3);
  const uint64_t N = 1 << 24;
  ICICLE_LOG_INFO << "N:\t" << N;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 2;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = true;

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;

  const int total_size = N * batch_size;
  // auto scalar_a = std::make_unique<scalar_t[]>(batch_size);
  // auto in_b = std::make_unique<scalar_t[]>(total_size);
  std::unique_ptr<scalar_t[]> scalar_a, in_b;
  std::string fname_a = "build/gen_data/in_a_total_size_" + std::to_string(batch_size) + ".bin";
  std::string fname_b = "build/gen_data/in_b_total_size_" + std::to_string(total_size) + ".bin";
  try
  {
    scalar_a = loadFromFile<scalar_t>(fname_a, batch_size);
    in_b = loadFromFile<scalar_t>(fname_b, total_size);
    ICICLE_LOG_INFO << "Data loaded";
  }
  catch(const std::exception& e)
  {
    ICICLE_LOG_INFO << "Data not found, generating new data";
    scalar_a = std::make_unique<scalar_t[]>(batch_size);
    in_b = std::make_unique<scalar_t[]>(total_size);
    scalar_t::rand_host_many(scalar_a.get(), batch_size);
    scalar_t::rand_host_many(in_b.get(), total_size);
    saveToFile<scalar_t>(fname_a, scalar_a, batch_size);
    saveToFile<scalar_t>(fname_b, in_b, total_size);
  }


  auto out_main = std::make_unique<scalar_t[]>(total_size);
  auto out_ref = std::make_unique<scalar_t[]>(total_size);
  ICICLE_LOG_INFO << "N = " << N;
  ICICLE_LOG_INFO << "batch_size = " << batch_size;
  ICICLE_LOG_INFO << "columns_batch = " << columns_batch;
  ICICLE_LOG_INFO << "columns_batch = " << columns_batch;

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
      // Warmup
      ICICLE_CHECK(vec_op_func(scalar_a.get(), in_b.get(), N, config, out));

      START_TIMER(VECADD_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(vec_op_func(scalar_a.get(), in_b.get(), N, config, out));
      }
      END_TIMER(VECADD_sync, oss.str().c_str(), measure);
    };

  // scalar add vec
  // scalar_t::rand_host_many(scalar_a.get(), batch_size);
  // scalar_t::rand_host_many(in_b.get(), total_size);

  // reference
  if (!s_is_cuda_registered) {
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
    //     uint64_t idx_b = columns_batch ? idx_in_N * batch_size + batch_idx : batch_idx * N + idx_in_N;
    //     out_ref[idx_b] = (scalar_a[batch_idx]) + in_b[idx_b];
    //   }
    // }
    run(s_reference_target, out_ref.get(), VERBOSE, omp_scalar_vector_add<scalar_t>, "scalar add vec OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, scalar_add_vec<scalar_t>, "scalar add vec", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, scalar_add_vec<scalar_t>, "scalar add vec", ITERS);

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));

  // scalar sub vec
  // scalar_t::rand_host_many(scalar_a.get(), batch_size);
  // scalar_t::rand_host_many(in_b.get(), total_size);

  if (!s_is_cuda_registered) {
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
    //     uint64_t idx_b = columns_batch ? idx_in_N * batch_size + batch_idx : batch_idx * N + idx_in_N;
    //     out_ref[idx_b] = (scalar_a[batch_idx]) - in_b[idx_b];
    //   }
    // }
    run(s_reference_target, out_ref.get(), VERBOSE, omp_scalar_vector_sub<scalar_t>, "scalar sub vec OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, scalar_sub_vec<scalar_t>, "scalar sub vec", ITERS);
  }

  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, scalar_sub_vec<scalar_t>, "scalar sub vec", ITERS);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));

  // scalar mul vec
  // scalar_t::rand_host_many(scalar_a.get(), batch_size);
  // scalar_t::rand_host_many(in_b.get(), total_size);

  if (!s_is_cuda_registered) {
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   for (uint64_t idx_in_N = 0; idx_in_N < N; idx_in_N++) {
    //     uint64_t idx_b = columns_batch ? idx_in_N * batch_size + batch_idx : batch_idx * N + idx_in_N;
    //     out_ref[idx_b] = (scalar_a[batch_idx]) * in_b[idx_b];
    //   }
    // }
    run(s_reference_target, out_ref.get(), VERBOSE, omp_scalar_vector_mul<scalar_t>, "scalar mul vec OMP", ITERS);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, scalar_mul_vec<scalar_t>, "scalar mul vec", ITERS);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, scalar_mul_vec<scalar_t>, "scalar mul vec", ITERS);

  ICICLE_LOG_INFO << "Comp 0 idx:\n\t" << out_main[0] << "\n\t" << out_ref[0];
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}

template <typename T> eIcicleError omp_matrix_transpose_out_of_place(
  const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  if (config.columns_batch)
  {
    #pragma omp parallel for collapse(3)
    for (uint64_t i = 0; i < nof_rows * config.batch_size; i += config.batch_size)
    {
      for (uint64_t j = 0; j < nof_cols * config.batch_size; j += config.batch_size)
      {
        for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
        {
          mat_out[batch_idx + i + nof_rows * j] = mat_in[batch_idx + nof_cols * i + j];
        }
      }
    }
  }
  else
  {
    #pragma omp parallel for collapse(3)
    for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
    {
      const uint64_t batch_offset = nof_rows * nof_cols * batch_idx;
      const T* curr_mat_in =        mat_in + batch_offset;
      T* curr_mat_out =             mat_out + batch_offset;
      for (uint64_t i = 0; i < nof_rows; i++)
      {
        for (uint64_t j = 0; j < nof_cols; j++)
        {
          curr_mat_out[i + nof_rows * j] = curr_mat_in[nof_cols * i + j];
        }
      }
    }
  }
  return eIcicleError::SUCCESS;
}

// Function to perform modulus with Mersenne number
uint64_t mersenne_mod(uint64_t shifted_idx, uint32_t total_bits)
{
  uint64_t mod = (1ULL << total_bits) - 1;
  shifted_idx = (shifted_idx & mod) + (shifted_idx >> total_bits);
  while (shifted_idx >= mod) {
    shifted_idx = (shifted_idx & mod) + (shifted_idx >> total_bits);
  }
  return shifted_idx;
}

// Recursive function to generate all k-ary necklaces and to replace the elements within the necklaces
template <typename T>
void gen_necklace(
  uint32_t t,
  uint32_t p,
  uint32_t k,
  uint32_t length,
  std::vector<uint32_t>& necklace,
  std::vector<uint64_t>& task_indices)
{
  if (t > length) {
    if (
      length % p == 0 &&
      !std::all_of(necklace.begin() + 1, necklace.begin() + length + 1, [first_element = necklace[1]](uint32_t x) {
        return x == first_element;
      })) {
      uint32_t start_idx = 0;
      uint64_t multiplier = 1;
      for (int i = length; i >= 1; --i) { // Compute start_idx as the decimal representation of the necklace
        start_idx += necklace[i] * multiplier;
        multiplier *= k;
      }
      task_indices.push_back(start_idx);
    }
    return;
  }

  necklace[t] = necklace[t - p];
  gen_necklace<T>(t + 1, p, k, length, necklace, task_indices);

  for (int i = necklace[t - p] + 1; i < k; ++i) {
    necklace[t] = i;
    gen_necklace<T>(t + 1, t, k, length, necklace, task_indices);
  }
}

uint32_t gcd(uint32_t a, uint32_t b)
{
  while (b != 0) {
    uint32_t temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

template <typename T> eIcicleError omp_matrix_transpose_in_place(
  uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat)
{
  uint32_t log_nof_rows = static_cast<uint32_t>(std::floor(std::log2(nof_rows)));
  uint32_t log_nof_cols = static_cast<uint32_t>(std::floor(std::log2(nof_cols)));
  uint32_t gcd_value = gcd(log_nof_rows, log_nof_cols);
  uint32_t k = 1 << gcd_value; // Base of necklaces
  const uint32_t total_bits = log_nof_rows + log_nof_cols;
  uint32_t length = (total_bits) / gcd_value; // length of necklaces. Since all are powers of 2, equivalent to
                                              // (log_nof_cols + log_nof_rows) / gcd_value;

  std::vector<uint32_t> necklace(length + 1, 0);
  std::vector<uint64_t> start_indices_in_mat; // Collect start indices
  gen_necklace<T>(1, 1, k, length, necklace, start_indices_in_mat);

  const uint64_t stride = config.columns_batch? config.batch_size : 1;

  #pragma omp parallel for collapse(2)
  for (uint64_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    const uint64_t batch_offset = config.columns_batch? idx_in_batch : idx_in_batch * nof_rows * nof_cols;
    for (uint64_t i = 0; i < start_indices_in_mat.size(); i++) {
      const uint64_t start_idx = start_indices_in_mat[i];
      int64_t idx = start_idx;
      uint64_t next_idx;
      T prev_start_cell = mat[stride * idx + batch_offset];
      T prev = mat[stride * idx + batch_offset];
      do {
        uint64_t shifted_idx = idx << log_nof_rows;
        next_idx = mersenne_mod(shifted_idx, total_bits);
        
        T next = mat[stride * next_idx + batch_offset];
        mat[stride * next_idx + batch_offset] = prev;
        prev = next;

        idx = next_idx;
      } while (idx != start_idx);
    }
  }
  return eIcicleError::SUCCESS;
}

template <typename T> eIcicleError omp_square_matrix_transpose_in_place(
  uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat)
{
  if (config.columns_batch)
  {
    const uint64_t stride = config.batch_size;
    #pragma omp parallel for collapse(3)
    for (uint64_t i = stride; i < nof_rows * stride; i += stride)
    {
      for (uint64_t j = 0; j < i; j += stride)
      {
        for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
        {
          std::swap(mat[batch_idx + i * nof_cols + j], mat[batch_idx + i + j * nof_rows]);
        }
      }
    }
  }
  else
  {
    #pragma omp parallel for collapse(3)
    for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
    {
      const uint64_t batch_offset = batch_idx * nof_rows * nof_cols;
      for (uint64_t i = 1; i < nof_rows; i++)
      {
        for (uint64_t j = 0; j < i; j++)
        {
          std::swap(mat[batch_offset + i * nof_cols + j], mat[batch_offset + i + j * nof_rows]);
        }
      }
    }
  }
  return eIcicleError::SUCCESS;
}

template <typename T> eIcicleError omp_matrix_transpose(
  const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out)
{
  ICICLE_ASSERT(mat_in && mat_out && nof_rows != 0 && nof_cols != 0) << "Invalid argument";
  // check if the number of rows and columns are powers of 2, if not use the basic transpose
  bool is_power_of_2 = (nof_rows & (nof_rows - 1)) == 0 && (nof_cols & (nof_cols - 1)) == 0;
  bool is_inplace = mat_in == mat_out;
  if (!is_inplace) {
    return (omp_matrix_transpose_out_of_place<T>(mat_in, nof_rows, nof_cols, config, mat_out));
  } else if (nof_cols == nof_rows) {
    return (omp_square_matrix_transpose_in_place<T>(nof_rows, nof_cols, config, mat_out));
  } else if (is_power_of_2) {
    return (omp_matrix_transpose_in_place<T>(nof_rows, nof_cols, config, mat_out));
  } else {
    ICICLE_LOG_ERROR << "Matrix transpose is not supported for inplace non power of 2 rows and columns";
    return eIcicleError::INVALID_ARGUMENT;
  }
}

TYPED_TEST(FieldApiTest, matrixAPIsAsync)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const int R =
    1
    << (rand() % 10 + 2); // cpu implementation for out of place transpose also supports sizes which are not powers of 2
  const int C =
    1
    << (rand() % 10 + 2); // cpu implementation for out of place transpose also supports sizes which are not powers of 2
  // const int C = R;
  const int batch_size = 1 << (rand() % 4);
  // const int batch_size = 2;
  const bool columns_batch = rand() % 2;
  // const bool columns_batch = 1;
  const bool is_in_place =
    s_is_cuda_registered ? 0 : rand() % 2;
  // const bool is_in_place = 1;

  ICICLE_LOG_DEBUG << "rows = " << R;
  ICICLE_LOG_DEBUG << "cols = " << C;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "is_in_place = " << is_in_place;

  const uint64_t total_size = R * C * batch_size;
  auto h_inout = std::make_unique<TypeParam[]>(total_size);
  auto h_inout_copy = std::make_unique<TypeParam[]>(total_size);
  auto h_out_main = std::make_unique<TypeParam[]>(total_size);
  auto h_out_ref = std::make_unique<TypeParam[]>(total_size);  

  auto run =
    [&](const std::string& dev_type, TypeParam* h_in, TypeParam* h_out, bool measure, auto vec_op_func, const char* msg, int iters) {
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
      icicle_copy_to_device_async(d_in, h_in, total_size * sizeof(TypeParam), config.stream);

      config.is_a_on_device = true;
      config.is_result_on_device = true;
      config.is_async = false;
    }

    TypeParam* in = device_props.using_host_memory ? h_in : d_in;
    TypeParam* out = device_props.using_host_memory ? h_out : d_out;
    // Warmup
    if (is_in_place) {
      // Need another matrix to not affect input before actual run
      auto temp_out = std::make_unique<TypeParam[]>(total_size);
      ICICLE_CHECK(vec_op_func(in, R, C, config, temp_out.get()));
    } else {
      ICICLE_CHECK(vec_op_func(in, R, C, config, out));
    }

    START_TIMER(TRANSPOSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(vec_op_func(in, R, C, config, out));
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
  std::memcpy(h_inout_copy.get(), h_inout.get(), total_size * sizeof(TypeParam));

  // Reference implementation
  if (!s_is_cuda_registered) {
    // const TypeParam* cur_mat_in = h_inout.get();
    // TypeParam* cur_mat_out = h_out_ref.get();
    // uint32_t stride = columns_batch ? batch_size : 1;
    // const uint64_t total_elements_one_mat = static_cast<uint64_t>(R) * C;
    // for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   // Perform the matrix transpose
    //   for (uint32_t i = 0; i < R; ++i) {
    //     for (uint32_t j = 0; j < C; ++j) {
    //       cur_mat_out[stride * (j * R + i)] = cur_mat_in[stride * (i * C + j)];
    //     }
    //   }
    //   cur_mat_in += (columns_batch ? 1 : total_elements_one_mat);
    //   cur_mat_out += (columns_batch ? 1 : total_elements_one_mat);
    // }
    run(s_reference_target, h_inout_copy.get(), (is_in_place ? h_inout_copy.get() : h_out_ref.get()), VERBOSE /*=measure*/, omp_matrix_transpose<TypeParam>, "transpose OMP", ITERS);
    
  } else {
    run(s_reference_target, h_inout_copy.get(), (is_in_place ? h_inout_copy.get() : h_out_ref.get()), VERBOSE /*=measure*/, matrix_transpose<TypeParam>, "transpose", ITERS);
  }

  run(s_main_target, h_inout.get(), (is_in_place ? h_inout.get() : h_out_main.get()), VERBOSE /*=measure*/, matrix_transpose<TypeParam>, "transpose", ITERS);

  if (is_in_place) {
    ASSERT_EQ(0, memcmp(h_inout.get(), h_inout_copy.get(), total_size * sizeof(TypeParam)));
  } else {
    ASSERT_EQ(0, memcmp(h_out_main.get(), h_out_ref.get(), total_size * sizeof(TypeParam)));
  }
}

/**
 * @brief Reverses element order in the input vector/s in a bitwise manner. I.e. out[i] = in[bitrev[i]]
 * @param vec_in Pointer to the vector to be bit-reversed.
 * @param size Size of the vector in T limbs. Must be power of 2 (uint64_t limiting max size).
 * @param config Configuration of the VecOp including batch size and columns.
 * @param vec_out Pointer to result vector, memory allocated by user.
 * @note If input and output point to the same memory location, the operation is performed in place.
 */
template <typename T> eIcicleError omp_bit_reverse(
  const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out)
{
  ICICLE_ASSERT(vec_in && vec_out && size != 0) << "Invalid argument";

  uint32_t logn = static_cast<uint32_t>(std::floor(std::log2(size)));
  ICICLE_ASSERT((1ULL << logn) == size) << "Invalid argument - size is not a power of 2";

  const bool is_in_place = vec_in == vec_out;
  const uint64_t stride = config.columns_batch? config.batch_size : 1;

  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    const unsigned batch_offset = config.columns_batch? batch_idx : batch_idx * size;
    const T* vec_in_w_batch_idx = vec_in + batch_offset;
    T* vec_out_w_batch_idx = vec_out + batch_offset;

    #pragma omp parallel for schedule(dynamic, 64)
    for (uint64_t vec_idx = 0; vec_idx < size; vec_idx++)
    {
      uint64_t rev_idx = vec_idx;
      // Bit reverse the iundex for 64 bits
      rev_idx = ((rev_idx >> 1) & 0x5555555555555555) | ((rev_idx & 0x5555555555555555) << 1);  // bit rev single bits
      rev_idx = ((rev_idx >> 2) & 0x3333333333333333) | ((rev_idx & 0x3333333333333333) << 2);  // bit rev 2 bits chunk
      rev_idx = ((rev_idx >> 4) & 0x0F0F0F0F0F0F0F0F) | ((rev_idx & 0x0F0F0F0F0F0F0F0F) << 4);  // bit rev 4 bits chunk
      rev_idx = ((rev_idx >> 8) & 0x00FF00FF00FF00FF) | ((rev_idx & 0x00FF00FF00FF00FF) << 8);  // bit rev 8 bits chunk
      rev_idx =
        ((rev_idx >> 16) & 0x0000FFFF0000FFFF) | ((rev_idx & 0x0000FFFF0000FFFF) << 16);        // bit rev 16 bits chunk
      rev_idx = (rev_idx >> 32) | (rev_idx << 32);                                              // bit rev 32 bits chunk
      rev_idx = rev_idx >> (64 - logn);                                                     // Align rev_idx to the LSB

      if (is_in_place)
      {
        // Check rev_idx vs vec_idx to ensure it is done once per pair
        if (rev_idx > vec_idx) {
          std::swap(vec_out_w_batch_idx[vec_idx * stride], vec_out_w_batch_idx[rev_idx * stride]);
        }
      }
      else
      {
        vec_out_w_batch_idx[vec_idx * stride] = vec_in_w_batch_idx[rev_idx * stride];
      }
    }
  }
  return eIcicleError::SUCCESS;
}

TYPED_TEST(FieldApiTest, bitReverse)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  // const uint64_t N = 1 << (rand() % 15 + 3);
  const uint64_t N = 1 << 24;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 3;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = 0;
  // const bool is_in_place = rand() % 2;
  const bool is_in_place = 0;
  const int total_size = N * batch_size;

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "is_in_place = " << is_in_place;

  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto in_a_ref = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run =
    [&](const std::string& dev_type, const TypeParam* in, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // Warmup
    // ICICLE_CHECK(vec_op_func(in_a.get(), N, config, out));

    START_TIMER(BIT_REVERSE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(vec_op_func(in, N, config, out));
    }
    END_TIMER(BIT_REVERSE, oss.str().c_str(), measure);
  };

  FieldApiTest<TypeParam>::random_samples(in_a.get(), total_size);
  std::memcpy(in_a_ref.get(), in_a.get(), total_size * sizeof(TypeParam));

  // Reference implementation
  if (!s_is_cuda_registered) {
    // uint64_t logn = 0;
    // uint64_t temp = N;
    // while (temp > 1) {
    //   temp >>= 1;
    //   logn++;
    // }
    // // BIT REVERSE FUNCTION
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      // for (uint64_t i = 0; i < N; i++) {
      //   int rev = 0;
      //   for (int j = 0; j < logn; ++j) {
      //     if (i & (1 << j)) { rev |= 1 << (logn - 1 - j); }
      //   }
      //   if (columns_batch) {
      //     out_ref[batch_idx + batch_size * i] = in_a[batch_idx + batch_size * rev];
      //   } else {
      //     out_ref[batch_idx * N + i] = in_a[batch_idx * N + rev];
      //   }
      // }
    // }
    run(s_reference_target, in_a_ref.get(), (is_in_place ? in_a_ref.get() : out_ref.get()), VERBOSE /*=measure*/, omp_bit_reverse<TypeParam>, "bit-reverse OMP", 1);
  } else {
    run(s_reference_target, in_a_ref.get(), (is_in_place ? in_a_ref.get() : out_ref.get()), VERBOSE /*=measure*/, omp_bit_reverse<TypeParam>, "bit-reverse", 1);
  }
  run(s_main_target, in_a.get(), (is_in_place ? in_a.get() : out_main.get()), VERBOSE /*=measure*/, bit_reverse<TypeParam>, "bit-reverse", 1);

  if (is_in_place) {
    ASSERT_EQ(0, memcmp(in_a.get(), in_a_ref.get(), N * sizeof(TypeParam)));
  } else {
    ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
  }
}

/**
 * @brief Extracts a slice from a vector or batch of vectors.
 *
 * @tparam T Type of the elements in the vector.
 * @param vec_in Pointer to the input vector(s).
 * @param offset Offset from which to start the slice in each vector.
 * @param stride Stride between elements in the slice.
 * @param size_in Number of elements in one input vector.
 * @param size_out Number of elements in one input vector.
 * @param config Configuration for the operation.
 * @param vec_out Pointer to the output vector(s) where the results will be stored.
 *                The output array should have the same storage layout as the input vectors.
 * @return eIcicleError Error code indicating success or failure.
 * @note The total input size is `size_in * config.batch_size`.
 *       The total output size is `size_out * config.batch_size`.
 *       parameters must satisfy: offset + (size_out-1) * stride < size_in
 */
template <typename T> eIcicleError omp_slice(
  const T* vec_in, uint64_t offset, uint64_t stride, uint64_t size_in, uint64_t size_out, const VecOpsConfig& config,
  T* vec_out)
{
  ICICLE_ASSERT(vec_in != nullptr && vec_out != nullptr) << "Error: Invalid argument - input or output vector is null";
  ICICLE_ASSERT(offset + (size_out - 1) * stride < size_in) << "Error: Invalid argument - slice out of bound";

  if (config.columns_batch)
  {
    #pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (uint64_t vec_idx = 0; vec_idx < size_out; vec_idx++)
    {
      for (int batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
      {
        vec_out[batch_idx + vec_idx * config.batch_size] =
          vec_in[batch_idx + (offset + vec_idx * stride) * config.batch_size];
      }
    }
  }
  else
  {
    #pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (int batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
    {
      for (uint64_t vec_idx = 0; vec_idx < size_out; vec_idx++)
      {
        vec_out[batch_idx * size_out + vec_idx] = vec_in[batch_idx * size_in + offset + vec_idx * stride];
      }
    } 
  }
  return eIcicleError::SUCCESS;
}

TYPED_TEST(FieldApiTest, Slice)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  // const uint64_t size_in = 1 << (rand() % 15 + 5);
  const uint64_t size_in = 1 << 15;
  const uint64_t offset = rand() % 15;
  const uint64_t stride = rand() % 4 + 1;
  const uint64_t size_out = rand() % (((size_in - offset) / stride) - 1) + 1;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 32;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = 1;

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

  auto run = 
    [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // Warmup
    ICICLE_CHECK(vec_op_func(in_a.get(), offset, stride, size_in, size_out, config, out));

    START_TIMER(SLICE)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(vec_op_func(in_a.get(), offset, stride, size_in, size_out, config, out));
    }
    END_TIMER(SLICE, oss.str().c_str(), measure);
  };

  // Reference implementation
  if (!s_is_cuda_registered) {
    // for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //   for (uint64_t i = 0; i < size_out; i++) {
    //     if (columns_batch) {
    //       out_ref[batch_idx + batch_size * i] = in_a[batch_idx + batch_size * (offset + i * stride)];
    //     } else {
    //       out_ref[batch_idx * size_out + i] = in_a[batch_idx * size_in + (offset + i * stride)];
    //     }
    //   }
    // }
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_slice<TypeParam>, "slice OMP", 1);
  } else {
    run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, slice<TypeParam>, "slice", 1);
  }
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, slice<TypeParam>, "slice", 1);

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size_out * sizeof(TypeParam)));
}

// NOTE Check performance when in release (After merge with new compiler)
/**
 * @brief Finds the highest non-zero index in a vector or batch of vectors.
 *
 * @tparam T Type of the elements in the vector.
 * @param vec_in Pointer to the input vector(s).
 * @param size Number of elements in each input vector.
 * @param config Configuration for the operation.
 * @param out_idx Pointer to an array where the output indices of the highest non-zero element in each input vector
 * will be stored. The array should have a length of `config.batch_size`.
 * @return eIcicleError Error code indicating success or failure.
 */
template <typename T> eIcicleError omp_highest_non_zero_idx(
  const T* vec_in, uint64_t size, const VecOpsConfig& config, int64_t* out_idx)
{
  ICICLE_ASSERT(vec_in && out_idx && size != 0) << "Error: Invalid argument";

  for (int batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    int64_t max_idx = -1;
    const uint64_t stride =   config.columns_batch? config.batch_size : 1;
    const T* vec_in_w_batch = config.columns_batch? vec_in + batch_idx : vec_in + batch_idx * size;

    #pragma omp parallel for reduction(max: max_idx)
    for (int64_t vec_idx = 0; vec_idx < size; vec_idx++)
    {
      if (vec_in_w_batch[vec_idx * stride] != T::zero()) { max_idx = std::max(max_idx, vec_idx); }
    }
    out_idx[batch_idx] = max_idx;
  }
  return eIcicleError::SUCCESS;
}

TEST_F(FieldApiTestBase, highestNonZeroIdx)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t N = 1 << 20;
  // const int batch_size = 1 << (rand() % 5);
  const int batch_size = 3;
  // const bool columns_batch = rand() % 2;
  const bool columns_batch = 1;
  const int total_size = N * batch_size;

  auto in_a = std::make_unique<scalar_t[]>(total_size);
  for (int i = 0; i < batch_size; ++i) {
    // randomize different rows with zeros in the end
    auto size = std::max(int64_t(N) / 4 - i, int64_t(1));
    scalar_t::rand_host_many(in_a.get() + i * N, size);
  }
  auto out_main = std::make_unique<int64_t[]>(batch_size);
  auto out_ref = std::make_unique<int64_t[]>(batch_size);

  auto run =
    [&](const std::string& dev_type, int64_t* out, bool measure, auto vec_op_func, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // Warmup
    ICICLE_CHECK(vec_op_func(in_a.get(), N, config, out));

    START_TIMER(highestNonZeroIdx)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(vec_op_func(in_a.get(), N, config, out));
    }
    END_TIMER(highestNonZeroIdx, oss.str().c_str(), measure);
  };

  run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_highest_non_zero_idx<scalar_t>, "highest_non_zero_idx OMP", 1);
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, highest_non_zero_idx<scalar_t>, "highest_non_zero_idx", 1);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), batch_size * sizeof(int64_t)));
}

/**
 * @brief Evaluates a polynomial at given domain points.
 *
 * @tparam T Type of the elements in the polynomial and domain.
 * @param coeffs Pointer to the array of coefficients of the polynomial(s).
 *               - The size of `coeffs` should be `coeffs_size * batch_size`.
 *               - If `config.columns_batch` is `false`, coefficients for each polynomial in the batch are stored
 *                 contiguously.
 *               - If `config.columns_batch` is `true`, coefficients are interleaved.
 * @param coeffs_size Number of coefficients in each polynomial.
 * @param domain Pointer to the array of points at which to evaluate the polynomial(s).
 *               - The same domain is used for all polynomials.
 *               - The size of `domain` should be `domain_size`.
 * @param domain_size Number of domain points.
 * @param config Configuration for the operation.
 * @param evals Pointer to the array where the evaluated results will be stored. This is an output parameter.
 *              - The size of `evals` should be `domain_size * batch_size`.
 *              - If `config.columns_batch` is `false`, results for each polynomial are stored contiguously.
 *              - If `config.columns_batch` is `true`, results are interleaved.
 * @return eIcicleError Error code indicating success or failure.
 */
template <typename T> eIcicleError omp_polynomial_eval(
  const T* coeffs,
  uint64_t coeffs_size,
  const T* domain,
  uint64_t domain_size,
  const VecOpsConfig& config,
  T* evals /*OUT*/)
{
  ICICLE_ASSERT(coeffs && domain && evals && coeffs_size != 0 && domain_size != 0) << "Error: Invalid argument";

  const uint64_t stride = config.columns_batch? config.batch_size : 1;
  
  #pragma omp parallel for collapse(2) schedule(dynamic, 64)
  for (uint64_t batch_idx = 0; batch_idx < config.batch_size; batch_idx++)
  {
    const T* curr_coeffs =  config.columns_batch? coeffs + batch_idx  : coeffs + batch_idx * coeffs_size;
    T* curr_evals =         config.columns_batch? evals + batch_idx   : evals + batch_idx * domain_size;

    for (uint64_t x_idx = 0; x_idx < domain_size; x_idx++)  
    {
      const T& x = domain[x_idx];
      curr_evals[x_idx * stride] = curr_coeffs[(coeffs_size - 1) * stride];
      
      for (int64_t coeff_idx = (coeffs_size - 2) * stride; coeff_idx >= 0; coeff_idx -= stride)
      {
        curr_evals[x_idx * stride] = curr_evals[x_idx * stride] * x + curr_coeffs[coeff_idx];
      }
    }
  } 
  
  return eIcicleError::SUCCESS;
}

TEST_F(FieldApiTestBase, polynomialEval)
{
  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const uint64_t coeffs_size = 1 << (rand() % 10 + 4);
  // const uint64_t coeffs_size = 1 << 4;
  const uint64_t domain_size = 1 << (rand() % 8 + 2);
  // const uint64_t domain_size = 1 << 4;
  const int batch_size = 1 << (rand() % 5);
  // const int batch_size = 1;
  const bool columns_batch = rand() % 2;
  // const bool columns_batch = 0;

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

  auto run =
  [&](const std::string& dev_type, scalar_t* out, bool measure, auto vec_op_func, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(polynomialEval)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(vec_op_func(in_coeffs.get(), coeffs_size, in_domain.get(), domain_size, config, out));
    }
    END_TIMER(polynomialEval, oss.str().c_str(), measure);
  };

  scalar_t::rand_host_many(in_coeffs.get(), total_coeffs_size);
  scalar_t::rand_host_many(in_domain.get(), domain_size);

  run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, omp_polynomial_eval<scalar_t>, "polynomial_eval OMP", 1);
  run(s_main_target, out_main.get(), VERBOSE /*=measure*/, polynomial_eval<scalar_t>, "polynomial_eval", 1);
  // run(s_reference_target, out_ref.get(), VERBOSE /*=measure*/, polynomial_eval<scalar_t>, "polynomial_eval", 1);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_result_size * sizeof(scalar_t)));
}

TEST_F(FieldApiTestBase, polynomialDivision)
{
  int seed = time(0);
  srand(seed);
  const uint64_t numerator_size = 1 << (rand() % 3 + 5);
  const uint64_t denominator_size = 1 << (rand() % 2 + 3);
  const uint64_t q_size = numerator_size - denominator_size + 1;
  const uint64_t r_size = numerator_size;
  const int batch_size = 10 + rand() % 10;

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

  int seed = time(0);
  srand(seed);
  ICICLE_LOG_DEBUG << "seed = " << seed;
  const bool inplace = rand() % 2;
  const int logn = rand() % 15 + 3;
  const uint64_t N = 1 << logn;
  const int log_ntt_domain_size = logn + 1;
  const int log_batch_size = rand() % 3;
  const int batch_size = 1 << log_batch_size;
  const int _ordering = rand() % 4;
  const Ordering ordering = static_cast<Ordering>(_ordering);
  bool columns_batch;
  if (logn == 7 || logn < 4) {
    columns_batch = false; // currently not supported (icicle_v3/backend/cuda/src/ntt/ntt.cuh line 578)
  } else {
    columns_batch = rand() % 2;
  }
  const NTTDir dir = static_cast<NTTDir>(rand() % 2); // 0: forward, 1: inverse
  const int log_coset_stride = rand() % 3;
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
  run(s_main_target, out_main.get(), "ntt", false /*=measure*/, 10 /*=iters*/); // warmup
  run(s_reference_target, out_ref.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
  run(s_main_target, out_main.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}
#endif // NTT

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}