#pragma once

#include <cstdint>
#include <gtest/gtest.h>

#include "icicle/vec_ops.h"
#include "icicle/mat_ops.h"

#include "icicle/fields/field_config.h"
#include "icicle/fields/field.h"

#include "icicle/utils/log.h"

#include "test_base.h"

using namespace field_config;
using namespace icicle;

// This class is for tests that define a single type
class MatrixTestBase : public IcicleTestBase
{
};

// This class is for tests that should be instantiated for multiple types
template <typename T>
class MatrixTest : public MatrixTestBase
{
};

#ifdef EXT_FIELD
typedef testing::Types<scalar_t, extension_t> MatrixTestTypes;
#elif defined(RING)
typedef testing::Types<scalar_t, scalar_rns_t, PolyRing> MatrixTestTypes;
#elif defined(FIELD)
typedef testing::Types<scalar_t> MatrixTestTypes;
#else
  #error invalid type for ring and field test
#endif

TYPED_TEST_SUITE(MatrixTest, MatrixTestTypes);

// Helper function to multiply matrices in O(n^3) time using host math
template <typename T>
void matmul_ref(
  const std::vector<T>& a,
  const std::vector<T>& b,
  std::vector<T>& out,
  size_t rows_a,
  size_t cols_a, // also rows_b
  size_t cols_b)
{
  // For each element of output matrix
  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < cols_b; j++) {
      // Initialize accumulator for dot product
      T sum = (a[i * cols_a + 0] * b[0 * cols_b + j]);

      // Compute dot product of row i from A and col j from B
      for (size_t k = 1; k < cols_a; k++) {
        sum = sum + (a[i * cols_a + k] * b[k * cols_b + j]);
      }

      out[i * cols_b + j] = sum;
    }
  }
}

// Matrix multiplication with non-square matrices
TEST_F(MatrixTestBase, MatrixMultiplicationNonSquare)
{
  const size_t N = 1 << 8;
  const size_t M = 1 << 10;

  auto direct_input_a = std::vector<scalar_t>(N);
  auto direct_input_b = std::vector<scalar_t>(N);
  auto direct_output = std::vector<scalar_t>(N);
  auto icicle_output = std::vector<scalar_t>(N);

  // Initialize input with N random vectors of M elements
  direct_input_a.resize(N * M);
  direct_input_b.resize(M * N);
  scalar_t::rand_host_many(direct_input_a.data(), N * M);
  scalar_t::rand_host_many(direct_input_b.data(), M * N);

  // Initialize output buffer with correct size
  direct_output.resize(N * M);
  icicle_output.resize(N * M);
  // Compute reference result using host math
  matmul_ref(direct_input_a, direct_input_b, direct_output, N, M, N);

  // Compute result using icicle device
  auto cfg = MatMulConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    ICICLE_CHECK(matmul(direct_input_a.data(), N, M, direct_input_b.data(), M, N, cfg, icicle_output.data()));

    // Compare results
    ASSERT_EQ(0, std::memcmp(direct_output.data(), icicle_output.data(), direct_output.size() * sizeof(scalar_t)));
  }
}

// Matrix multiplication sanity checks
TEST_F(MatrixTestBase, MatrixMultiplicationSanityChecks)
{
  const size_t matrix_size = 4; // Small size for easy verification

  // Test 1: Identity matrix multiplication
  std::vector<scalar_t> identity(matrix_size * matrix_size, scalar_t::zero());
  for (size_t i = 0; i < matrix_size; i++) {
    identity[i * matrix_size + i] = scalar_t::one();
  }

  std::vector<scalar_t> random_matrix(matrix_size * matrix_size);
  scalar_t::rand_host_many(random_matrix.data(), matrix_size * matrix_size);

  std::vector<scalar_t> result(matrix_size * matrix_size);
  auto cfg = MatMulConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // A * I = A
    ICICLE_CHECK(matmul(
      random_matrix.data(), matrix_size, matrix_size, identity.data(), matrix_size, matrix_size, cfg, result.data()));

    ASSERT_EQ(0, std::memcmp(random_matrix.data(), result.data(), random_matrix.size() * sizeof(scalar_t)));

    // I * A = A
    ICICLE_CHECK(matmul(
      identity.data(), matrix_size, matrix_size, random_matrix.data(), matrix_size, matrix_size, cfg, result.data()));

    ASSERT_EQ(0, std::memcmp(random_matrix.data(), result.data(), random_matrix.size() * sizeof(scalar_t)));

    // Test 2: Zero matrix multiplication
    std::vector<scalar_t> zero_matrix(matrix_size * matrix_size, scalar_t::zero());
    std::vector<scalar_t> expected_zero(matrix_size * matrix_size, scalar_t::zero());

    // A * 0 = 0
    ICICLE_CHECK(matmul(
      random_matrix.data(), matrix_size, matrix_size, zero_matrix.data(), matrix_size, matrix_size, cfg,
      result.data()));

    ASSERT_EQ(0, std::memcmp(expected_zero.data(), result.data(), expected_zero.size() * sizeof(scalar_t)));

    // 0 * A = 0
    ICICLE_CHECK(matmul(
      zero_matrix.data(), matrix_size, matrix_size, random_matrix.data(), matrix_size, matrix_size, cfg,
      result.data()));

    ASSERT_EQ(0, std::memcmp(expected_zero.data(), result.data(), expected_zero.size() * sizeof(scalar_t)));

    // Test 3: Matrix multiplication with 1x1 matrices
    std::vector<scalar_t> single_a(1);
    std::vector<scalar_t> single_b(1);
    std::vector<scalar_t> single_result(1);

    scalar_t::rand_host_many(single_a.data(), 1);
    scalar_t::rand_host_many(single_b.data(), 1);

    ICICLE_CHECK(matmul(single_a.data(), 1, 1, single_b.data(), 1, 1, cfg, single_result.data()));

    ASSERT_EQ(single_a[0] * single_b[0], single_result[0]);

    // Test 4: Matrix-Vector multiplication
    std::vector<scalar_t> vector(matrix_size);
    std::vector<scalar_t> vector_result(matrix_size);
    scalar_t::rand_host_many(vector.data(), matrix_size);

    // Matrix * Vector
    ICICLE_CHECK(
      matmul(random_matrix.data(), matrix_size, matrix_size, vector.data(), matrix_size, 1, cfg, vector_result.data()));

    // Verify result dimensions and values
    std::vector<scalar_t> expected_vector_result(matrix_size);
    matmul_ref(random_matrix, vector, expected_vector_result, matrix_size, matrix_size, 1);

    ASSERT_EQ(
      0, std::memcmp(
           expected_vector_result.data(), vector_result.data(), expected_vector_result.size() * sizeof(scalar_t)));

    // Test 5: Vector-Vector multiplication (outer product)
    std::vector<scalar_t> vector_a(matrix_size);
    std::vector<scalar_t> vector_b(matrix_size);
    std::vector<scalar_t> outer_product_result(matrix_size * matrix_size);
    scalar_t::rand_host_many(vector_a.data(), matrix_size);
    scalar_t::rand_host_many(vector_b.data(), matrix_size);

    // Vector * Vector^T (outer product)
    ICICLE_CHECK(
      matmul(vector_a.data(), matrix_size, 1, vector_b.data(), 1, matrix_size, cfg, outer_product_result.data()));

    // Verify outer product properties
    std::vector<scalar_t> expected_outer_product(matrix_size * matrix_size);
    for (size_t i = 0; i < matrix_size; i++) {
      for (size_t j = 0; j < matrix_size; j++) {
        expected_outer_product[i * matrix_size + j] = vector_a[i] * vector_b[j];
      }
    }

    ASSERT_EQ(
      0,
      std::memcmp(
        expected_outer_product.data(), outer_product_result.data(), expected_outer_product.size() * sizeof(scalar_t)));

    // Test 6: Vector-Matrix multiplication
    std::vector<scalar_t> vector_matrix_result(matrix_size);

    // Vector^T * Matrix
    ICICLE_CHECK(matmul(
      vector.data(), 1, matrix_size, random_matrix.data(), matrix_size, matrix_size, cfg, vector_matrix_result.data()));

    // Verify result dimensions and values
    std::vector<scalar_t> expected_vector_matrix_result(matrix_size);
    matmul_ref(vector, random_matrix, expected_vector_matrix_result, 1, matrix_size, matrix_size);

    ASSERT_EQ(
      0, std::memcmp(
           expected_vector_matrix_result.data(), vector_matrix_result.data(),
           expected_vector_matrix_result.size() * sizeof(scalar_t)));
  }
}

// Negative test cases for matrix multiplication
TEST_F(MatrixTestBase, MatrixMultiplicationDimensionMismatch)
{
  const size_t matrix_size = 4;
  auto cfg = MatMulConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  std::vector<scalar_t> matrix_a(matrix_size * matrix_size);
  std::vector<scalar_t> matrix_b((matrix_size + 1) * matrix_size); // Different number of rows
  std::vector<scalar_t> result(matrix_size * matrix_size);

  scalar_t::rand_host_many(matrix_a.data(), matrix_size * matrix_size);
  scalar_t::rand_host_many(matrix_b.data(), (matrix_size + 1) * matrix_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Should fail because inner dimensions don't match
    auto error = matmul(
      matrix_a.data(), matrix_size, matrix_size, matrix_b.data(), matrix_size + 1, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);
  }
}

TEST_F(MatrixTestBase, MatrixMultiplicationNullInputs)
{
  const size_t matrix_size = 4;
  auto cfg = MatMulConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  std::vector<scalar_t> valid_matrix(matrix_size * matrix_size);
  std::vector<scalar_t> result(matrix_size * matrix_size);
  scalar_t::rand_host_many(valid_matrix.data(), matrix_size * matrix_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Test null A matrix
    auto error = matmul(
      static_cast<const scalar_t*>(nullptr), matrix_size, matrix_size,
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);

    // Test null B matrix
    error = matmul(
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size,
      static_cast<const scalar_t*>(nullptr), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);

    // Test null result matrix
    error = matmul(
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size,
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size, cfg,
      static_cast<scalar_t*>(nullptr));
    ASSERT_NE(error, eIcicleError::SUCCESS);
  }
}

TEST_F(MatrixTestBase, MatrixMultiplicationZeroDimensions)
{
  const size_t matrix_size = 4;
  auto cfg = MatMulConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  std::vector<scalar_t> valid_matrix(matrix_size * matrix_size);
  std::vector<scalar_t> result(matrix_size * matrix_size);
  scalar_t::rand_host_many(valid_matrix.data(), matrix_size * matrix_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Test zero rows
    auto error =
      matmul(valid_matrix.data(), 0, matrix_size, valid_matrix.data(), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);

    // Test zero columns
    error =
      matmul(valid_matrix.data(), matrix_size, 0, valid_matrix.data(), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);
  }
}

#ifdef RING
/* Poly ring multiplier for the unit tests */

PolyRing operator*(const PolyRing& a, const PolyRing& b)
{
  PolyRing c;
  constexpr size_t degree = PolyRing::d;
  const Zq* a_zq = reinterpret_cast<const Zq*>(&a);
  const Zq* b_zq = reinterpret_cast<const Zq*>(&b);
  Zq* c_zq = reinterpret_cast<Zq*>(&c);
  auto config = default_vec_ops_config();
  ICICLE_CHECK(vector_mul(a_zq, b_zq, degree, config, c_zq));
  return c;
}

PolyRing operator+(const PolyRing& a, const PolyRing& b)
{
  PolyRing c;
  constexpr size_t degree = PolyRing::d;
  const Zq* a_zq = reinterpret_cast<const Zq*>(&a);
  const Zq* b_zq = reinterpret_cast<const Zq*>(&b);
  Zq* c_zq = reinterpret_cast<Zq*>(&c);
  auto config = default_vec_ops_config();
  ICICLE_CHECK(vector_add(a_zq, b_zq, degree, config, c_zq));
  return c;
}

// TODO: those can be testing scalar_t too but currently scalar_rns_t doesn't have matmul so

TEST_F(MatrixTestBase, VectorTimesMatrix)
{
  const size_t N = 4;
  const size_t M = 5;

  std::vector<PolyRing> vec(M), mat(M * N), expected(N), actual(N);

  Zq::rand_host_many(reinterpret_cast<Zq*>(vec.data()), PolyRing::d * M);
  Zq::rand_host_many(reinterpret_cast<Zq*>(mat.data()), PolyRing::d * M * N);

  matmul_ref(vec, mat, expected, 1, M, N);

  MatMulConfig cfg{};
  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    ICICLE_CHECK(matmul(vec.data(), 1, M, mat.data(), M, N, cfg, actual.data()));
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), N * sizeof(PolyRing)));
  }
}

TEST_F(MatrixTestBase, MatrixTimesVector)
{
  const size_t N = 4;
  const size_t M = 5;

  std::vector<PolyRing> mat(N * M), vec(M), expected(N), actual(N);

  Zq::rand_host_many(reinterpret_cast<Zq*>(mat.data()), PolyRing::d * N * M);
  Zq::rand_host_many(reinterpret_cast<Zq*>(vec.data()), PolyRing::d * M);

  matmul_ref(mat, vec, expected, N, M, 1);

  MatMulConfig cfg{};
  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    ICICLE_CHECK(matmul(mat.data(), N, M, vec.data(), M, 1, cfg, actual.data()));
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), N * sizeof(PolyRing)));
  }
}

TEST_F(MatrixTestBase, SquareMatrixTimesMatrix)
{
  const size_t N = 4;

  std::vector<PolyRing> a(N * N), b(N * N), expected(N * N), actual(N * N);

  Zq::rand_host_many(reinterpret_cast<Zq*>(a.data()), PolyRing::d * N * N);
  Zq::rand_host_many(reinterpret_cast<Zq*>(b.data()), PolyRing::d * N * N);

  matmul_ref(a, b, expected, N, N, N);

  MatMulConfig cfg{};
  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    ICICLE_CHECK(matmul(a.data(), N, N, b.data(), N, N, cfg, actual.data()));
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), N * N * sizeof(PolyRing)));
  }
}

TEST_F(MatrixTestBase, NonSquareMatrixTimesMatrix)
{
  const size_t N = 4;
  const size_t M = 5;
  const size_t P = 3;

  std::vector<PolyRing> a(N * M), b(M * P), expected(N * P), actual(N * P);

  Zq::rand_host_many(reinterpret_cast<Zq*>(a.data()), PolyRing::d * N * M);
  Zq::rand_host_many(reinterpret_cast<Zq*>(b.data()), PolyRing::d * M * P);

  matmul_ref(a, b, expected, N, M, P);

  MatMulConfig cfg{};
  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    ICICLE_CHECK(matmul(a.data(), N, M, b.data(), M, P, cfg, actual.data()));
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), N * P * sizeof(PolyRing)));
  }
}

TEST_F(MatrixTestBase, VectorTimesVector)
{
  const size_t N = 4;

  std::vector<PolyRing> a(N), b(N), expected(1), actual(1);

  Zq::rand_host_many(reinterpret_cast<Zq*>(a.data()), PolyRing::d * N);
  Zq::rand_host_many(reinterpret_cast<Zq*>(b.data()), PolyRing::d * N);

  matmul_ref(a, b, expected, 1, N, 1);

  MatMulConfig cfg{};
  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    ICICLE_CHECK(matmul(a.data(), 1, N, b.data(), N, 1, cfg, actual.data()));
    ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), sizeof(PolyRing)));
  }
}

#endif

TYPED_TEST(MatrixTest, matrixTranspose)
{
  auto transpose_ref_implementation =
    [](const std::vector<TypeParam>& h_inout, int batch_size, int R, int C, std::vector<TypeParam>& h_out_ref) {
      const std::vector<TypeParam> h_inout_copy = h_inout; // Copy to support in-place transpose
      const TypeParam* cur_mat_in = h_inout_copy.data();
      TypeParam* cur_mat_out = h_out_ref.data();
      const uint64_t total_elements_one_mat = static_cast<uint64_t>(R) * C;

      for (int idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch) {
        for (int i = 0; i < R; ++i) {
          for (int j = 0; j < C; ++j) {
            cur_mat_out[j * R + i] = cur_mat_in[i * C + j];
          }
        }
        cur_mat_in += total_elements_one_mat;
        cur_mat_out += total_elements_one_mat;
      }
    };

  const int nof_rows = 1 << 7;
  const int nof_cols = 1 << 8;
  const int batch_size = 3;

  const int total_size = nof_rows * nof_cols * batch_size;
  std::vector<TypeParam> h_inout(total_size);
  TypeParam::rand_host_many(h_inout.data(), total_size);

  for (auto device : IcicleTestBase::s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    std::stringstream timer_label, timer_label_inplace;
    timer_label << "matrix-transpoes [device=" << device << "]";
    timer_label_inplace << "matrix-transpose-inplace [device=" << device << "]";

    std::vector<TypeParam> h_out(total_size);
    std::vector<TypeParam> h_out_ref(total_size);

    auto config = default_vec_ops_config();
    config.batch_size = batch_size;

    START_TIMER(TRANSPOSE)
    ICICLE_CHECK(matrix_transpose(h_inout.data(), nof_rows, nof_cols, config, h_out.data()));
    END_TIMER(TRANSPOSE, timer_label.str().c_str(), true);

    // Run reference transpose and compare results
    transpose_ref_implementation(h_inout, batch_size, nof_rows, nof_cols, h_out_ref);
    ASSERT_EQ(0, memcmp(h_out.data(), h_out_ref.data(), total_size * sizeof(TypeParam)));

    // Repeat for in-place transpose
    START_TIMER(TRANSPOS_INPLACE)
    ICICLE_CHECK(matrix_transpose(h_inout.data(), nof_rows, nof_cols, config, h_inout.data()));
    END_TIMER(TRANSPOS_INPLACE, timer_label_inplace.str().c_str(), true);

    ASSERT_EQ(0, memcmp(h_inout.data(), h_out_ref.data(), total_size * sizeof(TypeParam)));
  }
}