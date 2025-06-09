#pragma once

#include <cstdint>
#include <gtest/gtest.h>

#include "icicle/vec_ops.h"

#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"

#include "test_base.h"

class MatrixTestBase : public IcicleTestBase
{
protected:
  // Set up logging for matrix tests
  void SetUp() override
  {
    IcicleTestBase::SetUp();
    Log::set_min_log_level(Log::Verbose); // Enable verbose logging for all matrix tests
  }

  // Helper function to multiply matrices in O(n^3) time using host math
  void multiply_matrices(
    const std::vector<scalar_t>& a,
    const std::vector<scalar_t>& b,
    std::vector<scalar_t>& out,
    size_t rows_a,
    size_t cols_a, // also rows_b
    size_t cols_b)
  {
    // For each element of output matrix
    for (size_t i = 0; i < rows_a; i++) {
      for (size_t j = 0; j < cols_b; j++) {
        // Initialize accumulator for dot product
        scalar_t sum = scalar_t::zero();

        // Compute dot product of row i from A and col j from B
        for (size_t k = 0; k < cols_a; k++) {
          sum = sum + (a[i * cols_a + k] * b[k * cols_b + j]);
        }

        out[i * cols_b + j] = sum;
      }
    }
  }

  std::vector<scalar_t> element_wise_multiply(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b)
  {
    std::vector<scalar_t> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  std::vector<scalar_t> element_wise_add(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b)
  {
    std::vector<scalar_t> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
      result[i] = a[i] + b[i];
    }
    return result;
  }

  void rq_multiply_matrices(
    size_t d,
    const std::vector<scalar_t>& a,
    const std::vector<scalar_t>& b,
    std::vector<scalar_t>& out,
    size_t rows_a,
    size_t cols_a, // also rows_b
    size_t cols_b)
  {
    // For each element of output matrix
    for (size_t i = 0; i < rows_a; i += d) {
      for (size_t j = 0; j < cols_b; j += d) {
        // Initialize accumulator for dot product
        std::vector<scalar_t> sum_vector;

        // Compute polynomial product of row i from A and col j from B
        for (size_t k = 0; k < cols_a; k++) {
          std::vector<scalar_t> rq_vector_1(a.begin() + (i * cols_a + k), a.begin() + (i * cols_a + k) + d);
          std::vector<scalar_t> rq_vector_2(b.begin() + (k * cols_b + j), b.begin() + (k * cols_b + j) + d);

          const std::vector<scalar_t>& zk_product = element_wise_multiply(rq_vector_1, rq_vector_2);
          sum_vector = element_wise_add(sum_vector, zk_product);
        }

        std::copy(sum_vector.begin(), sum_vector.end(), out.begin() + i * cols_b + j);
      }
    }
  }
};

// Batched matrix multiplication
TEST_F(MatrixTestBase, MatrixMultiplicationBatched)
{
  // Random batch size between 4 and 8
  const size_t batch_size = 4 + (rand() % 5); // 4 to 8 inclusive
  const size_t matrix_size = 1 << 8;          // 256x256 matrices

  // Create input and output vectors for the batch
  std::vector<std::vector<scalar_t>> batch_a(batch_size, std::vector<scalar_t>(matrix_size * matrix_size));
  std::vector<scalar_t> single_b(matrix_size * matrix_size);
  std::vector<std::vector<scalar_t>> direct_output(batch_size, std::vector<scalar_t>(matrix_size * matrix_size));
  std::vector<std::vector<scalar_t>> icicle_output(batch_size, std::vector<scalar_t>(matrix_size * matrix_size));

  // Initialize each matrix A in the batch with random values
  for (size_t i = 0; i < batch_size; i++) {
    scalar_t::rand_host_many(batch_a[i].data(), matrix_size * matrix_size);
  }
  // Initialize single B matrix with random values
  scalar_t::rand_host_many(single_b.data(), matrix_size * matrix_size);

  // Compute reference results using host math
  for (size_t i = 0; i < batch_size; i++) {
    multiply_matrices(batch_a[i], single_b, direct_output[i], matrix_size, matrix_size, matrix_size);
  }

  // Compute results using icicle CPU backend
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;
  cfg.batch_size = batch_size;

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Process each matrix A in the batch with the same B matrix
    for (size_t i = 0; i < batch_size; i++) {
      ICICLE_CHECK(matrix_mult(
        batch_a[i].data(), matrix_size, matrix_size, single_b.data(), matrix_size, matrix_size, cfg,
        icicle_output[i].data()));
    }

    // Compare results for each matrix in the batch
    for (size_t i = 0; i < batch_size; i++) {
      ASSERT_EQ(
        0, std::memcmp(direct_output[i].data(), icicle_output[i].data(), direct_output[i].size() * sizeof(scalar_t)));
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
  multiply_matrices(direct_input_a, direct_input_b, direct_output, N, M, N);

  // Compute result using icicle device
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    ICICLE_CHECK(matrix_mult(direct_input_a.data(), N, M, direct_input_b.data(), M, N, cfg, icicle_output.data()));

    // Compare results
    ASSERT_EQ(0, std::memcmp(direct_output.data(), icicle_output.data(), direct_output.size() * sizeof(scalar_t)));
  }
}


// Matrix multiplication with non-square matrices
TEST_F(MatrixTestBase, TqMatrixMultiplicationNonSquare)
{
  const size_t N = 1 << 8;
  const size_t M = 1 << 10;
  const size_t d = 1;

  auto direct_input_a = std::vector<scalar_t>(N);
  auto direct_input_b = std::vector<scalar_t>(N);
  auto direct_output = std::vector<scalar_t>(N);
  auto icicle_output = std::vector<scalar_t>(N);

  // Initialize input with N random vectors of (d * M ) elements
  direct_input_a.resize(N * M * d);
  direct_input_b.resize(M * N * d);
  scalar_t::rand_host_many(direct_input_a.data(), N * M * d);
  scalar_t::rand_host_many(direct_input_b.data(), M * N * d);

  // Initialize output buffer with correct size
  direct_output.resize(N * M * d);
  icicle_output.resize(N * M * d);
  // Compute reference result using host math
  rq_multiply_matrices(d, direct_input_a, direct_input_b, direct_output, N, M, N);

  // Compute result using icicle device
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    //ICICLE_CHECK(tq_matrix_mult(d, direct_input_a.data(), N, M, direct_input_b.data(), M, N, cfg, icicle_output.data()));
    ICICLE_CHECK(matrix_mult(direct_input_a.data(), N, M, direct_input_b.data(), M, N, cfg, icicle_output.data()));

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
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // A * I = A
    ICICLE_CHECK(matrix_mult(
      random_matrix.data(), matrix_size, matrix_size, identity.data(), matrix_size, matrix_size, cfg, result.data()));

    ASSERT_EQ(0, std::memcmp(random_matrix.data(), result.data(), random_matrix.size() * sizeof(scalar_t)));

    // I * A = A
    ICICLE_CHECK(matrix_mult(
      identity.data(), matrix_size, matrix_size, random_matrix.data(), matrix_size, matrix_size, cfg, result.data()));

    ASSERT_EQ(0, std::memcmp(random_matrix.data(), result.data(), random_matrix.size() * sizeof(scalar_t)));

    // Test 2: Zero matrix multiplication
    std::vector<scalar_t> zero_matrix(matrix_size * matrix_size, scalar_t::zero());
    std::vector<scalar_t> expected_zero(matrix_size * matrix_size, scalar_t::zero());

    // A * 0 = 0
    ICICLE_CHECK(matrix_mult(
      random_matrix.data(), matrix_size, matrix_size, zero_matrix.data(), matrix_size, matrix_size, cfg,
      result.data()));

    ASSERT_EQ(0, std::memcmp(expected_zero.data(), result.data(), expected_zero.size() * sizeof(scalar_t)));

    // 0 * A = 0
    ICICLE_CHECK(matrix_mult(
      zero_matrix.data(), matrix_size, matrix_size, random_matrix.data(), matrix_size, matrix_size, cfg,
      result.data()));

    ASSERT_EQ(0, std::memcmp(expected_zero.data(), result.data(), expected_zero.size() * sizeof(scalar_t)));

    // Test 3: Matrix multiplication with 1x1 matrices
    std::vector<scalar_t> single_a(1);
    std::vector<scalar_t> single_b(1);
    std::vector<scalar_t> single_result(1);

    scalar_t::rand_host_many(single_a.data(), 1);
    scalar_t::rand_host_many(single_b.data(), 1);

    ICICLE_CHECK(matrix_mult(single_a.data(), 1, 1, single_b.data(), 1, 1, cfg, single_result.data()));

    ASSERT_EQ(single_a[0] * single_b[0], single_result[0]);

    // Test 4: Matrix-Vector multiplication
    std::vector<scalar_t> vector(matrix_size);
    std::vector<scalar_t> vector_result(matrix_size);
    scalar_t::rand_host_many(vector.data(), matrix_size);

    // Matrix * Vector
    ICICLE_CHECK(matrix_mult(
      random_matrix.data(), matrix_size, matrix_size, vector.data(), matrix_size, 1, cfg, vector_result.data()));

    // Verify result dimensions and values
    std::vector<scalar_t> expected_vector_result(matrix_size);
    multiply_matrices(random_matrix, vector, expected_vector_result, matrix_size, matrix_size, 1);

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
      matrix_mult(vector_a.data(), matrix_size, 1, vector_b.data(), 1, matrix_size, cfg, outer_product_result.data()));

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
    ICICLE_CHECK(matrix_mult(
      vector.data(), 1, matrix_size, random_matrix.data(), matrix_size, matrix_size, cfg, vector_matrix_result.data()));

    // Verify result dimensions and values
    std::vector<scalar_t> expected_vector_matrix_result(matrix_size);
    multiply_matrices(vector, random_matrix, expected_vector_matrix_result, 1, matrix_size, matrix_size);

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
  auto cfg = VecOpsConfig{};
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
    auto error = matrix_mult(
      matrix_a.data(), matrix_size, matrix_size, matrix_b.data(), matrix_size + 1, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);
  }
}

TEST_F(MatrixTestBase, MatrixMultiplicationBatchedDimensionMismatch)
{
  const size_t matrix_size = 4;
  const size_t batch_size = 4;
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;
  cfg.batch_size = batch_size;

  std::vector<std::vector<scalar_t>> batch_a(batch_size, std::vector<scalar_t>(matrix_size * matrix_size));
  std::vector<scalar_t> single_b((matrix_size + 1) * matrix_size); // Different size

  for (size_t i = 0; i < batch_size; i++) {
    scalar_t::rand_host_many(batch_a[i].data(), matrix_size * matrix_size);
  }
  scalar_t::rand_host_many(single_b.data(), (matrix_size + 1) * matrix_size);

  std::vector<std::vector<scalar_t>> batch_result(batch_size, std::vector<scalar_t>(matrix_size * matrix_size));

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Should fail for each matrix in batch
    for (size_t i = 0; i < batch_size; i++) {
      auto error = matrix_mult(
        batch_a[i].data(), matrix_size, matrix_size, single_b.data(), matrix_size + 1, matrix_size, cfg,
        batch_result[i].data());
      ASSERT_NE(error, eIcicleError::SUCCESS);
    }
  }
}

TEST_F(MatrixTestBase, MatrixMultiplicationNullInputs)
{
  const size_t matrix_size = 4;
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  std::vector<scalar_t> valid_matrix(matrix_size * matrix_size);
  std::vector<scalar_t> result(matrix_size * matrix_size);
  scalar_t::rand_host_many(valid_matrix.data(), matrix_size * matrix_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Test null A matrix
    auto error = matrix_mult(
      static_cast<const scalar_t*>(nullptr), matrix_size, matrix_size,
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);

    // Test null B matrix
    error = matrix_mult(
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size,
      static_cast<const scalar_t*>(nullptr), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);

    // Test null result matrix
    error = matrix_mult(
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size,
      static_cast<const scalar_t*>(valid_matrix.data()), matrix_size, matrix_size, cfg,
      static_cast<scalar_t*>(nullptr));
    ASSERT_NE(error, eIcicleError::SUCCESS);
  }
}

TEST_F(MatrixTestBase, MatrixMultiplicationZeroDimensions)
{
  const size_t matrix_size = 4;
  auto cfg = VecOpsConfig{};
  cfg.is_a_on_device = false;
  cfg.is_b_on_device = false;
  cfg.is_result_on_device = false;

  std::vector<scalar_t> valid_matrix(matrix_size * matrix_size);
  std::vector<scalar_t> result(matrix_size * matrix_size);
  scalar_t::rand_host_many(valid_matrix.data(), matrix_size * matrix_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Test zero rows
    auto error = matrix_mult(
      valid_matrix.data(), 0, matrix_size, valid_matrix.data(), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);

    // Test zero columns
    error = matrix_mult(
      valid_matrix.data(), matrix_size, 0, valid_matrix.data(), matrix_size, matrix_size, cfg, result.data());
    ASSERT_NE(error, eIcicleError::SUCCESS);
  }
}