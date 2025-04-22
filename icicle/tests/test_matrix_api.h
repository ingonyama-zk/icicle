
#pragma once

#include <cstdint>
#include <gtest/gtest.h>

#include "icicle/vec_ops.h"

#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"


#include "test_base.h"


class MatrixTestBase : public IcicleTestBase
{
};


// Batched matrix multiplication
TEST_F(MatrixTestBase, MatrixMultiplicationBatched)
{
  const size_t N = 1 << 16;
  auto direct_input_a = std::vector<scalar_t>(N);
  auto direct_input_b = std::vector<scalar_t>(N);
  auto direct_output = std::vector<scalar_t>(N);
  //TODO: implement
  ASSERT_TRUE(false);
}

// Matrix multiplication with non-square matrices
TEST_F(MatrixTestBase, MatrixMultiplicationNonSquare)
{
  const size_t N = 1 << 16;
  auto direct_input = std::vector<scalar_t>(N);
  auto direct_output = std::vector<scalar_t>(N);

  //TODO: implement
  ASSERT_TRUE(false);
}