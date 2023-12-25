#include "err.h" // Include your error handling header file
#include <gtest/gtest.h>

__global__ void a_kernel_with_sticky_error()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Deliberately assert false
  if (idx == 0) { assert(false); }
}

// Test Fixture for CUDA tests
class CudaErrorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Perform any setup needed before each test
    cudaError_t err = cudaGetLastError(); // Clear any existing errors
  }

  void TearDown() override
  {
    // Clean up after each test if necessary
  }
};

// Test Case for Non-Sticky Error
TEST_F(CudaErrorTest, NonStickyErrorTest)
{
  cudaGetLastError();

  // Deliberately cause a non-sticky CUDA error
  cudaError_t err = cudaMalloc(nullptr, 0); // This should cause cudaErrorInvalidValue
  cudaError_t err2;

  // Check if the macro correctly reports the error without throwing an exception
  EXPECT_EQ(err, cudaErrorInvalidValue);
  EXPECT_NO_THROW({ err2 = CHECK_LAST_IS_STICKY_ERROR(); });
  EXPECT_EQ(err2, err);

  // Optionally, clear the error if needed
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// Test Case for Sticky Error
TEST_F(CudaErrorTest, StickyErrorTest)
{
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // Deliberately cause a sticky CUDA error
  a_kernel_with_sticky_error<<<1, 1>>>();

  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  cudaError_t sync_error = cudaDeviceSynchronize();

  EXPECT_NE(sync_error, cudaSuccess);

  // Check if the macro correctly throws an exception for a sticky error
  EXPECT_THROW({ CHECK_LAST_IS_STICKY_ERROR(); }, IcicleError);
}
