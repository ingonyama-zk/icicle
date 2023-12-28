#include "err.h" // Include your error handling header file
#include <gtest/gtest.h>

__global__ void a_kernel_with_conditional_sticky_error(bool is_failing)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaGetLastError() == cudaSuccess);
    // Deliberately assert false
    assert(!is_failing); // TODO: sticky according to https://stackoverflow.com/a/43659538
  }
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
  EXPECT_NO_THROW({ err2 = CHK_LAST(); });
  EXPECT_EQ(err2, err);

  // Optionally, clear the error if needed
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// Test Case for Sticky Error
TEST_F(CudaErrorTest, StickyErrorTest)
{
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // Deliberately cause a sticky CUDA error
  a_kernel_with_conditional_sticky_error<<<1, 1>>>(true);

  EXPECT_EQ(cudaGetLastError(), cudaSuccess); // no error until synchronization

  // Launch without error
  a_kernel_with_conditional_sticky_error<<<1, 1>>>(false);

  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  cudaError_t sync_error = cudaDeviceSynchronize(); // only cudaDeviceSynchronize() can help
                                                    // determine sticky error reliably,
                                                    // returning same error as failed kernel

  EXPECT_NE(sync_error, cudaSuccess);
  EXPECT_EQ(sync_error, cudaErrorAssert);

  EXPECT_EQ(cudaGetLastError(), cudaErrorAssert); // reports error after cudaDeviceSynchronize
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);     // resets error, despite it's sticky

  // Check if the macro correctly throws an exception for a sticky error
  EXPECT_THROW({ CHK_STICKY(cudaDeviceSynchronize()); }, IcicleError);
}

// Test Case for Sticky Error
TEST_F(CudaErrorTest, StickyErrorTestNotThrowing)
{
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // Deliberately cause a sticky CUDA error
  a_kernel_with_conditional_sticky_error<<<1, 1>>>(true);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaErrorAssert);

  // Check if the macro correctly throws an exception for a sticky error
  cudaError_t err = CHK_LAST_STICKY_NO_THROW();
  EXPECT_EQ(err, cudaErrorAssert);
}