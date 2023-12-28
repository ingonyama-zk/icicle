#include <gtest/gtest.h>
#include "../utils/error_handler.cuh"  // Include your error handling header file

class IcicleErrorTest : public ::testing::Test {
protected:
    // You can define helper functions or setup code here if needed
};

TEST_F(IcicleErrorTest, UndefinedErrorString) {
    std::string expected = "Undefined error occurred.";
    EXPECT_EQ(IcicleGetErrorString(IcicleError_t::UndefinedError), expected);
}

TEST_F(IcicleErrorTest, UnknownErrorCodeString) {
    // Using a made-up error code to test the default case
    std::string expected = "Unknown error code.";
    EXPECT_EQ(IcicleGetErrorString(static_cast<IcicleError_t>(999)), expected);
}

TEST_F(IcicleErrorTest, IcicleErrorWithCudaError) {
    // Example test for IcicleError constructor with cudaError_t
    std::string msg = "Test Message";
    cudaError_t cudaErr = cudaErrorInvalidDevice;

    IcicleError error(cudaErr, msg);
    EXPECT_EQ(error.getErrorCode(), static_cast<int>(cudaErr));
    EXPECT_NE(std::string(error.what()).find("CUDA Error:"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find(msg), std::string::npos);
}

TEST_F(IcicleErrorTest, IcicleErrorWithIcicleErrorCode) {
    // Example test for IcicleError constructor with IcicleError_t
    std::string msg = "Custom Error";
    IcicleError error(IcicleError_t::UndefinedError, msg);
    EXPECT_EQ(error.getErrorCode(), static_cast<int>(IcicleError_t::UndefinedError));
    EXPECT_NE(std::string(error.what()).find("Icicle Error:"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find(msg), std::string::npos);
}

