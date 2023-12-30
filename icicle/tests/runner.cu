#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

//include list of test files
#include "error_handler_test.cu"
#include "primitives_test.cu"
#include "device_error_test.cu"

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  printf("running gtests...\n");
  return RUN_ALL_TESTS();
}
