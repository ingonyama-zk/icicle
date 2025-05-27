
#include <gtest/gtest.h>
#include <iostream>

#include "test_base.h"
#include "icicle/runtime.h"
#include "icicle/utils/rand_gen.h"

using namespace icicle;

class PqcTest : public IcicleTestBase
{
};

TEST_F(PqcTest, MLKEM)
{
  // TODO: test ML-KEM functionality
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}