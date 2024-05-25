#include "gpu-utils/error_handler.cuh" // Include your error handling header file
#include <gtest/gtest.h>


#include "../src/vec_ops/acc.cu"

// using vec_ops;


class AccTest : public ::testing::Test
{
protected:
  // You can define helper functions or setup code here if needed
};

TEST(AccumulateTest, BasicTest) {
    int packed_len = 777;
    std::vector<float> column_data(packed_len, 1.0f);
    std::vector<float> other_data(packed_len, 2.0f);

    vec_ops::SecureColumn<float> column;
    vec_ops::SecureColumn<float> other;

    column.data = column_data.data();
    column.packed_len = packed_len;

    other.data = other_data.data();
    other.packed_len = packed_len;

    vec_ops::accumulate_async(column, other);

    for (int i = 0; i < packed_len; ++i) {
        EXPECT_FLOAT_EQ(column.data[i], 3.0f);
    }
}
