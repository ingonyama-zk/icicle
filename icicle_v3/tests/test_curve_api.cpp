
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"

#include "icicle/runtime.h"
#include "icicle/msm.h"
#include "icicle/vec_ops.h"
#include "icicle/curves/montgomery_conversion.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

static bool VERBOSE = true;

class CurveApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    icicle_load_backend(BACKEND_BUILD_DIR);
    s_regsitered_devices = get_registered_devices();
    ASSERT_GT(s_regsitered_devices.size(), 0);
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CurveApiTest, MSM)
{
  const int logn = 5;
  const int N = 1 << logn;
  auto scalars = std::make_unique<scalar_t[]>(N);
  auto bases = std::make_unique<affine_t[]>(N);

  scalar_t::rand_host_many(scalars.get(), N);
  projective_t::rand_host_many_affine(bases.get(), N);

  projective_t result{};

  auto run = [&](const char* dev_type, projective_t* result, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    auto config = default_msm_config();

    START_TIMER(MSM_sync)
    for (int i = 0; i < iters; ++i) {
      // TODO real test
      msm_precompute_bases(bases.get(), N, 1, default_msm_pre_compute_config(), bases.get());
      msm(scalars.get(), bases.get(), N, config, result);
    }
    END_TIMER(MSM_sync, msg, measure);
  };

  run("CPU", &result, "CPU msm", VERBOSE /*=measure*/, 1 /*=iters*/);
  // TODO test something
}

#ifdef G2
TEST_F(CurveApiTest, MSM_G2)
{
  const int logn = 5;
  const int N = 1 << logn;
  auto scalars = std::make_unique<scalar_t[]>(N);
  auto bases = std::make_unique<g2_affine_t[]>(N);

  scalar_t::rand_host_many(scalars.get(), N);
  g2_projective_t::rand_host_many_affine(bases.get(), N);

  g2_projective_t result{};

  auto run = [&](const char* dev_type, g2_projective_t* result, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    auto config = default_msm_config();

    START_TIMER(MSM_sync)
    for (int i = 0; i < iters; ++i) {
      // TODO real test
      msm_precompute_bases(bases.get(), N, 1, default_msm_pre_compute_config(), bases.get());
      msm(scalars.get(), bases.get(), N, config, result);
    }
    END_TIMER(MSM_sync, msg, measure);
  };

  run("CPU", &result, "CPU msm g2", VERBOSE /*=measure*/, 1 /*=iters*/);
  // TODO test something
}
#endif // G2

TEST_F(CurveApiTest, MontConversion)
{
  // Note: this test doesn't really test correct mont conversion (since there is no arithmetic in mont) but checks that
  // it does some conversion and back to original
  affine_t affine_point, affine_point_converted;
  projective_t projective_point, projective_point_converted;

  projective_point = projective_t::rand_host();
  affine_point = projective_point.to_affine();

  icicle_set_device({"CPU", 0});

  // (1) converting to mont and check not equal to original
  auto config = default_convert_montgomery_config();
  points_convert_montgomery(&affine_point, 1, true /*into mont*/, config, &affine_point_converted);
  points_convert_montgomery(&projective_point, 1, true /*into mont*/, config, &projective_point_converted);

  ASSERT_NE(affine_point, affine_point_converted);             // check that it was converted to mont
  ASSERT_NE(projective_point.x, projective_point_converted.x); // check that it was converted to mont

  // (2) converting back from mont and check equal
  points_convert_montgomery(&projective_point_converted, 1, false /*from mont*/, config, &projective_point_converted);
  points_convert_montgomery(&affine_point_converted, 1, false /*from mont*/, config, &affine_point_converted);

  ASSERT_EQ(affine_point, affine_point_converted);
  ASSERT_EQ(projective_point, projective_point_converted);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}