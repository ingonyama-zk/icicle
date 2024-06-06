
#include <gtest/gtest.h>
#include <iostream>
#include "dlfcn.h"

#include "icicle/runtime.h"
#include "icicle/ecntt.h"
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

  template <typename A, typename P>
  void MSM_test()
  {
    const int logn = 5;
    const int N = 1 << logn;
    auto scalars = std::make_unique<scalar_t[]>(N);
    auto bases = std::make_unique<A[]>(N);

    scalar_t::rand_host_many(scalars.get(), N);
    P::rand_host_many_affine(bases.get(), N);

    P result{};

    auto run = [&](const char* dev_type, P* result, const char* msg, bool measure, int iters) {
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

  template <typename A, typename P>
  void mont_conversion_test()
  {
    // Note: this test doesn't really test correct mont conversion (since there is no arithmetic in mont) but checks
    // that
    // it does some conversion and back to original
    A affine_point, affine_point_converted;
    P projective_point, projective_point_converted;

    projective_point = P::rand_host();
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
};

TEST_F(CurveApiTest, MSM) { MSM_test<affine_t, projective_t>(); }
TEST_F(CurveApiTest, MontConversion) { mont_conversion_test<affine_t, projective_t>(); }

#ifdef G2
TEST_F(CurveApiTest, MSM_G2) { MSM_test<g2_affine_t, g2_projective_t>(); }
TEST_F(CurveApiTest, MontConversionG2) { mont_conversion_test<g2_affine_t, g2_projective_t>(); }
#endif // G2

#ifdef ECNTT
TEST_F(CurveApiTest, ecntt)
{
  const int logn = 15;
  const int N = 1 << logn;
  auto input = std::make_unique<projective_t[]>(N);
  projective_t::rand_host_many(input.get(), N);

  auto out_cpu = std::make_unique<projective_t[]>(N);
  auto out_cuda = std::make_unique<projective_t[]>(N);

  auto run = [&](const char* dev_type, projective_t* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    ntt_init_domain(scalar_t::omega(logn), ConfigExtension());

    auto config = default_ntt_config<scalar_t>();

    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i)
      ntt(input.get(), N, NTTDir::kForward, config, out);
    END_TIMER(NTT_sync, msg, measure);

    ntt_release_domain<scalar_t>();
  };

  run("CPU", out_cpu.get(), "CPU ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
}
#endif // ECNTT

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}