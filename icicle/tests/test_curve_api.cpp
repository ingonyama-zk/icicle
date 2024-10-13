
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <list>
#include <random>
#include "dlfcn.h"

#include "icicle/runtime.h"
#include "icicle/ntt.h"
#include "icicle/msm.h"
#include "icicle/vec_ops.h"
#include "icicle/curves/montgomery_conversion.h"
#include "icicle/curves/curve_config.h"
#include "icicle/backend/msm_config.h"

using namespace curve_config;
using namespace icicle;

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

static bool VERBOSE = true;
static inline std::string s_main_target;
static inline std::string s_ref_target;

class CurveApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_registered_devices;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
#ifdef BACKEND_BUILD_DIR
    setenv("ICICLE_BACKEND_INSTALL_DIR", BACKEND_BUILD_DIR, 0 /*=replace*/);
#endif
    icicle_load_backend_from_env_or_default();

    const bool is_cuda_registered = is_device_registered("CUDA");
    if (!is_cuda_registered) { ICICLE_LOG_ERROR << "CUDA device not found. Testing CPU vs CPU"; }
    s_main_target = is_cuda_registered ? "CUDA" : "CPU";
    s_ref_target = "CPU";
  }
  static void TearDownTestSuite()
  {
    // make sure to fail in CI if only have one device
    ICICLE_ASSERT(is_device_registered("CUDA")) << "missing CUDA backend";
  }

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}

  template <typename T>
  void random_scalars(T* arr, uint64_t count)
  {
    for (uint64_t i = 0; i < count; i++)
      arr[i] = T::rand_host();
  }

  template <typename A, typename P>
  void MSM_test()
  {
    const int logn = 12;
    const int batch = 3;
    const int N = 1 << logn;
    const int precompute_factor = (rand() & 7) + 1; // between 1 and 8
    const int total_nof_elemets = batch * N;

    auto scalars = std::make_unique<scalar_t[]>(total_nof_elemets);
    auto bases = std::make_unique<A[]>(N);
    auto precomp_bases = std::make_unique<A[]>(N * precompute_factor);
    scalar_t::rand_host_many(scalars.get(), total_nof_elemets);
    P::rand_host_many(bases.get(), N);

    auto result_main = std::make_unique<P[]>(batch);
    auto result_ref = std::make_unique<P[]>(batch);

    auto config = default_msm_config();
    config.batch_size = batch;
    config.are_points_shared_in_batch = true;
    config.precompute_factor = precompute_factor;

    auto run = [&](const std::string& dev_type, P* result, const char* msg, bool measure, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      ICICLE_CHECK(msm_precompute_bases(bases.get(), N, config, precomp_bases.get()));

      START_TIMER(MSM_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(msm(scalars.get(), precomp_bases.get(), N, config, result));
      }
      END_TIMER(MSM_sync, oss.str().c_str(), measure);
    };

    run(s_main_target, result_main.get(), "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
    run(s_ref_target, result_ref.get(), "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
    for (int res_idx = 0; res_idx < batch; ++res_idx) {
      ASSERT_EQ(true, P::is_on_curve(result_main[res_idx]));
      ASSERT_EQ(true, P::is_on_curve(result_ref[res_idx]));
      ASSERT_EQ(result_main[res_idx], result_ref[res_idx]);
    }
  }

  template <typename T, typename P>
  void mont_conversion_test()
  {
    const int N = 1 << 6;
    auto elements = std::make_unique<T[]>(N);
    auto main_output = std::make_unique<T[]>(N);
    auto ref_output = std::make_unique<T[]>(N);
    P::rand_host_many(elements.get(), N);

    auto run =
      [&](const std::string& dev_type, T* input, T* output, bool into_mont, bool measure, const char* msg, int iters) {
        Device dev = {dev_type, 0};
        icicle_set_device(dev);
        auto config = default_vec_ops_config();

        std::ostringstream oss;
        oss << dev_type << " " << msg;

        START_TIMER(MONTGOMERY)
        for (int i = 0; i < iters; ++i) {
          ICICLE_CHECK(convert_montgomery(input, N, into_mont, config, output));
        }
        END_TIMER(MONTGOMERY, oss.str().c_str(), measure);
      };

    run(s_main_target, elements.get(), main_output.get(), true /*into*/, VERBOSE /*=measure*/, "to-montgomery", 1);
    run(s_ref_target, elements.get(), ref_output.get(), true /*into*/, VERBOSE /*=measure*/, "to-montgomery", 1);
    ASSERT_EQ(0, memcmp(main_output.get(), ref_output.get(), N * sizeof(T)));

    run(s_main_target, main_output.get(), main_output.get(), false /*from*/, VERBOSE /*=measure*/, "to-montgomery", 1);
    run(s_ref_target, ref_output.get(), ref_output.get(), false /*from*/, VERBOSE /*=measure*/, "to-montgomery", 1);
    ASSERT_EQ(0, memcmp(main_output.get(), ref_output.get(), N * sizeof(T)));
  }
};

#ifdef MSM
TEST_F(CurveApiTest, msm) { MSM_test<affine_t, projective_t>(); }
TEST_F(CurveApiTest, MontConversionAffine) { mont_conversion_test<affine_t, projective_t>(); }
TEST_F(CurveApiTest, MontConversionProjective) { mont_conversion_test<projective_t, projective_t>(); }

  #ifdef G2
TEST_F(CurveApiTest, msmG2) { MSM_test<g2_affine_t, g2_projective_t>(); }
TEST_F(CurveApiTest, MontConversionG2Affine) { mont_conversion_test<g2_affine_t, g2_projective_t>(); }
TEST_F(CurveApiTest, MontConversionG2Projective) { mont_conversion_test<g2_projective_t, g2_projective_t>(); }
  #endif // G2
#endif   // MSM

#ifdef ECNTT
TEST_F(CurveApiTest, ecntt)
{
  const int logn = 5;
  const int N = 1 << logn;
  auto input = std::make_unique<projective_t[]>(N);
  projective_t::rand_host_many(input.get(), N);

  auto out_main = std::make_unique<projective_t[]>(N);
  auto out_ref = std::make_unique<projective_t[]>(N);

  auto run = [&](const std::string& dev_type, projective_t* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    auto init_domain_config = default_ntt_init_domain_config();
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(logn), init_domain_config));

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto config = default_ntt_config<scalar_t>();

    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i)
      ICICLE_CHECK(ntt(input.get(), N, NTTDir::kForward, config, out));
    END_TIMER(NTT_sync, oss.str().c_str(), measure);

    ntt_release_domain<scalar_t>();
  };

  run(s_main_target, out_main.get(), "ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  run(s_ref_target, out_ref.get(), "ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  // ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), N * sizeof(projective_t))); // TODO ucomment when CPU is
  // implemented
}
#endif // ECNTT

template <typename T>
class CurveSanity : public ::testing::Test
{
};

#ifdef G2
typedef testing::Types<projective_t, g2_projective_t> CTImplementations;
#else
typedef testing::Types<projective_t> CTImplementations;
#endif

TYPED_TEST_SUITE(CurveSanity, CTImplementations);

// Note: this is testing host arithmetic. Other tests against CPU backend should guarantee correct device arithmetic too
TYPED_TEST(CurveSanity, CurveSanityTest)
{
  auto a = TypeParam::rand_host();
  auto b = TypeParam::rand_host();
  ASSERT_EQ(true, TypeParam::is_on_curve(a) && TypeParam::is_on_curve(b));               // rand is on curve
  ASSERT_EQ(a + TypeParam::zero(), a);                                                   // zero addition
  ASSERT_EQ(a + b - a, b);                                                               // addition,subtraction cancel
  ASSERT_EQ(a + TypeParam::neg(a), TypeParam::zero());                                   // addition with neg cancel
  ASSERT_EQ(a + a + a, scalar_t::from(3) * a);                                           // scalar multiplication
  ASSERT_EQ(scalar_t::from(3) * (a + b), scalar_t::from(3) * a + scalar_t::from(3) * b); // distributive
  ASSERT_EQ(a + b, a + TypeParam::to_affine(b)); // mixed addition projective+affine
  ASSERT_EQ(a - b, a - TypeParam::to_affine(b)); // mixed subtraction projective-affine
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}