
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
static inline std::string s_main_target;
static inline std::string s_ref_target;

class CurveApiTest : public ::testing::Test
{
public:
  static inline std::list<std::string> s_regsitered_devices;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    icicle_load_backend(BACKEND_BUILD_DIR, true);

    // check targets are loaded and choose main and reference targets
    auto regsitered_devices = get_registered_devices_list();
    ASSERT_GE(regsitered_devices.size(), 2);

    const bool is_cuda_registered = is_device_registered("CUDA");
    const bool is_cpu_registered = is_device_registered("CPU");
    const bool is_cpu_ref_registered = is_device_registered("CPU_REF");
    // if cuda is available, want main="CUDA", ref="CPU", otherwise main="CPU", ref="CPU_REF".
    s_main_target = is_cuda_registered ? "CUDA" : "CPU";
    s_ref_target = is_cuda_registered ? "CPU" : "CPU_REF";
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}

  template <typename T>
  void random_scalars(T* arr, uint64_t count)
  {
    for (uint64_t i = 0; i < count; i++)
      arr[i] = i < 1000 ? T::rand_host() : arr[i - 1000];
  }

  template <typename A, typename P>
  void MSM_test()
  {
    const int logn = 5;
    const int N = 1 << logn;
    auto scalars = std::make_unique<scalar_t[]>(N);
    auto bases = std::make_unique<A[]>(N);

    scalar_t::rand_host_many(scalars.get(), N);
    P::rand_host_many(bases.get(), N);

    P result_main{};
    P result_ref{};

    auto run = [&](const std::string& dev_type, P* result, const char* msg, bool measure, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      auto config = default_msm_config();
      START_TIMER(MSM_sync)
      for (int i = 0; i < iters; ++i) {
        // TODO real test
        msm_precompute_bases(bases.get(), N, config, bases.get());
        msm(scalars.get(), bases.get(), N, config, result);
      }
      END_TIMER(MSM_sync, oss.str().c_str(), measure);
    };

    run(s_main_target, &result_main, "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
    run(s_ref_target, &result_ref, "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
    ASSERT_EQ(result_ref, result_main);
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

TEST_F(CurveApiTest, msm) { MSM_test<affine_t, projective_t>(); }
TEST_F(CurveApiTest, MontConversionAffine) { mont_conversion_test<affine_t, projective_t>(); }
TEST_F(CurveApiTest, MontConversionProjective) { mont_conversion_test<projective_t, projective_t>(); }

#ifdef G2
TEST_F(CurveApiTest, msmG2) { MSM_test<g2_affine_t, g2_projective_t>(); }
TEST_F(CurveApiTest, MontConversionG2Affine) { mont_conversion_test<g2_affine_t, g2_projective_t>(); }
TEST_F(CurveApiTest, MontConversionG2Projective) { mont_conversion_test<g2_projective_t, g2_projective_t>(); }
#endif // G2

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
    ntt_init_domain(scalar_t::omega(logn), init_domain_config);

    auto config = default_ntt_config<scalar_t>();

    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i)
      ntt(input.get(), N, NTTDir::kForward, config, out);
    END_TIMER(NTT_sync, msg, measure);

    ntt_release_domain<scalar_t>();
  };

  run(s_main_target, out_main.get(), "ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  run(s_ref_target, out_ref.get(), "ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  // ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), N * sizeof(projective_t))); // TODO ucomment when CPU is
  // implemented
}
#endif // ECNTT

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}