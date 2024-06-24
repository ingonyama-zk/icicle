
#include <gtest/gtest.h>
#include <iostream>
#include <list>
#include "dlfcn.h"

#include "icicle/runtime.h"
#include "icicle/msm.h"
#include "icicle/vec_ops.h"
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
#ifdef BACKEND_BUILD_DIR
    icicle_load_backend(BACKEND_BUILD_DIR, true);
    s_regsitered_devices = get_registered_devices();
    ASSERT_GT(s_regsitered_devices.size(), 0);
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

  template <typename T>
  void random_samples(T* arr, uint64_t count)
  {
    for (uint64_t i = 0; i < count; i++)
      arr[i] = i < 1000 ? T::rand_host() : arr[i - 1000];
  }
};

TEST_F(CurveApiTest, MSM)
{
  const int logn = 10;
  const int N = 1 << logn;
  auto scalars = std::make_unique<scalar_t[]>(N);
  auto bases = std::make_unique<affine_t[]>(N);

  bool conv_mont = false;

  scalar_t::rand_host_many(scalars.get(), N);
  projective_t::rand_host_many_affine(bases.get(), N);
  if (conv_mont) {for (int i=0; i<N; i++) bases[i] = affine_t::to_montgomery(bases[i]); }
  projective_t result_cpu{};
  projective_t result_cpu_dbl_n_add{};
  projective_t result_cpu_ref{};

  projective_t result{};

  auto run = [&](const char* dev_type, projective_t* result, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    const int log_p = 2;
    const int c = std::max(logn, 8) - 1;
    const int pcf = 1 << log_p;
    auto config = default_msm_config();
    config.ext.set("c", c);
    config.precompute_factor = pcf;
    config.are_scalars_montgomery_form = false;
    config.are_points_montgomery_form = conv_mont;

    auto pc_config = default_msm_pre_compute_config();
    pc_config.ext.set("c", c);
    pc_config.ext.set("is_mont", config.are_points_montgomery_form);

    auto precomp_bases = std::make_unique<affine_t[]>(N*pcf);
    msm_precompute_bases(bases.get(), N, pcf, pc_config, precomp_bases.get());
    START_TIMER(MSM_sync)
    for (int i = 0; i < iters; ++i) {
      // TODO real test
      // msm_precompute_bases(bases.get(), N, 1, default_msm_pre_compute_config(), bases.get());
      msm(scalars.get(), precomp_bases.get(), N, config, result);
    }
    END_TIMER(MSM_sync, msg, measure);
  };

  // run("CPU", &result_cpu_dbl_n_add, "CPU msm", false /*=measure*/, 1 /*=iters*/); // warmup
  run("CPU", &result_cpu, "CPU msm", VERBOSE /*=measure*/, 1 /*=iters*/);
  run("CPU_REF", &result_cpu_ref, "CPU_REF msm", VERBOSE /*=measure*/, 1 /*=iters*/);
  ASSERT_EQ((result_cpu),(result_cpu_ref));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}