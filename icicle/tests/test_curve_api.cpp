
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <list>
#include <random>

#include "icicle/runtime.h"
#include "icicle/ntt.h"
#include "icicle/msm.h"
#include "icicle/vec_ops.h"
#include "icicle/curves/montgomery_conversion.h"
#include "icicle/curves/curve_config.h"
#include "icicle/backend/msm_config.h"
#include "icicle/backend/ntt_config.h"

#include "test_base.h"

using namespace curve_config;
using namespace icicle;

static bool VERBOSE = true;

class CurveApiTest : public IcicleTestBase
{
public:
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
    const int N = (1 << logn) - rand() % (5 * logn); // make it not always power of two
    const int precompute_factor = (rand() & 7) + 1;  // between 1 and 8
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

    run(IcicleTestBase::main_device(), result_main.get(), "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
    run(IcicleTestBase::reference_device(), result_ref.get(), "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
    for (int res_idx = 0; res_idx < batch; ++res_idx) {
      ASSERT_EQ(true, P::is_on_curve(result_main[res_idx]));
      ASSERT_EQ(true, P::is_on_curve(result_ref[res_idx]));
      ASSERT_EQ(result_main[res_idx], result_ref[res_idx]);
    }
  }

  template <typename A, typename P>
  void MSM_CPU_THREADS_test()
  {
    const int logn = 8;
    const int c = 3;
    // Low c to have a large amount of tasks required in phase 2
    // For example for bn254: #bms = ceil(254/3)=85
    // #tasks in phase 2 = 2 * #bms = 170 > 64 = TASK_PER_THREAD
    // As such the default amount of tasks and 1 thread shouldn't be enough and the program should readjust the task
    // number per thread.
    const int batch = 3;
    const int N = (1 << logn) - rand() % (5 * logn); // make it not always power of two
    const int precompute_factor = 1;                 // Precompute is 1 to increase number of BMs
    const int total_nof_elemets = batch * N;

    auto scalars = std::make_unique<scalar_t[]>(total_nof_elemets);
    auto bases = std::make_unique<A[]>(N);
    scalar_t::rand_host_many(scalars.get(), total_nof_elemets);
    P::rand_host_many(bases.get(), N);

    auto result_multi_thread = std::make_unique<P[]>(batch);
    auto result_single_thread = std::make_unique<P[]>(batch);

    auto config = default_msm_config();
    config.batch_size = batch;
    config.are_points_shared_in_batch = true;
    config.precompute_factor = precompute_factor;
    config.c = c;

    auto run = [&](const std::string& dev_type, P* result, const char* msg, bool measure, int iters) {
      Device dev = {dev_type, 0};
      icicle_set_device(dev);

      std::ostringstream oss;
      oss << dev_type << " " << msg;

      START_TIMER(MSM_sync)
      for (int i = 0; i < iters; ++i) {
        ICICLE_CHECK(msm(scalars.get(), bases.get(), N, config, result));
      }
      END_TIMER(MSM_sync, oss.str().c_str(), measure);
    };
    if (IcicleTestBase::reference_device() == "CPU") {
      run(IcicleTestBase::reference_device(), result_multi_thread.get(), "msm", VERBOSE /*=measure*/, 1 /*=iters*/);
      // Adjust config to have one worker thread
      ConfigExtension ext;
      ext.set(CpuBackendConfig::CPU_NOF_THREADS, 1);
      config.ext = &ext;
      run(IcicleTestBase::reference_device(), result_single_thread.get(), "msm", VERBOSE /*=measure*/, 1 /*=iters*/);

      for (int res_idx = 0; res_idx < batch; ++res_idx) {
        ASSERT_EQ(true, P::is_on_curve(result_multi_thread[res_idx]));
        ASSERT_EQ(true, P::is_on_curve(result_single_thread[res_idx]));
        ASSERT_EQ(result_multi_thread[res_idx], result_single_thread[res_idx]);
      }
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

    run(
      IcicleTestBase::main_device(), elements.get(), main_output.get(), true /*into*/, VERBOSE /*=measure*/,
      "to-montgomery", 1);
    run(
      IcicleTestBase::reference_device(), elements.get(), ref_output.get(), true /*into*/, VERBOSE /*=measure*/,
      "to-montgomery", 1);
    ASSERT_EQ(0, memcmp(main_output.get(), ref_output.get(), N * sizeof(T)));

    run(
      IcicleTestBase::main_device(), main_output.get(), main_output.get(), false /*from*/, VERBOSE /*=measure*/,
      "to-montgomery", 1);
    run(
      IcicleTestBase::reference_device(), ref_output.get(), ref_output.get(), false /*from*/, VERBOSE /*=measure*/,
      "to-montgomery", 1);
    ASSERT_EQ(0, memcmp(main_output.get(), ref_output.get(), N * sizeof(T)));
  }
};

#ifdef MSM
TEST_F(CurveApiTest, msm) { MSM_test<affine_t, projective_t>(); }
TEST_F(CurveApiTest, msmCpuThreads) { MSM_CPU_THREADS_test<affine_t, projective_t>(); }
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

  run(IcicleTestBase::main_device(), out_main.get(), "ecntt", VERBOSE /*=measure*/, 1);
  run(IcicleTestBase::reference_device(), out_ref.get(), "ecntt", VERBOSE /*=measure*/, 1);

  // note that memcmp is tricky here because projetive points can have many representations
  for (uint64_t i = 0; i < N; ++i) {
    ASSERT_FALSE(projective_t::is_zero(out_ref[i]));
    ASSERT_EQ(out_ref[i], out_main[i]);
  }
}

TEST_F(CurveApiTest, ecnttDeviceMem)
{
  // (TODO) Randomize configuration
  const bool inplace = false;
  const int logn = 10;
  const uint64_t N = 1 << logn;
  const int log_ntt_domain_size = logn;
  const int log_batch_size = 0;
  const int batch_size = 1 << log_batch_size;
  const Ordering ordering = static_cast<Ordering>(0);
  bool columns_batch = false;
  const NTTDir dir = static_cast<NTTDir>(0); // 0: forward, 1: inverse

  const int total_size = N * batch_size;
  auto input = std::make_unique<projective_t[]>(total_size);
  projective_t::rand_host_many(input.get(), total_size);
  auto out_main = std::make_unique<projective_t[]>(total_size);
  auto out_ref = std::make_unique<projective_t[]>(total_size);

  auto run = [&](const std::string& dev_type, projective_t* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // init domain
    auto init_domain_config = default_ntt_init_domain_config();
    ConfigExtension ext;
    ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
    init_domain_config.ext = &ext;
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log_ntt_domain_size), init_domain_config));

    projective_t *d_in, *d_out;
    ICICLE_CHECK(icicle_malloc((void**)&d_in, total_size * sizeof(projective_t)));
    ICICLE_CHECK(icicle_malloc((void**)&d_out, total_size * sizeof(projective_t)));
    ICICLE_CHECK(icicle_copy(d_in, input.get(), total_size * sizeof(projective_t)));

    auto config = default_ntt_config<scalar_t>();
    config.batch_size = batch_size;       // default: 1
    config.columns_batch = columns_batch; // default: false
    config.ordering = ordering;           // default: kNN
    config.are_inputs_on_device = true;
    config.are_outputs_on_device = true;

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(ntt(d_in, N, dir, config, inplace ? d_in : d_out));
    }
    END_TIMER(NTT_sync, oss.str().c_str(), measure);

    ICICLE_CHECK(
      icicle_copy_to_host_async(out, inplace ? d_in : d_out, total_size * sizeof(projective_t), config.stream));

    ICICLE_CHECK(icicle_free(d_in));
    ICICLE_CHECK(icicle_free(d_out));

    ICICLE_CHECK(ntt_release_domain<scalar_t>());
  };

  run(IcicleTestBase::main_device(), out_main.get(), "ecntt", false /*=measure*/, 1 /*=iters*/); // warmup
  run(IcicleTestBase::reference_device(), out_ref.get(), "ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  run(IcicleTestBase::main_device(), out_main.get(), "ecntt", VERBOSE /*=measure*/, 1 /*=iters*/);
  // note that memcmp is tricky here because projetive points can have many representations
  for (uint64_t i = 0; i < N; ++i) {
    ASSERT_FALSE(projective_t::is_zero(out_ref[i]));
    ASSERT_EQ(out_ref[i], out_main[i]);
  }
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

// Note: this is testing host arithmetic. Other tests against CPU backend should guarantee correct device arithmetic
// too
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

TYPED_TEST(CurveSanity, ScalarMultTest)
{
  const auto point = TypeParam::rand_host();
  const auto scalar = scalar_t::rand_host();

  START_TIMER(main)
  const auto mult = scalar * point;
  END_TIMER(main, "scalar mult window method", true);

  auto expected_mult = TypeParam::zero();
  START_TIMER(ref)
  for (int i = 0; i < scalar_t::NBITS; i++) {
    if (i > 0) { expected_mult = TypeParam::dbl(expected_mult); }
    if (scalar.get_scalar_digit(scalar_t::NBITS - i - 1, 1)) { expected_mult = expected_mult + point; }
  }
  END_TIMER(ref, "scalar mult double-and-add", true);

  ASSERT_EQ(mult, expected_mult);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}