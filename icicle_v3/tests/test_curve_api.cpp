
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
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

template <typename T>
bool read_inputs(T* arr, const int arr_size, const std::string fname)
{
  std::ifstream in_file(fname);
  bool status = in_file.is_open();
  if (status) {
    for (int i = 0; i < arr_size; i++) {
      in_file.read(reinterpret_cast<char*>(&arr[i]), sizeof(T));
    }
    in_file.close();
  }
  return status;
}

template <typename T>
void store_inputs(T* arr, const int arr_size, const std::string fname)
{
  std::ofstream out_file(fname);
  if (!out_file.is_open()) {
    std::cerr << "Failed to open " << fname << " for writing.\n";
    return;
  }
  for (int i = 0; i < arr_size; i++) {
    out_file.write(reinterpret_cast<char*>(&arr[i]), sizeof(T));
  }
  out_file.close();
}

void get_inputs(affine_t* bases, scalar_t* scalars, const int n) // TODO add precompute factor
{
  // Scalars
  std::string scalar_file = "build/generated_data/scalars_N" + std::to_string(n) + ".dat";
  if (!read_inputs<scalar_t>(scalars, n, scalar_file)) {
    std::cout << "Generating scalars.\n";
    scalar_t::rand_host_many(scalars, n);
    store_inputs<scalar_t>(scalars, n, scalar_file);
  }
  // Bases
  std::string base_file = "build/generated_data/bases_N" + std::to_string(n) + ".dat";
  if (!read_inputs<affine_t>(bases, n, base_file)) {
    std::cout << "Generating bases.\n";
    projective_t::rand_host_many_affine(bases, n);
    store_inputs<affine_t>(bases, n, base_file);
  }
}

TEST_F(CurveApiTest, MSM)
{
  const int logn = 12;
  const int N = 1 << logn;
  auto scalars = std::make_unique<scalar_t[]>(N);
  auto bases = std::make_unique<affine_t[]>(N);

  bool conv_mont = false;
  get_inputs(bases.get(), scalars.get(), N);

  if (conv_mont) {
    for (int i = 0; i < N; i++)
      bases[i] = affine_t::to_montgomery(bases[i]);
  }
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

    const int n_threads = 8;
    const int tasks_per_thread = 2;

    auto config = default_msm_config();
    config.ext.set("c", c);
    config.ext.set("n_threads", n_threads);
    config.ext.set("tasks_per_thread", tasks_per_thread);
    config.precompute_factor = pcf;
    config.are_scalars_montgomery_form = false;
    config.are_points_montgomery_form = conv_mont;

    auto pc_config = default_msm_pre_compute_config();
    pc_config.ext.set("c", c);
    pc_config.ext.set("is_mont", config.are_points_montgomery_form);

    auto precomp_bases = std::make_unique<affine_t[]>(N * pcf);
    // TODO update cmake to include directory?
    std::string precomp_fname =
      "build/generated_data/precomp_N" + std::to_string(N) + "_pcf" + std::to_string(pcf) + ".dat";
    if (!read_inputs<affine_t>(precomp_bases.get(), N * pcf, precomp_fname)) {
      std::cout << "Precomputing bases." << '\n';
      msm_precompute_bases(bases.get(), N, pcf, pc_config, precomp_bases.get());
      store_inputs<affine_t>(precomp_bases.get(), N * pcf, precomp_fname);
    }

    // int test_size = 10000;
    // std::cout << "NUm additions:\t" << test_size << '\n';
    // scalar_t* a = new scalar_t[test_size];
    // scalar_t* b = new scalar_t[test_size];
    // scalar_t* apb = new scalar_t[test_size];
    // {
    //   scalar_t* bases_p = scalars.get();
    //   for (int i = 0; i < test_size; i++)
    //   {
    //     a[i] = bases_p[i];
    //     b[i] = bases_p[i + test_size];
    //     apb[i] = scalar_t::zero();
    //   }
    // }

    START_TIMER(MSM_sync)
    for (int i = 0; i < iters; ++i) {
      msm(scalars.get(), precomp_bases.get(), N, config, result);
    }
    // for (int i = 0; i < test_size; i++)
    // {
    //   apb[i] = a[i] * b[i];
    // }
    END_TIMER(MSM_sync, msg, measure);
  };

  // run("CPU", &result_cpu_dbl_n_add, "CPU msm", false /*=measure*/, 1 /*=iters*/); // warmup
  int iters = 1;
  run("CPU", &result_cpu, "CPU msm", VERBOSE /*=measure*/, iters /*=iters*/);
  run("CPU_REF", &result_cpu_ref, "CPU_REF msm", VERBOSE /*=measure*/, iters /*=iters*/);

  std::cout << projective_t::to_affine(result_cpu) << std::endl;
  std::cout << projective_t::to_affine(result_cpu_ref) << std::endl;
  ASSERT_EQ(result_cpu, result_cpu_ref);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}