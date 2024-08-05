#include "icicle/errors.h"
#include "icicle/device.h"
#include "icicle/config_extension.h"
#include <random>
#include <cassert>

// #define DUMMY_TYPES
#define DEBUG_PRINTS
#define P_MACRO 1000

class DummyScalar
{
public:
  static constexpr unsigned NBITS = 32;

  unsigned x;
  unsigned p = P_MACRO;
  // unsigned p = 1<<30;

  static DummyScalar zero() { return {0}; }

  static DummyScalar one() { return {1}; }

  static DummyScalar to_montgomery(const DummyScalar& s) { return {s.x}; }

  static DummyScalar from_montgomery(const DummyScalar& s) { return {s.x}; }

  friend std::ostream& operator<<(std::ostream& os, const DummyScalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
  {
    return (x >> (digit_num * digit_width)) & ((1 << digit_width) - 1);
  }

  friend DummyScalar operator+(DummyScalar p1, const DummyScalar& p2) { return {(p1.x + p2.x) % p1.p}; }

  friend DummyScalar operator-(DummyScalar p1, const DummyScalar& p2) { return p1 + neg(p2); }

  friend bool operator==(const DummyScalar& p1, const DummyScalar& p2) { return (p1.x == p2.x); }

  friend bool operator==(const DummyScalar& p1, const unsigned p2) { return (p1.x == p2); }

  static DummyScalar neg(const DummyScalar& scalar) { return {scalar.p - scalar.x}; }
  static DummyScalar rand_host(std::mt19937_64& rand_generator)
  {
    // return {(unsigned)rand() % P_MACRO};
    std::uniform_int_distribution<unsigned> distribution(0, P_MACRO - 1);
    return {distribution(rand_generator)};
  }

  static void rand_host_many(DummyScalar* out, int size, std::mt19937_64& rand_generator)
  {
    for (int i = 0; i < size; i++)
      // out[i] = (i % size < 100) ? rand_host(rand_generator) : out[i - 100];
      out[i] = rand_host(rand_generator);
  }
};

class DummyPoint
{
public:
  DummyScalar x;

  static DummyPoint zero() { return {0}; }

  static DummyPoint one() { return {1}; }

  static DummyPoint to_affine(const DummyPoint& point) { return {point.x}; }

  static DummyPoint from_affine(const DummyPoint& point) { return {point.x}; }

  static DummyPoint to_montgomery(const DummyPoint& point) { return {point.x}; }

  static DummyPoint from_montgomery(const DummyPoint& point) { return {point.x}; }

  static DummyPoint neg(const DummyPoint& point) { return {DummyScalar::neg(point.x)}; }

  static DummyPoint copy(const DummyPoint& point) { return {point.x}; }

  friend DummyPoint operator+(DummyPoint p1, const DummyPoint& p2) { return {p1.x + p2.x}; }

  friend DummyPoint operator-(DummyPoint p1, const DummyPoint& p2) { return {p1.x - p2.x}; }

  static DummyPoint dbl(const DummyPoint& point) { return {point.x + point.x}; }

  // friend  DummyPoint operator-(DummyPoint p1, const DummyPoint& p2) {
  //   return p1 + neg(p2);
  // }

  friend std::ostream& operator<<(std::ostream& os, const DummyPoint& point)
  {
    os << point.x;
    return os;
  }

  friend DummyPoint operator*(DummyScalar scalar, const DummyPoint& point)
  {
    DummyPoint res = zero();
#ifdef CUDA_ARCH
    UNROLL
#endif
    for (int i = 0; i < DummyScalar::NBITS; i++) {
      if (i > 0) { res = res + res; }
      if (scalar.get_scalar_digit(DummyScalar::NBITS - i - 1, 1)) { res = res + point; }
    }
    return res;
  }

  friend bool operator==(const DummyPoint& p1, const DummyPoint& p2) { return (p1.x == p2.x); }

  static bool is_zero(const DummyPoint& point) { return point.x == 0; }

  static DummyPoint rand_host(std::mt19937_64& rand_generator)
  {
    return {(unsigned)rand() % P_MACRO};
    // return {(unsigned)rand()};
  }

  static void rand_host_many(DummyPoint* out, int size, std::mt19937_64& rand_generator)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? to_affine(rand_host(rand_generator)) : out[i - 100];
  }
};

#include "cpu_msm.hpp"
using namespace icicle;

// template <typename T>
// bool read_inputs(T* arr, const int arr_size, const std::string fname)
// {
//   std::ifstream in_file(fname);
//   bool status = in_file.is_open();
//   if (status) {
//     for (int i = 0; i < arr_size; i++) {
//       in_file.read(reinterpret_cast<char*>(&arr[i]), sizeof(T));
//     }
//     in_file.close();
//   }
//   return status;
// }

// template <typename T>
// void store_inputs(T* arr, const int arr_size, const std::string fname)
// {
//   std::ofstream out_file(fname);
//   if (!out_file.is_open()) {
//     std::cerr << "Failed to open " << fname << " for writing.\n";
//     return;
//   }
//   for (int i = 0; i < arr_size; i++) {
//     out_file.write(reinterpret_cast<char*>(&arr[i]), sizeof(T));
//   }
//   out_file.close();
// }

// void get_inputs(affine_t* bases, scalar_t* scalars, const int n) // TODO add precompute factor
// {
//   // Scalars
//   std::string scalar_file = "build/generated_data/scalars_N" + std::to_string(n) + ".dat";
//   if (!read_inputs<scalar_t>(scalars, n, scalar_file)) {
//     std::cout << "Generating scalars.\n";
//     scalar_t::rand_host_many(scalars, n);
//     store_inputs<scalar_t>(scalars, n, scalar_file);
//   }
//   // Bases
//   std::string base_file = "build/generated_data/bases_N" + std::to_string(n) + ".dat";
//   if (!read_inputs<affine_t>(bases, n, base_file)) {
//     std::cout << "Generating bases.\n";
//     projective_t::rand_host_many(bases, n);
//     store_inputs<affine_t>(bases, n, base_file);
//   }
// }

int main()
{
  int seed = 0;
  auto t = Timer("Time till failure");

  while (true) {
    const int logn = 4;
    const int N = 1 << logn;
    auto scalars = std::make_unique<scalar_t[]>(N);
    auto bases = std::make_unique<affine_t[]>(N);

    bool conv_mont = false;

    std::mt19937_64 generator(seed);

    #ifdef DUMMY_TYPES
    scalar_t::rand_host_many(scalars.get(), N, generator);
    projective_t::rand_host_many(bases.get(), N, generator);
    #else
    scalar_t::rand_host_many(scalars.get(), N);
    projective_t::rand_host_many(bases.get(), N);
    #endif
    if (conv_mont) {
      for (int i = 0; i < N; i++)
        bases[i] = affine_t::to_montgomery(bases[i]);
    }
    projective_t result_cpu{};
    projective_t result_cpu_ref{};

    auto run = [&](const char* dev_type, projective_t* result, const char* msg, bool measure, int iters, auto cpu_msm) {
      const int log_p = 0;
      const int c = std::max(logn, 8) - 1;
      const int pcf = 1 << log_p;

      int hw_threads = std::thread::hardware_concurrency();
      if (hw_threads <= 0) { std::cout << "Unable to detect number of hardware supported threads - fixing it to 1\n"; }
      // const int n_threads = (hw_threads > 1)? hw_threads-2 : 1;
      const int n_threads = 8;

      const int tasks_per_thread = 4;

      auto config = default_msm_config();
      ConfigExtension ext;
      ext.set("c", c);
      ext.set("n_threads", n_threads);
      ext.set("tasks_per_thread", tasks_per_thread);
      
      config.ext = &ext;
      config.precompute_factor = pcf;
      config.are_scalars_montgomery_form = false;
      config.are_points_montgomery_form = conv_mont;

      // auto precomp_bases = std::make_unique<scalar_t[]>(N * pcf);
      // cpu_msm_precompute_bases<scalar_t>(Device(), bases.get(), N, pcf, config, precomp_bases.get());
      // START_TIMER(MSM_sync)
      for (int i = 0; i < iters; ++i) {
        // TODO real test
        // msm_precompute_bases(bases.get(), N, 1, default_msm_pre_compute_config(), bases.get());
        cpu_msm("CPU", scalars.get(), bases.get(), N, config, result);
      }
      // END_TIMER(MSM_sync, msg, measure);
    };

    // run("CPU", &result_cpu_dbl_n_add, "CPU msm", false /*=measure*/, 1 /*=iters*/); // warmup
    run("CPU", &result_cpu, "CPU msm", true /*=measure*/, 1 /*=iters*/, cpu_msm<projective_t>);
    run("CPU_REF", &result_cpu_ref, "CPU_REF msm", true /*=measure*/, 1 /*=iters*/, cpu_msm_single_thread<projective_t>);
    std::cout << projective_t::to_affine(result_cpu) << std::endl;
    std::cout << projective_t::to_affine(result_cpu_ref) << std::endl;
    std::cout << "Seed is: " << seed << '\n';
    assert(result_cpu == result_cpu_ref);
  }

  return 0;
}
