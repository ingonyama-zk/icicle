#include "icicle/errors.h"
#include "icicle/device.h"
#include "icicle/config_extension.h"
#include "icicle/curves/curve_config.h"
#include <random>
#include <cassert>
// #include "timer.hpp"

#define DUMMY_TYPES
// #define DEBUG_PRINTS
#define P_MACRO 1000

class DummyScalar
{
public:
  static constexpr unsigned NBITS = 10;

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
    // if (digit_num * digit_width > 10) { std::cout << "Overflow output(" << digit_num << '*' << digit_width << "):\t"
    // << ((x >> (digit_num * digit_width)) & ((1 << digit_width) - 1)) << "\t(" << this << ")\n";} else { std::cout <<
    // "Reg output(" << digit_num << "):\t" << ((x >> (digit_num * digit_width)) & ((1 << digit_width) - 1)) << '\n';}
    return ((x % p) >> (digit_num * digit_width)) & ((1 << digit_width) - 1);
  }

  friend DummyScalar operator+(DummyScalar p1, const DummyScalar& p2) { return {(p1.x + p2.x) % p1.p}; }

  friend DummyScalar operator-(DummyScalar p1, const DummyScalar& p2) { return p1 + neg(p2); }

  friend bool operator==(const DummyScalar& p1, const DummyScalar& p2) { return (p1.x == p2.x); }

  friend bool operator==(const DummyScalar& p1, const unsigned p2) { return (p1.x == p2); }

  static DummyScalar neg(const DummyScalar& scalar) { return {scalar.p - scalar.x}; }
  static DummyScalar rand_host() { return {(unsigned)rand() % P_MACRO}; }

  static void rand_host_many(DummyScalar* out, int size)
  {
    for (int i = 0; i < size; i++)
      // out[i] = (i % size < 100) ? rand_host(rand_generator) : out[i - 100];
      out[i] = rand_host();
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

  static DummyPoint rand_host() { return {(unsigned)rand() % P_MACRO}; }

  static void rand_host_many(DummyPoint* out, int size)
  {
    for (int i = 0; i < size; i++)
      // out[i] = (i % size < 100) ? to_affine(rand_host()) : out[i - 100];
      out[i] = to_affine(rand_host());
  }
};

#ifdef DUMMY_TYPES // for testing
using A = DummyPoint;
using P = DummyPoint;
using scalar_t = DummyScalar;
#else
using A = curve_config::g2_affine_t;
using P = curve_config::g2_projective_t;
using scalar_t = curve_config::scalar_t;
#endif

#include "cpu_msm.hpp"
using namespace icicle;

template <typename Point>
std::vector<Point> msm_bucket_accumulator(
  const scalar_t* scalars,
  const A* bases,
  const unsigned int c,
  const unsigned int num_bms,
  const unsigned int msm_size,
  const unsigned int precompute_factor,
  const bool is_s_mont,
  const bool is_b_mont)
{
  /**
   * Accumulate into the different bkts
   * @param scalars - original scalars given from the msm result
   * @param bases - point bases to add
   * @param c - address width of bucket modules to split scalars above
   * @param msm_size - number of scalars to add
   * @param is_s_mont - flag indicating input scalars are in Montgomery form
   * @param is_b_mont - flag indicating input bases are in Montgomery form
   * @return bkts - points array containing all bkts
   */
  auto t = Timer("P1:bucket-accumulator");
  uint32_t num_bkts = 1 << (c - 1);
  std::vector<Point> bkts(num_bms * num_bkts, Point::zero());
  uint32_t coeff_bit_mask = num_bkts - 1;
  const int num_windows_m1 = (scalar_t::NBITS - 1) / c;
  int carry;

#ifdef DEBUG_PRINTS
  std::string trace_fname = "trace_bucket_single.txt";
  std::ofstream trace_f(trace_fname);
  if (!trace_f.good()) { std::cout << "ERROR: can't open file:\t" << trace_fname << std::endl; }
#endif
  for (int i = 0; i < msm_size; i++) {
    carry = 0;
    scalar_t scalar = is_s_mont ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    bool negate_p_and_s = scalar.get_scalar_digit(scalar_t::NBITS - 1, 1) > 0;
    if (negate_p_and_s) scalar = scalar_t::neg(scalar);
    for (int j = 0; j < precompute_factor; j++) {
      A point = is_b_mont ? A::from_montgomery(bases[precompute_factor * i + j]) : bases[precompute_factor * i + j];
      if (negate_p_and_s) point = A::neg(point);
      for (int k = 0; k < num_bms; k++) {
        // In case precompute_factor*c exceeds the scalar width
        if (num_bms * j + k > num_windows_m1) { break; }

        uint32_t curr_coeff = scalar.get_scalar_digit(num_bms * j + k, c) + carry;
        if ((curr_coeff & ((1 << c) - 1)) != 0) {
          if (curr_coeff < num_bkts) {
#ifdef DEBUG_PRINTS
            int bkt_idx = num_bkts * k + curr_coeff;
            if (Point::is_zero(bkts[bkt_idx])) {
              trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << point.x << '\n';
            } else {
              trace_f << '#' << bkt_idx << ":\tRead for addition:\t" << Point::to_affine(bkts[bkt_idx]).x
                      << "\t(With new point:\t" << point.x << " = " << Point::to_affine(bkts[bkt_idx] + point).x
                      << ")\n";
              trace_f << '#' << bkt_idx << ":\tWrite (res) free cell:\t" << Point::to_affine(bkts[bkt_idx] + point).x
                      << '\n';
            }
#endif

            bkts[num_bkts * k + curr_coeff] = Point::is_zero(bkts[num_bkts * k + curr_coeff])
                                                ? Point::from_affine(point)
                                                : bkts[num_bkts * k + curr_coeff] + point;
            carry = 0;
          } else {
#ifdef DEBUG_PRINTS
            int bkt_idx = num_bkts * k + ((-curr_coeff) & coeff_bit_mask);
            if (Point::is_zero(bkts[bkt_idx])) {
              trace_f << '#' << bkt_idx << ":\tWrite free cell:\t" << A::neg(point).x << '\n';
            } else {
              trace_f << '#' << bkt_idx << ":\tRead for subtraction:\t" << Point::to_affine(bkts[bkt_idx]).x
                      << "\t(With new point:\t" << point.x << " = " << Point::to_affine(bkts[bkt_idx] - point).x
                      << ")\n";
              trace_f << '#' << bkt_idx << ":\tWrite (res) free cell:\t" << Point::to_affine(bkts[bkt_idx] - point).x
                      << '\n';
            }
#endif

            bkts[num_bkts * k + ((-curr_coeff) & coeff_bit_mask)] =
              Point::is_zero(bkts[num_bkts * k + ((-curr_coeff) & coeff_bit_mask)])
                ? Point::neg(Point::from_affine(point))
                : bkts[num_bkts * k + ((-curr_coeff) & coeff_bit_mask)] - point;
            carry = 1;
          }
        } else {
          carry = curr_coeff >> c; // Edge case for coeff = 1 << c
        }
      }
    }
  }
#ifdef DEBUG_PRINTS
  trace_f.close();
  std::string b_fname = "buckets_single.txt";
  std::ofstream bkts_f_single(b_fname);
  if (!bkts_f_single.good()) { std::cout << "ERROR: can't open file:\t" << b_fname << std::endl; }
  for (int i = 0; i < num_bms; i++)
    for (int j = 0; j < num_bkts; j++)
      bkts_f_single << '(' << i << ',' << j << "):\t" << Point::to_affine(bkts[num_bkts * i + j]).x << '\n';
  bkts_f_single.close();
#endif
  return bkts;
}

// They will only be kept in a test file when phase 2 and 3 sums are implemented in msm class
/**
 * @brief Single threaded implementation of BM sum (phase 2) of MSM.
 * @param bkts - vector of buckets given from the previous phase 1 function.
 * @param c - Pipenger's constant. Used to calculate the bucket size of a BM.
 * @param num_bms - number of BMs.
 * @return - vector<Point> of size <num_bms>, containing each of the bms sums.
 */
template <typename Point>
std::vector<Point> msm_bm_sum(std::vector<Point>& bkts, const unsigned int c, const unsigned int num_bms)
{
  auto t = Timer("P2:bucket-module-sum");
  uint32_t num_bkts = 1 << (c - 1);

  std::vector<Point> bm_sums(num_bms);

  // Calculate the weighted "triangle" sum by using two sums in series:
  // A partial sum holding the current line sum, and a total sum holding the sums of the lines (The triangle).
  for (int k = 0; k < num_bms; k++) {
    bm_sums[k] = bkts[num_bkts * k];        // Start with bucket zero that holds the weight @num_bkts
    Point partial_sum = bkts[num_bkts * k]; // And the triangle and the line start with the same value

    for (int i = num_bkts - 1; i > 0; i--) {
      partial_sum = partial_sum + bkts[num_bkts * k + i];
      bm_sums[k] = bm_sums[k] + partial_sum;
    }
  }

  return bm_sums;
}

/**
 * @brief Single threaded implementation of the final accumulator (phase 3) of MSM.
 * @param bm_sums - vector of Points which are the BM sums from the previous phase 2 function.
 * @param c - Pipenger's constant. Used to calculate the bucket size of a BM.
 * @param num_bms - number of BMs.
 * @return - Point, MSM result.
 */
template <typename Point>
Point msm_final_sum(std::vector<Point>& bm_sums, const unsigned int c, const unsigned int num_bms)
{
  /**
   * Sum the bucket module sums to get the final result.
   * @param bm_sums - point array containing bucket module sums.
   * @param c - bucket module width / shift between subsequent bkts.
   * @param is_b_mont - flag indicating input bases are in Montgomery form.
   * @return result - MSM calculation.
   */
  auto t = Timer("P3:final-accumulator");
  Point result = bm_sums[num_bms - 1];
  for (int k = num_bms - 2; k >= 0; k--) {
    // Check if the current value is not zero before doing the c doubling below.
    if (Point::is_zero(result)) {
      result = bm_sums[k];
    } else {
      // Every bm sum is 2^c times larger than the subsequent less significant bm-sum.
      // Double the current result c times before adding the next bm-sum.
      for (int dbl = 0; dbl < c; dbl++) {
        result = Point::dbl(result);
      }
      result = result + bm_sums[k];
    }
  }
  return result;
}

// Pipenger
template <typename Point>
eIcicleError cpu_msm_single_thread(
  const Device& device, const scalar_t* scalars, const A* bases, int msm_size, const MSMConfig& config, Point* results)
{
  auto t = Timer("total-msm-single-threaded");

  const unsigned int c = config.ext->get<int>("c");
  const unsigned int precompute_factor = config.precompute_factor;
  const int num_bms = ((scalar_t::NBITS - 1) / (precompute_factor * c)) + 1;
  std::cout << "\n\nnum_bms = " << num_bms << ", c=" << c << ", precomp=" << precompute_factor << "\n\n\n";

  for (int i = 0; i < config.batch_size; i++) {
    std::vector<Point> bkts = msm_bucket_accumulator<Point>(
      &scalars[msm_size * i], bases, c, num_bms, msm_size, precompute_factor, config.are_scalars_montgomery_form,
      config.are_points_montgomery_form);
    std::vector<Point> bm_sums = msm_bm_sum<Point>(bkts, c, num_bms);
    Point res = msm_final_sum<Point>(bm_sums, c, num_bms);
    results[i] = res;
  }

  return eIcicleError::SUCCESS;
}

// Most naive implementation as backup
template <typename Point>
eIcicleError cpu_msm_ref(
  const Device& device, const scalar_t* scalars, const A* bases, int msm_size, const MSMConfig& config, Point* results)
{
  const unsigned int precompute_factor = config.precompute_factor;
  Point res = Point::zero();
  for (auto i = 0; i < msm_size; ++i) {
    scalar_t scalar = config.are_scalars_montgomery_form ? scalar_t::from_montgomery(scalars[i]) : scalars[i];
    A point = config.are_points_montgomery_form ? A::from_montgomery(bases[precompute_factor * i])
                                                : bases[precompute_factor * i];
    res = res + scalar * Point::from_affine(point);
  }
  // results[0] = config.are_points_montgomery_form? Point::to_montgomery(res) : res;
  results[0] = res;
  return eIcicleError::SUCCESS;
}

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

void get_inputs(A* bases, scalar_t* scalars, const int n, const int batch_size, bool gen_new = false)
{
  // Scalars
  std::string scalar_file = "build/generated_data/scalars_N" + std::to_string(n * batch_size) + ".dat";
  if (gen_new || !read_inputs<scalar_t>(scalars, n * batch_size, scalar_file)) {
    std::cout << "Generating scalars.\n";
    scalar_t::rand_host_many(scalars, n * batch_size);
    store_inputs<scalar_t>(scalars, n * batch_size, scalar_file);
  }
  // Bases
  std::string base_file = "build/generated_data/bases_N" + std::to_string(n) + ".dat";
  if (gen_new || !read_inputs<A>(bases, n, base_file)) {
    std::cout << "Generating bases.\n";
    P::rand_host_many(bases, n);
    store_inputs<A>(bases, n, base_file);
  }
}

int main()
{
  while (true) {
    // MSM config
    const int logn = 17;
    const int N = 1 << logn;
    const int log_p = 3;
    const int batch_size = 3;
    bool conv_mont = false;

    bool gen_new = true;

    auto scalars = std::make_unique<scalar_t[]>(N * batch_size);
    auto bases = std::make_unique<A[]>(N);
    get_inputs(bases.get(), scalars.get(), N, batch_size, gen_new);

    if (conv_mont) {
      std::cout << "Convertiting inputs to Montgomery form\n";
      for (int i = 0; i < N; i++)
        bases[i] = A::to_montgomery(bases[i]);
    }
    P* result_cpu = new P[batch_size];
    P* result_cpu_ref = new P[batch_size];
    std::fill_n(result_cpu, batch_size, P::zero());
    std::fill_n(result_cpu_ref, batch_size, P::zero());

    auto run = [&](const char* dev_type, P* result, const char* msg, bool measure, int iters, auto msm_func) {
      const int c = std::max(logn, 8) - 1;
      // const int c = 4;
      std::cout << "c:\t" << c << '\n';
      const int pcf = 1 << log_p;

      int hw_threads = std::thread::hardware_concurrency();
      if (hw_threads <= 0) { std::cout << "Unable to detect number of hardware supported threads - fixing it to 1\n"; }
      const int n_threads = (hw_threads > 1) ? hw_threads - 1 : 1;
      // const int n_threads = 1;
      std::cout << "Num threads: " << n_threads << '\n';

      const int tasks_per_thread = 4;

      MSMConfig config = default_msm_config();
      ConfigExtension ext;
      ext.set("c", c);
      ext.set("n_threads", n_threads);

      config.c = c;
      config.ext = &ext;
      config.precompute_factor = pcf;
      config.are_scalars_montgomery_form = false;
      config.are_points_montgomery_form = conv_mont;
      config.batch_size = batch_size;

      auto precomp_bases = std::make_unique<A[]>(N * pcf);
      std::string precomp_fname =
        "build/generated_data/precomp_N" + std::to_string(N) + "_pcf" + std::to_string(pcf) + ".dat";
      if (gen_new || !read_inputs<A>(precomp_bases.get(), N * pcf, precomp_fname)) {
        std::cout << "Precomputing bases.";
        cpu_msm_precompute_bases<A, P>("CPU", bases.get(), N, config, precomp_bases.get());
        std::cout << " Storing.\n";
        store_inputs<A>(precomp_bases.get(), N * pcf, precomp_fname);
      }
      // START_TIMER(MSM_sync)
      std::cout << "Starting msm (N=" << N << ", pcf=" << pcf << ")\n";
      for (int i = 0; i < iters; ++i) {
        msm_func("CPU", scalars.get(), precomp_bases.get(), N, config, result);
      }
      // END_TIMER(MSM_sync, msg, measure);
    };

    // run("CPU", &result_cpu_dbl_n_add, "CPU msm", false /*=measure*/, 1 /*=iters*/); // warmup
    run("CPU_REF", result_cpu_ref, "CPU_REF msm", true /*=measure*/, 1 /*=iters*/, cpu_msm_single_thread<P>);
    run("CPU", result_cpu, "CPU msm", true /*=measure*/, 1 /*=iters*/, cpu_msm<A, P>);

    for (int i = 0; i < batch_size; i++) {
      std::cout << "Batch no. " << i << ":\n";
      std::cout << "CPU:\t\t" << P::to_affine(result_cpu[i]) << std::endl;
      std::cout << "CPU REF:\t" << P::to_affine(result_cpu_ref[i]) << std::endl;
      assert(result_cpu[i] == result_cpu_ref[i]);
    }

    delete[] result_cpu;
    delete[] result_cpu_ref;
  }

  return 0;
}
