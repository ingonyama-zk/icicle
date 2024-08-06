#include "icicle/errors.h"
#include "icicle/config_extension.h"
using namespace icicle;

struct MSMConfig {
  int nof_bases; /**< Number of bases in the MSM for batched MSM. Set to 0 if all MSMs use the same bases or set to
                  * 'batch X #scalars' otherwise.  Default value: 0 (that is reuse bases for all batch elements). */
  int precompute_factor;            /**< The number of extra points to pre-compute for each point. See the
                                     *   [precompute_msm_bases](@ref precompute_msm_bases) function, `precompute_factor` passed
                                     *   there needs to be equal to the one used here. Larger values decrease the
                                     *   number of computations to make, on-line memory footprint, but increase the static
                                     *   memory footprint. Default value: 1 (i.e. don't pre-compute). */
  int batch_size;                   /**< The number of MSMs to compute. Default value: 1. */
  bool are_scalars_on_device;       /**< True if scalars are on device and false if they're on host. Default value:
                                     *   false. */
  bool are_scalars_montgomery_form; /**< True if scalars are in Montgomery form and false otherwise. Default value:
                                     *   true. */
  bool are_points_on_device; /**< True if points are on device and false if they're on host. Default value: false. */
  bool are_points_montgomery_form; /**< True if coordinates of points are in Montgomery form and false otherwise.
                                    *   Default value: true. */
  bool are_results_on_device; /**< True if the results should be on device and false if they should be on host. If set
                               *   to false, `is_async` won't take effect because a synchronization is needed to
                               *   transfer results to the host. Default value: false. */
  bool is_async;              /**< Whether to run the MSM asynchronously. If set to true, the MSM function will be
                               *   non-blocking and you'd need to synchronize it explicitly by running
                               *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the MSM
                               *   function will block the current CPU thread. */

  ConfigExtension ext; /** backend specific extensions*/
};

struct MsmPreComputeConfig {
  bool is_input_on_device;
  bool is_output_on_device;
  bool is_async;

  ConfigExtension ext; /** backend specific extensions*/
};

#define P_MACRO 1000

struct Device {
};

class Dummy_Scalar
{
public:
  static constexpr unsigned NBITS = 32;

  unsigned x;
  unsigned p = P_MACRO;
  // unsigned p = 1<<30;

  static Dummy_Scalar zero() { return {0}; }

  static Dummy_Scalar one() { return {1}; }

  static Dummy_Scalar to_montgomery(const Dummy_Scalar& s) { return {s.x}; }

  static Dummy_Scalar from_montgomery(const Dummy_Scalar& s) { return {s.x}; }

  friend std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
  {
    return (x >> (digit_num * digit_width)) & ((1 << digit_width) - 1);
  }

  friend Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2) { return {(p1.x + p2.x) % p1.p}; }

  friend Dummy_Scalar operator-(Dummy_Scalar p1, const Dummy_Scalar& p2) { return p1 + neg(p2); }

  friend bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) { return (p1.x == p2.x); }

  friend bool operator==(const Dummy_Scalar& p1, const unsigned p2) { return (p1.x == p2); }

  static Dummy_Scalar neg(const Dummy_Scalar& scalar) { return {scalar.p - scalar.x}; }
  static Dummy_Scalar rand_host(std::mt19937_64& rand_generator)
  {
    // return {(unsigned)rand() % P_MACRO};
    std::uniform_int_distribution<unsigned> distribution(0, P_MACRO - 1);
    return {distribution(rand_generator)};
  }

  static void rand_host_many(Dummy_Scalar* out, int size, std::mt19937_64& rand_generator)
  {
    for (int i = 0; i < size; i++)
      // out[i] = (i % size < 100) ? rand_host(rand_generator) : out[i - 100];
      out[i] = rand_host(rand_generator);
  }
};

class Dummy_Projective
{
public:
  Dummy_Scalar x;

  static Dummy_Projective zero() { return {0}; }

  static Dummy_Projective one() { return {1}; }

  static Dummy_Projective to_affine(const Dummy_Projective& point) { return {point.x}; }

  static Dummy_Projective from_affine(const Dummy_Projective& point) { return {point.x}; }

  static Dummy_Projective to_montgomery(const Dummy_Projective& point) { return {point.x}; }

  static Dummy_Projective from_montgomery(const Dummy_Projective& point) { return {point.x}; }

  static Dummy_Projective neg(const Dummy_Projective& point) { return {Dummy_Scalar::neg(point.x)}; }

  static Dummy_Projective copy(const Dummy_Projective& point) { return {point.x}; }

  friend Dummy_Projective operator+(Dummy_Projective p1, const Dummy_Projective& p2) { return {p1.x + p2.x}; }

  friend Dummy_Projective operator-(Dummy_Projective p1, const Dummy_Projective& p2) { return {p1.x - p2.x}; }

  static Dummy_Projective dbl(const Dummy_Projective& point) { return {point.x + point.x}; }

  // friend  Dummy_Projective operator-(Dummy_Projective p1, const Dummy_Projective& p2) {
  //   return p1 + neg(p2);
  // }

  friend std::ostream& operator<<(std::ostream& os, const Dummy_Projective& point)
  {
    os << point.x;
    return os;
  }

  friend Dummy_Projective operator*(Dummy_Scalar scalar, const Dummy_Projective& point)
  {
    Dummy_Projective res = zero();
#ifdef CUDA_ARCH
    UNROLL
#endif
    for (int i = 0; i < Dummy_Scalar::NBITS; i++) {
      if (i > 0) { res = res + res; }
      if (scalar.get_scalar_digit(Dummy_Scalar::NBITS - i - 1, 1)) { res = res + point; }
    }
    return res;
  }

  friend bool operator==(const Dummy_Projective& p1, const Dummy_Projective& p2) { return (p1.x == p2.x); }

  static bool is_zero(const Dummy_Projective& point) { return point.x == 0; }

  static Dummy_Projective rand_host(std::mt19937_64& rand_generator)
  {
    return {(unsigned)rand() % P_MACRO};
    // return {(unsigned)rand()};
  }

  static void rand_host_many_affine(Dummy_Projective* out, int size, std::mt19937_64& rand_generator)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? to_affine(rand_host(rand_generator)) : out[i - 100];
  }
};