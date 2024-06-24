
#include "icicle/backend/msm_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"

// #define NDEBUG

#include <chrono>
#include <string>
#include <format>

class Timer
{
  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
    std::string fname;
  public:
    Timer(std::string func_name)
    {
      start_point = std::chrono::high_resolution_clock::now();
      fname = func_name;
    }

    ~Timer()
    {
      Stop();
    }

    void Stop()
    {
      auto end_point = std::chrono::high_resolution_clock::now();
      auto start_time = std::chrono::time_point_cast<std::chrono::microseconds>(start_point).time_since_epoch().count();
      auto end_time = std::chrono::time_point_cast<std::chrono::microseconds>(end_point).time_since_epoch().count();
      auto duration = end_time - start_time;

      double dur_s = duration * 0.001;
      std::cout << "Time of " << fname << ":\t" << dur_s << "ms\n";
    }
};

using namespace curve_config;
using namespace icicle;

// TODO ask for help about memory management before / at C.R.
// COMMENT maybe switch to 1d array?
uint32_t* msm_bucket_coeffs(
  const scalar_t* scalars,
  const unsigned int msm_size,
  const unsigned int c,
  const unsigned int num_bms,
  const unsigned int pcf)
{
  /**
   * Split msm scalars to c-wide coefficients for use in the bucket method
   * @param scalars - original scalar array
   * @param msm_size - length of the above array
   * @param c - Bucket module address width (inverse to number of buckets)
   * @param num_bms - NBITS/c
   * @param coefficients - output array of the decomposed scalar
   * @return status of function success / failure in the case of invalid arguments
  */
  uint32_t* coefficients = new uint32_t[msm_size*pcf*num_bms];
  // std::fill_n(coefficients, msm_size*pcf*num_bms/2, 0); // TODO zero init is required only once / when chaging config (c, pcf)

  uint32_t half_val = 1 << (c - 1);
  for (int i = 0; i < msm_size; i++)
  {
    uint32_t curr_coeff = scalars[i].get_scalar_digit(0, c);
    for (int j = 0; j < pcf; j++)  
    {
      for (int k = 0; k < num_bms; k++)
      {
        if (curr_coeff <= half_val)
        {
          coefficients[(msm_size*j + i)*num_bms + k] = curr_coeff;
          curr_coeff = scalars[i].get_scalar_digit(num_bms*j + k + 1, c);
        }
        else
        {
          coefficients[(msm_size*j + i)*num_bms + k] = -curr_coeff;
          curr_coeff = scalars[i].get_scalar_digit(num_bms*j + k + 1, c) + 1;
        }        
      }
    }
  }
  return coefficients;
}

template <typename P>
std::vector<P>* sort_buckets(
  const scalar_t* scalars,
  const affine_t* bases,
  const unsigned int msm_size,
  const unsigned int c,
  const unsigned int num_bms,
  const unsigned int pcf
)
{
  /**
   * Split msm scalars to c-wide coefficients for use in the bucket method
   * @param scalars - original scalar array
   * @param bases - points to be added
   * @param msm_size - length of the scalar array (and 1/pcf of the length of bases)
   * @param c - Bucket module address width (inverse to number of buckets)
   * @param num_bms - NBITS/c
   * @return p2buckets - sorted array of points to be added to each bucket
  */
  uint32_t num_buckets = 1<<(c-1);
  auto p2buckets = new std::vector<P>[num_buckets*num_bms];
  // std::fill_n(coefficients, msm_size*pcf*num_bms/2, 0); // TODO zero init is required only once / when chaging config (c, pcf)

  for (int i = 0; i < msm_size; i++)
  {
    uint32_t curr_coeff = scalars[i].get_scalar_digit(0, c);
    for (int j = 0; j < pcf; j++)  
    {
      for (int k = 0; k < num_bms; k++)
      {
        if (curr_coeff <= num_buckets)
        {
          // p2buckets[k*num_buckets + curr_coeff].push_back(bases[i]);
          curr_coeff = scalars[i].get_scalar_digit(num_bms*j + k + 1, c);
        }
        else
        {
          uint32_t truncated_coeff = (curr_coeff&(num_bms-1));
          // p2buckets[k*num_buckets + truncated_coeff].push_back(-bases[i]);
          curr_coeff = scalars[i].get_scalar_digit(num_bms*j + k + 1, c) + 1;
        }        
      }
    }
  }
  return p2buckets;
}

template <typename P>
P* msm_bucket_accumulator(
  const scalar_t* scalars,
  const affine_t* bases,
  const unsigned int c,
  const unsigned int num_bms,
  const unsigned int msm_size,
  const unsigned int pcf,
  const bool is_s_mont,
  const bool is_b_mont)
{
  /**
   * Accumulate into the different buckets
   * @param scalars - original scalars given from the msm result
   * @param bases - point bases to add
   * @param c - address width of bucket modules to split scalars above
   * @param msm_size - number of scalars to add
   * @param is_s_mont - flag indicating input scalars are in Montgomery form
   * @param is_b_mont - flag indicating input bases are in Montgomery form
   * @return buckets - points array containing all buckets
  */
  auto t = Timer("P1:bucket-accumulator");
  uint32_t num_buckets = 1<<(c-1);
  P* buckets;
  {
    auto t2 = Timer("P1:memory_allocation");
    buckets = new P[num_bms*num_buckets];
  }
  {
    auto t3 = Timer("P1:memory_init");
    std::fill_n(buckets, num_buckets*num_bms, P::zero());
  }
  uint32_t coeff_bit_mask = num_buckets - 1;
  const int num_windows_m1 = (scalar_t::NBITS -1) / c;
  int carry;
  for (int i = 0; i < msm_size; i++)
  {
    carry = 0;
    scalar_t scalar = is_s_mont?  scalar_t::from_montgomery(scalars[i]) : 
                                  scalars[i];
    for (int j = 0; j < pcf; j++)  
    {
      affine_t point = is_b_mont? affine_t::from_montgomery(bases[msm_size*j + i]) : 
                                  bases[msm_size*j + i];
      for (int k = 0; k < num_bms; k++)
      {
        // In case pcf*c exceeds the scalar width
        if (num_bms*j + k > num_windows_m1) { break; }

        uint32_t curr_coeff = scalar.get_scalar_digit(num_bms*j + k, c) + carry;
        if ((curr_coeff & ((1 << c) - 1)) != 0)
        {
          if (curr_coeff < num_buckets)
          {
            buckets[num_buckets*k + curr_coeff] = P::is_zero(buckets[num_buckets*k + curr_coeff])?  P::from_affine(point) :
                                                                                                    buckets[num_buckets*k + curr_coeff] + point;
            carry = 0;
          }
          else
          {
            buckets[num_buckets*k + ((-curr_coeff)&coeff_bit_mask)] = P::is_zero(buckets[num_buckets*k + ((-curr_coeff)&coeff_bit_mask)])?  P::neg(P::from_affine(point)) :
                                                                                                                                      buckets[num_buckets*k + ((-curr_coeff)&coeff_bit_mask)] - point;
            carry = 1;
          }
        }
        else 
        { 
          carry = curr_coeff >> c; // Edge case for coeff = 1 << c
        }
      }
    }
    // COMMENT what happens if the carry is needed in the last BM?
  }
  return buckets;
}

template <typename P>
P* msm_bm_sum(
  P* buckets,
  const unsigned int c,
  const unsigned int num_bms)
{
  /**
   * Sum bucket modules to one point each
   * @param buckets - point array containing all buckets ordered by bucket module
   * @param c - bucket width
   * @param num_bms - number of bucket modules
   * @return bm_sums - point array containing the bucket modules' sums
   */
  auto t = Timer("P2:bucket-module-sum");
  uint32_t num_buckets = 1<<(c-1); // NOTE implicitly assuming that c<32

  P* bm_sums = new P[num_bms];

  for (int k = 0; k < num_bms; k++)
  {
    bm_sums[k] = P::copy(buckets[num_buckets*k]); // Start with bucket zero that holds the weight <num_buckets>
    P partial_sum = P::copy(buckets[num_buckets*k]);

    for (int i = num_buckets-1; i > 0; i--)
    {
      if (!P::is_zero(buckets[num_buckets*k +i])) partial_sum = partial_sum + buckets[num_buckets*k +i];
      if (!P::is_zero(partial_sum)) bm_sums[k] = bm_sums[k] + partial_sum;
    }
  }
  return bm_sums;
}

template <typename P>
P msm_final_sum(
  P* bm_sums,
  const unsigned int c,
  const unsigned int num_bms,
  const bool is_b_mont)
{
  /**
   * Sum the bucket module sums to the final result
   * @param bm_sums - point array containing bucket module sums
   * @param c - bucket module width / shift between subsequent buckets
   * @param is_b_mont - flag indicating input bases are in Montgomery form
   * @return result - msm calculation
   */
  auto t = Timer("P3:final-accumulator");
  P result = bm_sums[num_bms - 1];
  for (int k = num_bms - 2; k >= 0; k--)
  {
    if (P::is_zero(result)){
      if (!P::is_zero(bm_sums[k])) result = P::copy(bm_sums[k]);
    }
    else
    {
      for (int dbl = 0; dbl < c; dbl++)
      {
        result = P::dbl(result);
      }
      if (!P::is_zero(bm_sums[k])) result = result + bm_sums[k];
    }
  }
  // auto result_converted = is_b_mont? P::to_montgomery(result) : result;
  return result;
}

template <typename P>
void msm_delete_arrays(
  P* buckets,
  P* bms,
  const unsigned int num_bms)
{
  // TODO memory management
  delete[] buckets;
  delete[] bms;
}

eIcicleError not_supported(const MSMConfig& c)
{
  /**
   * Check config for tests that are currently not supported
  */
  if (c.batch_size > 1) return eIcicleError::INVALID_ARGUMENT; // TODO add support
  if (c.are_scalars_on_device | c.are_points_on_device | c.are_results_on_device) return eIcicleError::INVALID_DEVICE; // COMMENT maybe requires policy change given the possibility of multiple devices on one machine
  if (c.is_async) return eIcicleError::INVALID_DEVICE; //TODO add support
  // FIXME fill non-implemented features from MSMConfig
  return eIcicleError::SUCCESS;
}

// Pipenger
template <typename P>
eIcicleError cpu_msm(
  const Device& device,
  const scalar_t* scalars, // COMMENT it assumes no negative scalar inputs
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  P* results)
{
  auto t = Timer("total-msm");
  // TODO remove at the end
  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext.get<int>("c"); // TODO calculate instead of param
  const unsigned int pcf = config.precompute_factor;
  const int num_bms = ((scalar_t::NBITS-1) / (pcf * c)) + 1;

  P* buckets = msm_bucket_accumulator<P>(scalars, bases, c, num_bms, msm_size, pcf, config.are_scalars_montgomery_form, config.are_points_montgomery_form);
  P* bm_sums = msm_bm_sum<P>(buckets, c, num_bms);
  P res = msm_final_sum<P>(bm_sums, c, num_bms, config.are_points_montgomery_form);

  results[0] = res;
  msm_delete_arrays(buckets, bm_sums, num_bms);
  return eIcicleError::SUCCESS;
}

template <typename P>
eIcicleError cpu_msm_ref(
  const Device& device,
  const scalar_t* scalars,
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  P* results)
{
  P res = P::zero();
  for (auto i = 0; i < msm_size; ++i) {
    scalar_t scalar = config.are_scalars_montgomery_form? scalar_t::from_montgomery(scalars[i]) : 
                                                          scalars[i];
    affine_t point = config.are_points_montgomery_form? affine_t::from_montgomery(bases[i]) : 
                                                        bases[i];
    res = res + P::from_affine(point) * scalar;
  }
  // results[0] = config.are_points_montgomery_form? P::to_montgomery(res) : res;
  results[0] = res;
  return eIcicleError::SUCCESS;
}

template <typename A>
eIcicleError cpu_msm_precompute_bases(
  const Device& device,
  const A* input_bases,
  int nof_bases,
  int precompute_factor,
  const MsmPreComputeConfig& config,
  A* output_bases) // Pre assigned?
{
  bool is_mont = config.ext.get<bool>("is_mont");
  const unsigned int c = config.ext.get<int>("c");
  const unsigned int num_bms_no_precomp = (scalar_t::NBITS - 1) / c + 1;
  const unsigned int shift = c * ((num_bms_no_precomp - 1) / precompute_factor + 1);

  for (int i = 0; i < nof_bases; i++)
  {
    output_bases[i] = input_bases[i]; // COMMENT Should I copy? (not by reference)
    projective_t point = projective_t::from_affine(is_mont? A::from_montgomery(input_bases[i]) : input_bases[i]);
    for (int j = 1; j < precompute_factor; j++)
    {
      for (int k = 0; k < shift; k++)
      {
        point = projective_t::dbl(point);
      }
      output_bases[nof_bases*j + i] = is_mont? A::to_montgomery(projective_t::to_affine(point)) : projective_t::to_affine(point);
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU", cpu_msm<projective_t>);

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU_REF", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_ref<projective_t>);
