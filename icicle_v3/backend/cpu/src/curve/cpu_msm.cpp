
#include "icicle/backend/msm_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"

// #define NDEBUG
#include <cassert> // TODO remove

using namespace curve_config;
using namespace icicle;

// TODO ask for help about memory management before / at C.R.
// COMMENT maybe switch to 1d array?
uint32_t* msm_bucket_coeffs(
  const scalar_t* scalars,
  const unsigned int msm_size,
  const unsigned int c,
  const unsigned int num_windows,
  const unsigned int pcf)
{
  /**
   * Split msm scalars to c-wide coefficients for use in the bucket method
   * @param scalars - original scalar array
   * @param msm_size - length of the above array
   * @param c - window-size (inverse to number of buckets)
   * @param num_windows - NBITS/c
   * @param coefficients - output array of the decomposed scalar
   * @return status of function success / failure in the case of invalid arguments
  */
  uint32_t* coefficients = new uint32_t[msm_size*pcf*num_windows];
  // std::fill_n(coefficients, msm_size*pcf*num_windows/2, 0); // TODO zero init is required only once / when chaging config (c, pcf)

  uint32_t half_val = 1 << (c - 1);
  for (int i = 0; i < msm_size; i++)
  {
    uint32_t curr_coeff = scalars[i].get_scalar_digit(0, c);
    for (int j = 0; j < pcf; j++)  
    {
      for (int w = 0; w < num_windows; w++)
      {
        if (curr_coeff <= half_val)
        {
          coefficients[(msm_size*j + i)*num_windows + w] = curr_coeff;
          curr_coeff = scalars[i].get_scalar_digit(num_windows*j + w + 1, c);
        }
        else if (curr_coeff > half_val)
        {
          coefficients[(msm_size*j + i)*num_windows + w] = -curr_coeff;
          curr_coeff = scalars[i].get_scalar_digit(num_windows*j + w + 1, c) + 1;
        }        
      }
    }
  }
  return coefficients;
}

template <typename P>
P* msm_bucket_accumulator(
  const scalar_t* scalars,
  const affine_t* bases,
  const unsigned int c,
  const unsigned int num_windows,
  const unsigned int msm_size,
  const unsigned int pcf)
{
  /**
   * Accumulate into the different buckets
   * @param scalars - original scalars given from the msm result
   * @param bases - point bases to add
   * @param c - width of windows to split scalars above
   * @param msm_size - number of scalars to add
   * @param buckets - points array containing all buckets
  */
  uint32_t* coefficients = msm_bucket_coeffs(scalars, msm_size, c, num_windows, pcf);

  uint32_t num_buckets = 1<<(c-1);
  P* buckets = new P[num_windows*num_buckets];
  // std::fill_n(buckets, num_buckets*num_windows, P::zero());

  uint32_t coeff_bit_mask = (1 << (c - 1)) - 1;
  for (int i = 0; i < msm_size; i++)
  {
    for (int j = 0; j < pcf; j++)
    {
      for (int w = 0; w < num_windows; w++)
      {
        int coeff = coefficients[(msm_size*j + i)*num_windows + w];
        if (coeff != 0)
        {
          int b_index;
          if (coeff > 0) 
          {
            b_index = num_buckets*w + (coeff & coeff_bit_mask);
            buckets[b_index] = P::is_zero(buckets[b_index])?  P::from_affine(bases[msm_size*j + i]) :
                                                              buckets[b_index] + bases[msm_size*j + i];
          }
          else
          {
            b_index = num_buckets*w + ((-coeff) & coeff_bit_mask);
            buckets[b_index] = P::is_zero(buckets[b_index])?  P::from_affine(bases[msm_size*j + i]) :
                                                              buckets[b_index] + bases[msm_size*j + i];
          }
          
        }
      }
    }
  }
  // TODO memory management
  delete[] coefficients;

  return buckets;
}

template <typename P>
P* msm_window_sum(
  P* buckets,
  const unsigned int c,
  const unsigned int num_windows)
{
  uint32_t num_buckets = 1<<c; // NOTE implicitly assuming that c<32

  P* window_sums = new P[num_windows];

  for (int w = 0; w < num_windows; w++)
  {
    window_sums[w] = P::copy(buckets[num_buckets*(w+1) - 1]);
    P partial_sum = P::copy(buckets[num_buckets*(w+1) - 1]);

    for (int i = num_buckets-2; i > 0; i--)
    {
      if (!P::is_zero(buckets[num_buckets*w +i])) partial_sum = partial_sum + buckets[num_buckets*w +i];
      if (!P::is_zero(partial_sum)) window_sums[w] = window_sums[w] + partial_sum;
    }
  }
  return window_sums;
}

template <typename P>
P msm_final_sum(
  P* window_sums,
  const unsigned int c,
  const unsigned int num_windows)
{
  P result = window_sums[num_windows - 1];
  for (int w = num_windows - 2; w >= 0; w--)
  {
    if (P::is_zero(result)){
      if (!P::is_zero(window_sums[w])) result = P::copy(window_sums[w]);
    }
    else
    {
      for (int dbl = 0; dbl < c; dbl++)
      {
        result = P::dbl(result);
      }
      if (!P::is_zero(window_sums[w])) result = result + window_sums[w];
    }
  }
  return result;
}

template <typename P>
void msm_delete_arrays(
  P* buckets,
  P* windows,
  const unsigned int num_windows)
{
  // TODO memory management
  delete[] buckets;
  delete[] windows;
}

eIcicleError not_supported(const MSMConfig& c)
{
  /**
   * Check config for tests that are currently not supported
  */
  if (c.batch_size > 1) return eIcicleError::INVALID_ARGUMENT; // TODO add support
  if (c.are_scalars_on_device | c.are_points_on_device | c.are_results_on_device) return eIcicleError::INVALID_DEVICE; // COMMENT maybe requires policy change given the possibility of multiple devices on one machine
  if (c.are_scalars_montgomery_form | c.are_points_montgomery_form) return eIcicleError::INVALID_ARGUMENT; // TODO add support
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
  // TODO remove at the end
  if (not_supported(config) != eIcicleError::SUCCESS) return not_supported(config);

  const unsigned int c = config.ext.get<int>("c"); // TODO integrate into msm config
  const unsigned int pcf = config.precompute_factor;
  const int num_windows = ((scalar_t::NBITS-1) / (pcf * c)) + 1;

  P* buckets = msm_bucket_accumulator<P>(scalars, bases, c, num_windows, msm_size, pcf);
  P* window_sums = msm_window_sum<P>(buckets, c, num_windows);
  P res = msm_final_sum<P>(window_sums, c, num_windows);

  results[0] = res;
  msm_delete_arrays(buckets, window_sums, num_windows);
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
    res = res + P::from_affine(bases[i]) * scalars[i];
  }
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
  const unsigned int c = config.ext.get<int>("c");
  const unsigned int num_windows_no_precomp = (scalar_t::NBITS - 1) / c + 1;
  const unsigned int shift = c * ((num_windows_no_precomp - 1) / precompute_factor + 1);

  for (int i = 0; i < nof_bases; i++)
  {
    projective_t point = projective_t::from_affine(input_bases[i]);
    output_bases[i] = input_bases[i]; // COMMENT Should I copy? (not by reference)
    for (int j = 1; j < precompute_factor; j++)
    {
      for (int k = 0; k < shift; k++)
      {
        point = projective_t::dbl(point);
      }
      output_bases[nof_bases*j + i] = projective_t::to_affine(point);
    }
  }
  return eIcicleError::SUCCESS;
}

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU", cpu_msm<projective_t>);

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU_REF", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_ref<projective_t>);
