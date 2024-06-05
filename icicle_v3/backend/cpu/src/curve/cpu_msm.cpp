
#include "icicle/backend/msm_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

uint32_t** msm_bucket_coeffs(
  const scalar_t* scalars,
  const unsigned int msm_size,
  const unsigned int c,
  const unsigned int num_windows)
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
  // TODO add check that c divides NBITS
  uint32_t** coefficients = new uint32_t*[msm_size];
  for (int i = 0; i < msm_size; i++)
  {
    coefficients[i] = new uint32_t[num_windows];
    for (int w = 0; w < num_windows; w++)
    {
      coefficients[i][w] = scalars[i].get_scalar_digit(w, c);
    }
  }
  return coefficients;
}

projective_t** msm_bucket_accumulator(
  const scalar_t* scalars,
  const affine_t* bases,
  const unsigned int c,
  const unsigned int num_windows,
  int msm_size)
{
  /**
   * Accumulate into the different buckets
   * @param scalars - original scalars given from the msm result
   * @param bases - point bases to add
   * @param c - width of windows to split scalars above
   * @param msm_size - number of scalars to add
   * @param buckets - points array containing all buckets
  */
  uint32_t** coefficients = msm_bucket_coeffs(scalars, msm_size, c, num_windows);

  uint32_t num_buckets = 1<<c;
  projective_t** buckets = new projective_t*[num_windows];

  for (int w = 0; w < num_windows; w++)
  {
    buckets[w] = new projective_t[num_buckets]; // COMMENT is it ok to define such a potentially large array?
    std::fill_n(buckets[w], num_buckets, projective_t::zero());
  }

  for (int i = 0; i < msm_size; i++)
  {
    for (int w = 0; w < num_windows; w++)
    {
      if (coefficients[i][w] != 0) buckets[w][coefficients[i][w]] = buckets[w][coefficients[i][w]] + bases[i];
    }
  }

  for (int i = 0; i < msm_size; i++)
  {
    delete[] coefficients[i];
  }
  delete[] coefficients;

  return buckets;
}

projective_t* msm_window_sum(
  projective_t** buckets,
  const unsigned int c,
  const unsigned int num_windows)
{
  uint32_t num_buckets = 1<<c; // NOTE implicitly assuming that c<32

  projective_t* window_sums = new projective_t[num_windows];

  for (int w = 0; w < num_windows; w++)
  {
    // window_sums[w] = projective_t::copy(buckets[w][num_buckets - 1]); // COMMENT how do I make it copy by value?
    window_sums[w] = buckets[w][num_buckets - 1];
    projective_t partial_sum = buckets[w][num_buckets - 1];

    for (int i = num_buckets-2; i > 0; i--)
    {
      if (!projective_t::is_zero(buckets[w][i])) partial_sum = partial_sum + buckets[w][i];
      window_sums[w] = window_sums[w] + partial_sum;
    }
  }
  return window_sums;
}

projective_t msm_final_sum(
  projective_t* window_sums,
  const unsigned int c,
  const unsigned int num_windows)
{
  projective_t result = window_sums[num_windows - 1];
  for (int w = num_windows - 2; w >= 0; w--)
  {
    for (int dbl = 0; dbl < c; dbl++)
    {
      result = projective_t::dbl(result);
    }
    result = result + window_sums[w];
  }
  return result;
}

void msm_delete_arrays(
  projective_t** buckets,
  projective_t* windows,
  const unsigned int num_windows)
{
  for (int w = 0; w < num_windows; w++)
  {
    delete[] buckets[w];
  }
  delete[] buckets;
  delete[] windows;
}

// Double and add
eIcicleError cpu_msm(
  const Device& device,
  const scalar_t* scalars, // COMMENT it assumes no negative scalar inputs
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  projective_t* results)
{
  const unsigned int c = 15; // TODO integrate into msm config
  const int num_windows = (scalar_t::NBITS / c) + ((scalar_t::NBITS % c != 0)? 1 : 0); 

  projective_t** buckets = msm_bucket_accumulator(scalars, bases, c, num_windows, msm_size);
  projective_t* window_sums = msm_window_sum(buckets, c, num_windows);
  projective_t res = msm_final_sum(window_sums, c, num_windows);
  // COMMENT do I need to delete the buckets manually or is it handled automatically when the function finishes?
  results[0] = res;
  msm_delete_arrays(buckets, window_sums, num_windows);
  return eIcicleError::SUCCESS;
}

eIcicleError cpu_msm_ref(
  const Device& device,
  const scalar_t* scalars,
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  projective_t* results)
{
  projective_t res = projective_t::zero();
  for (auto i = 0; i < msm_size; ++i) {
    res = res + projective_t::from_affine(bases[i]) * scalars[i];
  }
  return eIcicleError::SUCCESS;
}

template <typename A>
eIcicleError cpu_msm_precompute_bases(
  const Device& device, const A* input_bases, int nof_bases, const MSMConfig& config, A* output_bases)
{
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU", (cpu_msm));

REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU_REF", cpu_msm_precompute_bases<affine_t>);
REGISTER_MSM_BACKEND("CPU_REF", cpu_msm_ref);
