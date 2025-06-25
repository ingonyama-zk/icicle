#include "utils.h"

PolyRing zero()
{
  PolyRing z;
  for (size_t i = 0; i < PolyRing::d; i++) {
    z.values[i] = Zq::zero();
  }
  return z;
}

eIcicleError scale_diagonal_with_mask(
  const Tq* matrix,  // Input n×n matrix (row-major order)
  Zq scaling_factor, // Factor to scale diagonal by
  size_t n,          // Matrix dimension (n×n)
  const VecOpsConfig& config,
  Tq* output) // Output matrix
{
  // Create scaling mask: diagonal = scaling_factor, off-diagonal = 1
  size_t d = Tq::d;
  std::vector<Zq> mask(n * d * n * d, Zq::from(1));

  // Set diagonal elements to scaling factor
  for (uint64_t i = 0; i < n; i++) {
    std::fill(&mask[i * n * d + i], &mask[i * n * d + i + d], scaling_factor);
  }

  // Use vector_mul to apply the mask
  return vector_mul(
    reinterpret_cast<const Zq*>(matrix), mask.data(), n * d * n * d, config, reinterpret_cast<Zq*>(output));
}
