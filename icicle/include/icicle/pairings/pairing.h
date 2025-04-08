#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

namespace icicle {
  /**
   * @brief Computes a cryptographic pairing
   *
   * @tparam A Type for first group element (typically affine point in G1)
   * @tparam A2 Type for second group element (typically affine point in G2)
   * @tparam Pairing Pairing configuration type containing field definitions and implementation details
   * @tparam TargetField Target field type for pairing result (typically a field extension)
   * @param p First input group element
   * @param q Second input group element
   * @param output Pointer to store the pairing result in the target field
   * @return eIcicleError Error code indicating success or failure
   *
   * The pairing is a bilinear map e: G1 × G2 → GT, where:
   * - G1, G2 are elliptic curve groups
   * - GT is a multiplicative subgroup of a field extension
   * - The map preserves the bilinear property: e(aP, bQ) = e(P,Q)^(ab)
   */
  template <typename A, typename A2, typename PairingConfig, typename TargetField = typename PairingConfig::TargetField>
  eIcicleError pairing(const A& p, const A2& q, TargetField* output);
} // namespace icicle