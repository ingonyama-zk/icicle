#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

namespace icicle {
  /**
   * @brief Computes a cryptographic pairing
   *
   * @tparam PairingConfig Pairing configuration type containing field definitions and implementation details
   *
   * @param p First input group element. Typically affine point in G1.
   * @param q Second input group element. Typically affine point in G2.
   * @param output reference to store the pairing result in the target field
   * @return eIcicleError Error code indicating success or failure
   *
   * The pairing is a bilinear map e: G1 × G2 → GT, where:
   * - G1, G2 are elliptic curve groups
   * - GT is a multiplicative subgroup of a field extension
   * - The map preserves the bilinear property: e(aP, bQ) = e(P,Q)^(ab)
   */
  template <typename PairingConfig>
  eIcicleError pairing(
    const typename PairingConfig::G1Affine& p,
    const typename PairingConfig::G2Affine& q,
    typename PairingConfig::TargetField& output);
} // namespace icicle