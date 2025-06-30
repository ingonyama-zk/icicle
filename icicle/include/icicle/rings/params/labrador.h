#pragma once

#include "icicle/math/storage.h"
#include "icicle/rings/integer_ring.h"
#include "icicle/fields/params_gen.h"

// RNS fields
#include "icicle/rings/integer_ring_rns.h"
#include "icicle/fields/stark_fields/babybear.h"
#include "icicle/fields/stark_fields/koalabear.h"

// Polynomial ring (Rq, Tq)
#include "icicle/rings/polynomial_ring.h"

namespace labrador {

  /**
   * @brief Configuration for the integer ring Z_q where q = P_bb * P_kb.
   *
   * The modulus q is the product of two small prime fields:
   *  - BabyBear: P_bb
   *  - KoalaBear: P_kb
   *
   * This ring is used as the base ring for all polynomial and cryptographic operations in Labrador.
   */
  struct zq_config {
    static constexpr storage<2> modulus = {0xf7000001, 0x3b880000}; ///< q = Pbb * Pkb
    static constexpr uint64_t modulus_u64 = 0x3b880000f7000001;
    PARAMS(modulus); ///< Defines type aliases and parameters like `limbs_count`

    static constexpr storage<2> rou = {0x8be440ed, 0x0309b111}; ///< Root of unity
    static constexpr unsigned omegas_count = 24;                ///< Number of powers of root used for NTT

    static constexpr storage_array<omegas_count, limbs_count> inv =
      params_gen::template get_invs<limbs_count, omegas_count>(modulus); ///< Inverses of root powers
  };

  /**
   * @brief Defines the integer ring Z_q as a runtime type.
   */
  using Zq = IntegerRing<zq_config>;
  using scalar_t = Zq; ///< Alias for compatibility with field-oriented code (macros etc.)
  using field_t = Zq;  ///< Logical field type (even though it's a ring) for generic templates

  /**
   * @brief RNS configuration for representing Z_q using BabyBear × KoalaBear fields.
   *
   * Enables efficient ring operations using SIMD-style parallelism in native STARK fields.
   */
  struct zq_rns_config {
    using Fields = std::tuple<babybear::scalar_t, koalabear::scalar_t>; ///< Underlying fields in the RNS basis
    static constexpr unsigned limbs_count = 2;

    // Layout for limbs in packed storage
    static constexpr std::array<unsigned, 2> field_offset = {0, 1};
    static constexpr std::array<unsigned, 2> field_limb_counts = {1, 1};

    static constexpr size_t omegas_count = 24;

    /// CRT weights for reconstructing values from RNS representation
    static constexpr storage_array<2, 2> crt_weights = {
      storage<2>{0x30924914, 0x22049241}, storage<2>{0xc66db6ee, 0x19836dbf}};

    using Zq = IntegerRing<zq_config>; ///< Base ring in direct representation
  };

  /**
   * @brief RNS representation of Z_q using BabyBear and KoalaBear fields.
   */
  using scalar_rns_t = IntegerRingRns<zq_rns_config>;

  /**
   * @brief Polynomial ring R_q = Z_q[x] / (x^d + 1), where d = 64.
   *
   * This structure supports both coefficient and NTT (evaluation) domain representations.
   */
  using PolyRing = icicle::PolynomialRing<Zq, 64>;

  using Rq = PolyRing;

} // namespace labrador