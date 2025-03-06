#pragma once

#include "icicle/math/storage.h"
#include "icicle/rings/integer_ring.h"
#include "icicle/fields/params_gen.h"

// RNS
#include "icicle/rings/integer_ring_rns.h"
#include "icicle/fields/stark_fields/babybear.h"
#include "icicle/fields/stark_fields/koalabear.h"

namespace labrador {

  /**
   * @brief Configuration for the integer ring Zq where q = Pbb * Pkb.
   * Pbb and Pkb are the primes of the BabyBear and KoalaBear fields.
   */
  struct zq_config {
    static constexpr storage<2> modulus = {0xf7000001, 0x3b880000};
    PARAMS(modulus);

    static constexpr storage<2> rou = {0x8be440ed, 0x0309b111};
    static constexpr unsigned omegas_count = 24;

    static constexpr storage_array<omegas_count, limbs_count> inv =
      params_gen::template get_invs<limbs_count, omegas_count>(modulus);
  };

  /**
   * @brief Defines the integer ring Zq.
   */
  using ZqRing = IntegerRing<zq_config>;
  using scalar_t = ZqRing;

  /**
   * @brief Configuration for the RNS representation of Zq.
   * It represents integers using the BabyBear and KoalaBear fields.
   */
  struct zq_rns_config {
    using Fields = std::tuple<babybear::scalar_t, koalabear::scalar_t>;
    static constexpr unsigned limbs_count = 2;

    // Offset+size in limbs_storage for each field
    static constexpr std::array<unsigned, 2> field_offset = {0, 1};
    static constexpr std::array<unsigned, 2> field_limb_counts = {1, 1};

    // Roots of unity
    static constexpr size_t omegas_count = 24;

    // Auto-generated (by scripts/python/integer_ring.py) CRT weights for reconstructing values from RNS representation
    static constexpr storage_array<2, 2> crt_weights = {
      storage<2>{0x30924914, 0x22049241}, storage<2>{0xc66db6ee, 0x19836dbf}};

    // Zq is the direct representation type
    using Zq = IntegerRing<zq_config>;
  };

  /**
   * @brief Defines the integer ring Zq in RNS representation.
   */
  using scalar_rns_t = IntegerRingRns<zq_rns_config>;

} // namespace labrador