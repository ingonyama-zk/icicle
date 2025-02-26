#pragma once

#include "icicle/math/storage.h"
#include "icicle/rings/integer_ring.h"
#include "icicle/fields/params_gen.h"

// RNS
#include "icicle/rings/integer_ring_rns.h"
#include "icicle/fields/stark_fields/babybear.h"
#include "icicle/fields/stark_fields/koalabear.h"
// TODO missing fields

namespace greyhound {

  struct zq_config {
    // TODO: currently mock uses babybear 6 times, need to replace with actual values
    // static constexpr storage<4> modulus = {0xf9000001, 0x65f60001, 0x6bc19001, 0x0b82cc00};
    // static constexpr storage<8> modulus = {0xf2000001, 0xb01d0003, 0x5bc7ac06, 0x74af4a8a,
    //  0xdbc2fc7c, 0x0b0c4d13, 0xdc40beb3, 0x0084805b};
    static constexpr storage<5> modulus = {0x15000001, 0x9d320002, 0x92e87801, 0x174bf7c0, 0x01424e50};

    // static constexpr storage<2> modulus = {0xf7000001, 0x3b880000};
    PARAMS(modulus);
  };

  using Zq = IntegerRing<zq_config>;
  using scalar_t = Zq;

  // TODO Yuval use the actual fields
  struct zq_rns_config {
    using Fields =
      std::tuple<babybear::scalar_t, babybear::scalar_t, babybear::scalar_t, babybear::scalar_t, babybear::scalar_t>;
    static constexpr unsigned limbs_count = 5;
    // Offset in limbs_storage for each field
    static constexpr std::array<unsigned, 5> FieldOffset = {0, 1, 2, 3, 4};
    static constexpr size_t omegas_count = 24;
  };

  /**
   *  Define the integer ring ZqRns
   */
  using scalar_rns_t = IntegerRingRns<zq_rns_config>;

} // namespace greyhound