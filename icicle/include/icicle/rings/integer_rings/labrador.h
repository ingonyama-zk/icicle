#pragma once

#include "icicle/math/storage.h"
#include "icicle/rings/integer_ring.h"
#include "icicle/fields/params_gen.h"

namespace labrador {
  // Zq such that q = Pbb*Pkb for Pbb,Pkb the primes of baby-bear and koala-bear fields
  struct zq_config {
    static constexpr storage<2> modulus = {0xf7000001, 0x3b880000};
    PARAMS(modulus);

    // This rou is generating a subgroup smaller than Zq*
    // It has roots of unity up to logn=27
    static constexpr storage<2> rou = {0xa8ba8afd, 0x13bea756};
    static constexpr unsigned omegas_count = 27;
    static constexpr storage_array<omegas_count, limbs_count> inv =
      params_gen::template get_invs<limbs_count, omegas_count>(modulus);
  };

  /**
   *  Define the integer ring Zq
   */
  using Zq = IntegerRing<zq_config>;
  using scalar_t = Zq;

  // TODO Yuval: define ZqRNS too

} // namespace labrador