#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"

namespace koalabear {
  struct fp_config {
    static constexpr storage<1> modulus = {0x7f000001};
    PARAMS(modulus)

    static constexpr storage<1> rou = {0x6ac49f88};
    TWIDDLES(modulus, rou)

  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;

} // namespace koalabear
