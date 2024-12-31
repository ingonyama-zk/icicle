#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

namespace koalabear {
  struct fp_config {
    static constexpr storage<1> modulus = {0x7f000001};
    PARAMS(modulus)

    static constexpr storage<1> rou = {0x6ac49f88};
    TWIDDLES(modulus, rou)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 3;
    // true if nonresidue is negative.
    static constexpr bool nonresidue_is_negative = false;
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;

  /**
   * Quartic extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef QuarticExtensionField<fp_config, scalar_t> q_extension_t;

  /**
   * The default extension type
   */
  typedef q_extension_t extension_t;
} // namespace koalabear
