#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

namespace koalabear {
  struct fp_config {
    static constexpr storage<1> modulus = {0x7f000001};
    static constexpr storage_array<17, 1> reduced_digits = {
      {{0x00000001},
       {0x01fffffe},
       {0x17f7efe4},
       {0x18af7f37},
       {0x423d7c8c},
       {0x1b79fadf},
       {0x7b9d44cd},
       {0x64d31ca2},
       {0x5bc34d59},
       {0x0f077438},
       {0x4fb48092},
       {0x2d55aa32},
       {0x389de76c},
       {0x02dff10b},
       {0x023486f8},
       {0x4e8e0e2d},
       {0x55a7320c}}};
    static constexpr unsigned reduced_digits_count = 17;
    PARAMS(modulus)
    MOD_SQR_SUBS()

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
