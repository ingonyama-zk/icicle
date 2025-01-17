#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

namespace babybear {
  struct fp_config {
    static constexpr storage<1> modulus = {0x78000001};
    static constexpr storage_array<17, 1> reduced_digits = {{
      {0x1},
      {0xffffffe},
      {0x45dddde3},
      {0x12f37bfb},
      {0x27922ab6},
      {0x6394fa39},
      {0xb8efb44},
      {0x57578192},
      {0x25abb864},
      {0x6fa2bae8},
      {0x11d80ae0},
      {0x71eed7bd},
      {0x4cf166f8},
      {0x43dae013},
      {0x173e21fb},
      {0x5e6a622e},
      {0x169483e4}}};
    static constexpr unsigned reduced_digits_count = 17;
    PARAMS(modulus)

    static constexpr storage<1> rou = {0x00000089};
    TWIDDLES(modulus, rou)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 11;
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
} // namespace babybear
