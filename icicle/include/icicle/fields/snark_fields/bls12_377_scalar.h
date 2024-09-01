#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quadratic_extension.h"
#include "icicle/fields/params_gen.h"

namespace bls12_377 {
  struct fp_config {
    static constexpr storage<32> modulus = {std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x11}, std::byte{0x0a}, std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0xd0}, std::byte{0xfe}, std::byte{0x76}, std::byte{0xaa}, std::byte{0x59}, std::byte{0x01}, std::byte{0xb0}, std::byte{0x37}, std::byte{0x5c}, std::byte{0x1e}, std::byte{0x4d}, std::byte{0xb4}, std::byte{0x60}, std::byte{0x56}, std::byte{0xa5}, std::byte{0x2c}, std::byte{0x9a}, std::byte{0x5e}, std::byte{0x65}, std::byte{0xab}, std::byte{0x12}};

;
    PARAMS(modulus)

    static constexpr storage<8> rou = {0xec2a895e, 0x476ef4a4, 0x63e3f04a, 0x9b506ee3,
                                       0xd1a8a12f, 0x60c69477, 0x0cb92cc1, 0x11d4b7f6};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bls12_377
