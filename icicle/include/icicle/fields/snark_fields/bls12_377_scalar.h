#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quadratic_extension.h"
#include "icicle/fields/params_gen.h"

namespace bls12_377 {
  struct fp_config {
    static constexpr storage<32> modulus = {std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x11}, std::byte{0x0a}, std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0xd0}, std::byte{0xfe}, std::byte{0x76}, std::byte{0xaa}, std::byte{0x59}, std::byte{0x01}, std::byte{0xb0}, std::byte{0x37}, std::byte{0x5c}, std::byte{0x1e}, std::byte{0x4d}, std::byte{0xb4}, std::byte{0x60}, std::byte{0x56}, std::byte{0xa5}, std::byte{0x2c}, std::byte{0x9a}, std::byte{0x5e}, std::byte{0x65}, std::byte{0xab}, std::byte{0x12}};
    static constexpr storage<32> m = {std::byte{0xea}, std::byte{0x79}, std::byte{0x1e}, std::byte{0x15}, std::byte{0x21}, std::byte{0x4c}, std::byte{0x20}, std::byte{0xf5}, std::byte{0x58}, std::byte{0xe2}, std::byte{0x69}, std::byte{0x8d}, std::byte{0x0b}, std::byte{0x18}, std::byte{0x0a}, std::byte{0xfd}, std::byte{0x48}, std::byte{0x05}, std::byte{0xa8}, std::byte{0xfa}, std::byte{0x49}, std::byte{0x1e}, std::byte{0xe5}, std::byte{0xe4}, std::byte{0x9e}, std::byte{0x2c}, std::byte{0x0b}, std::byte{0xc4}, std::byte{0x1e}, std::byte{0x49}, std::byte{0xd9}, std::byte{0x36}};
    static constexpr unsigned num_of_reductions = 1;
    static constexpr storage<32> modulus_2 = {std::byte{0x02}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x23}, std::byte{0x14}, std::byte{0x02}, std::byte{0x00}, std::byte{0x00}, std::byte{0xa0}, std::byte{0xfd}, std::byte{0xed}, std::byte{0x54}, std::byte{0xb3}, std::byte{0x02}, std::byte{0x60}, std::byte{0x6f}, std::byte{0xb8}, std::byte{0x3c}, std::byte{0x9a}, std::byte{0x68}, std::byte{0xc1}, std::byte{0xac}, std::byte{0x4a}, std::byte{0x59}, std::byte{0x34}, std::byte{0xbd}, std::byte{0xca}, std::byte{0x56}, std::byte{0x25}};
    static constexpr storage<32> modulus_4 = {std::byte{0x04}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x46}, std::byte{0x28}, std::byte{0x04}, std::byte{0x00}, std::byte{0x00}, std::byte{0x40}, std::byte{0xfb}, std::byte{0xdb}, std::byte{0xa9}, std::byte{0x66}, std::byte{0x05}, std::byte{0xc0}, std::byte{0xde}, std::byte{0x70}, std::byte{0x79}, std::byte{0x34}, std::byte{0xd1}, std::byte{0x82}, std::byte{0x59}, std::byte{0x95}, std::byte{0xb2}, std::byte{0x68}, std::byte{0x7a}, std::byte{0x95}, std::byte{0xad}, std::byte{0x4a}};
    static constexpr storage<32> neg_modulus = {std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0x7f}, std::byte{0xee}, std::byte{0xf5}, std::byte{0xfe}, std::byte{0xff}, std::byte{0xff}, std::byte{0x2f}, std::byte{0x01}, std::byte{0x89}, std::byte{0x55}, std::byte{0xa6}, std::byte{0xfe}, std::byte{0x4f}, std::byte{0xc8}, std::byte{0xa3}, std::byte{0xe1}, std::byte{0xb2}, std::byte{0x4b}, std::byte{0x9f}, std::byte{0xa9}, std::byte{0x5a}, std::byte{0xd3}, std::byte{0x65}, std::byte{0xa1}, std::byte{0x9a}, std::byte{0x54}, std::byte{0xed}};

    PARAMS(modulus)

    // static constexpr storage<8> rou = {0xec2a895e, 0x476ef4a4, 0x63e3f04a, 0x9b506ee3,
    //                                    0xd1a8a12f, 0x60c69477, 0x0cb92cc1, 0x11d4b7f6};
    // TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bls12_377
