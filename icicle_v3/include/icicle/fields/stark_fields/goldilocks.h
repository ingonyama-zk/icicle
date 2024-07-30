#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"

// modulus = 18446744069414584321 (2^64 - 2^32 + 1)(0xFFFFFFFF00000001)
// rou = 2717 (0x00000A9D)

namespace goldilocks {
  struct fp_config {
    // static constexpr storage<8> modulus = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000011, 0x08000000};//stark252
    // static constexpr storage<1> modulus = {0x78000001};//babybear
    // static constexpr storage<2> modulus = {0x00000001, 0xffffffff};//goldilocks                                      
    // static constexpr storage<2> modulus = {0x00000001, 0xffffffff};//games   
    static constexpr storage<2> modulus = {0x00000001, 0xffffffff};//games                                    
    PARAMS(modulus)

    // static constexpr storage<8> rou = {0x42f8ef94, 0x6070024f, 0xe11a6161, 0xad187148, 0x9c8b0fa5, 0x3f046451, 0x87529cfa, 0x005282db};//stark252
    // static constexpr storage<1> rou = {0x00000089};//babybear
    static constexpr storage<1> rou = {0x00000a9d};//goldilocks

    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace goldilocks