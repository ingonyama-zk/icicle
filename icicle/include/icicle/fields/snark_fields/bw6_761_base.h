#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/params_gen.h"

namespace bw6_761 {
  struct fq_config {
    static constexpr storage<24> modulus = {0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68, 0xeaf0a437, 0x160cf8ae,
                                            0x5667a8f8, 0x98a116c2, 0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
                                            0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638, 0x8087be41, 0x528275ef,
                                            0x81d14688, 0xb926186a, 0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824};
    static constexpr storage<24> mont_inv_modulus = {
      0x8fa798dd, 0x0a559356, 0xa884646f, 0xfcfd2af7, 0xaca996ba, 0xe254c3f6, 0x92516274, 0xfb2c5cce,
      0xdab1f1d7, 0xbe8267ae, 0x1abbcb60, 0xbfa82461, 0x443838f0, 0x231e6d7f, 0x19f29313, 0xf319b762,
      0x4d3b5de7, 0x1aef4e9e, 0x9a1dd7de, 0xd3914abc, 0x6e55bcdc, 0xa6d84f9f, 0x57c3b534, 0xc5b52577};
    PARAMS_NO_MOD_INV(modulus)
  };

  // Note: this fq_config_g2 is a workaround to have different types for G1 and G2.
  // Otherwise, they have the same types, thus APIs have the same type, thus we don't know which to call when
  // specializing g2 templates.
  struct fq_config_g2 {
    static constexpr storage<24> modulus = {0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68, 0xeaf0a437, 0x160cf8ae,
                                            0x5667a8f8, 0x98a116c2, 0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
                                            0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638, 0x8087be41, 0x528275ef,
                                            0x81d14688, 0xb926186a, 0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824};
    static constexpr storage<24> mont_inv_modulus = {
      0x8fa798dd, 0x0a559356, 0xa884646f, 0xfcfd2af7, 0xaca996ba, 0xe254c3f6, 0x92516274, 0xfb2c5cce,
      0xdab1f1d7, 0xbe8267ae, 0x1abbcb60, 0xbfa82461, 0x443838f0, 0x231e6d7f, 0x19f29313, 0xf319b762,
      0x4d3b5de7, 0x1aef4e9e, 0x9a1dd7de, 0xd3914abc, 0x6e55bcdc, 0xa6d84f9f, 0x57c3b534, 0xc5b52577};

    PARAMS_NO_MOD_INV(modulus)
  };
} // namespace bw6_761
