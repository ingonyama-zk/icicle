#pragma once
#include <cstdint>

#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))

template <unsigned LIMBS_COUNT> struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) storage {
  static constexpr unsigned LC = LIMBS_COUNT;
  uint32_t limbs[LIMBS_COUNT];
};
