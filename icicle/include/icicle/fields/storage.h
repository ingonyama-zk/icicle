#pragma once
#include <cstdint>

#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))

template <unsigned LIMBS_COUNT>
struct ALIGN(LIMBS_ALIGNMENT(LIMBS_COUNT)) storage;

// Specialization for LIMBS_COUNT == 1
template <>
struct ALIGN(LIMBS_ALIGNMENT(1)) storage<1> {
  static constexpr unsigned LC = 1;
  uint32_t limbs[1];
};

// Specialization for LIMBS_COUNT == 3
template <>
struct ALIGN(LIMBS_ALIGNMENT(3)) storage<3> {
  static constexpr unsigned LC = 3;
  uint32_t limbs[3];
};

// General template for LIMBS_COUNT > 1
template <unsigned LIMBS_COUNT>
struct ALIGN(LIMBS_ALIGNMENT(LIMBS_COUNT)) storage {
  static_assert(LIMBS_COUNT % 2 == 0, "odd number of limbs is not supported\n");
  static constexpr unsigned LC = LIMBS_COUNT;
  union { // works only with even LIMBS_COUNT
    uint32_t limbs[LIMBS_COUNT];
    uint64_t limbs64[LIMBS_COUNT / 2];
  };
};

template <unsigned OMEGAS_COUNT, unsigned LIMBS_COUNT>
struct ALIGN(LIMBS_ALIGNMENT(LIMBS_COUNT)) storage_array {
  storage<LIMBS_COUNT> storages[OMEGAS_COUNT];
};