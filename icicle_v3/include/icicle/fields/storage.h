#pragma once
#include <cstdint>

#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))

template <unsigned LIMBS_COUNT>
struct
#ifdef __CUDA_ARCH__
  __align__(LIMBS_ALIGNMENT(LIMBS_COUNT))
#endif
    storage
{
  static constexpr unsigned LC = LIMBS_COUNT;
  union {
    uint32_t limbs[LIMBS_COUNT];
    uint64_t limbs64[(LIMBS_COUNT + 1) / 2];
  };
};

template <unsigned OMEGAS_COUNT, unsigned LIMBS_COUNT>
struct
#ifdef __CUDA_ARCH__
  __align__(LIMBS_ALIGNMENT(LIMBS_COUNT))
#endif
    storage_array
{
  storage<LIMBS_COUNT> storages[OMEGAS_COUNT];
};