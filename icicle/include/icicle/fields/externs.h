#pragma once

#define ICICLE_FFI_CONCAT3(a, b, c)            a##b##c
#define ICICLE_FFI_EXPAND_AND_CONCAT3(a, b, c) ICICLE_FFI_CONCAT3(a, b, c)

#define ICICLE_DEFINE_FIELD_FFI_FUNCS(PREFIX, TYPE)                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _generate_random)(TYPE * scalars, int size) \
  {                                                                                                                    \
    TYPE::rand_host_many(scalars, size);                                                                               \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _sub)(TYPE * a, TYPE * b, TYPE * result)    \
  {                                                                                                                    \
    *result = *a - *b;                                                                                                 \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _add)(TYPE * a, TYPE * b, TYPE * result)    \
  {                                                                                                                    \
    *result = *a + *b;                                                                                                 \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _mul)(TYPE * a, TYPE * b, TYPE * result)    \
  {                                                                                                                    \
    *result = *a * *b;                                                                                                 \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _inv)(TYPE * a, TYPE * result)              \
  {                                                                                                                    \
    *result = a->inverse();                                                                                            \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _sqr)(TYPE * a, TYPE * result)              \
  {                                                                                                                    \
    *result = a->sqr();                                                                                                \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _pow)(TYPE * base, int exp, TYPE* result)   \
  {                                                                                                                    \
    *result = base->pow(exp);                                                                                          \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _from_u32)(uint32_t val, TYPE * result)     \
  {                                                                                                                    \
    *result = TYPE::from(val);                                                                                         \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _eq)(TYPE * a, TYPE * b, bool* result)      \
  {                                                                                                                    \
    *result = *a == *b;                                                                                                \
  }                                                                                                                    \
  extern "C" void ICICLE_FFI_EXPAND_AND_CONCAT3(ICICLE_FFI_PREFIX, PREFIX, _from_bytes_le)(                            \
    uint8_t * bytes, TYPE * result)                                                                                    \
  {                                                                                                                    \
    *result = TYPE::reduce_from_bytes((std::byte*)bytes);                                                              \
  }