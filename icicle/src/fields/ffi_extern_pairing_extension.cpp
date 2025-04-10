#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"
#include "icicle/pairing/pairing_config.h"
using namespace tower_config;

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_generate_scalars)(target_field_t* scalars, int size)
{
  target_field_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_sub)(
  target_field_t* scalar1, target_field_t* scalar2, target_field_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_add)(
  const target_field_t* scalar1, const target_field_t* scalar2, target_field_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_mul)(
  const target_field_t* scalar1, const target_field_t* scalar2, target_field_t* result)
{
  *result = *scalar1 * *scalar2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_inv)(const target_field_t* scalar1, target_field_t* result)
{
  *result = target_field_t::inverse(*scalar1);
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_pow)(const target_field_t* base, int exp, target_field_t* result)
{
  *result = target_field_t::pow(*base, exp);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing_target_field_from_u32)(uint32_t val, target_field_t* result)
{
  *result = target_field_t::from(val);
}