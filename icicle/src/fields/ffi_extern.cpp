#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"

using namespace field_config;

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, generate_scalars)(scalar_t* scalars, int size)
{
  scalar_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, sub)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, add)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, mul)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 * *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, inv)(scalar_t* scalar1, scalar_t* result)
{
  *result = scalar_t::inverse(*scalar1);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, pow)(scalar_t* base, int exp, scalar_t* result)
{
  *result = scalar_t::pow(*base, exp);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, from_u32)(uint32_t val, scalar_t* result)
{
  *result = scalar_t::from(val);
}

#ifdef EXT_FIELD
extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_generate_scalars)(extension_t* scalars, int size)
{
  extension_t::rand_host_many(scalars, size);
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_sub)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_add)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_mul)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 * *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_inv)(extension_t* scalar1, extension_t* result)
{
  *result = extension_t::inverse(*scalar1);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_pow)(extension_t* base, int exp, extension_t* result)
{
  *result = extension_t::pow(*base, exp);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_from_u32)(uint32_t val, extension_t* result)
{
  *result = extension_t::from(val);
}

#endif // EXT_FIELD

#ifdef RING
extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_generate_scalars)(scalar_rns_t* scalars, int size)
{
  scalar_rns_t::rand_host_many(scalars, size);
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_sub)(scalar_rns_t* scalar1, scalar_rns_t* scalar2, scalar_rns_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_add)(scalar_rns_t* scalar1, scalar_rns_t* scalar2, scalar_rns_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void
CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_mul)(scalar_rns_t* scalar1, scalar_rns_t* scalar2, scalar_rns_t* result)
{
  *result = *scalar1 * *scalar2;
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_inv)(scalar_rns_t* scalar1, scalar_rns_t* result)
{
  *result = scalar_rns_t::inverse(*scalar1);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_pow)(scalar_rns_t* base, int exp, scalar_rns_t* result)
{
  *result = scalar_rns_t::pow(*base, exp);
}

extern "C" void CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_from_u32)(uint32_t val, scalar_rns_t* result)
{
  *result = scalar_rns_t::from(val);
}
#endif // RING