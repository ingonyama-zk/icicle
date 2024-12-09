#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"

using namespace field_config;

extern "C" void CONCAT_EXPAND(FIELD, generate_scalars)(scalar_t* scalars, int size)
{
  scalar_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(FIELD, from_u32)(uint32_t val, scalar_t* res)
{
  *res = scalar_t::from(val);
}

extern "C" void CONCAT_EXPAND(FIELD, to_montgomery)(const scalar_t& scalar, scalar_t* res)
{
  *res = scalar_t::to_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(FIELD, from_montgomery)(const scalar_t& scalar, scalar_t* res)
{
  *res = scalar_t::from_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(FIELD, sub)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 - *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, add)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 + *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, mul)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 * *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, inv)(scalar_t* scalar1, scalar_t* result)
{
  *result = scalar_t::inverse(*scalar1);
}

#ifdef EXT_FIELD
extern "C" void CONCAT_EXPAND(FIELD, extension_generate_scalars)(extension_t* scalars, int size)
{
  extension_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(FIELD, extension_from_u32)(uint32_t val, extension_t* res)
{
  *res = extension_t::from(val);
}

extern "C" void CONCAT_EXPAND(FIELD, extension_to_montgomery)(const extension_t& scalar, extension_t* res)
{
  *res = extension_t::to_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(FIELD, extension_from_montgomery)(const extension_t& scalar, extension_t* res)
{
  *res = extension_t::from_montgomery(scalar);
}

extern "C" void CONCAT_EXPAND(FIELD, extension_sub)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 - *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, extension_add)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 + *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, extension_mul)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 * *scalar2;
}
extern "C" void CONCAT_EXPAND(FIELD, extension_inv)(extension_t* scalar1, extension_t* result)
{
  *result = extension_t::inverse(*scalar1);
}
#endif // EXT_FIELD
