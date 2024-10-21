#include <memory>
#include "icicle/hash/hash.h"
#include "icicle/hash/poseidon.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

extern "C" {
typedef icicle::Hash* HasherHandle;

eIcicleError CONCAT_EXPAND(FIELD, poseidon_init_constants)(const PoseidonConstantsInitOptions<scalar_t>* options)
{
  return Poseidon::init_constants<scalar_t>(options);
}

eIcicleError CONCAT_EXPAND(FIELD, poseidon_init_default_constants)()
{
  return Poseidon::init_default_constants<scalar_t>();
}

HasherHandle CONCAT_EXPAND(FIELD, create_poseidon_hasher)(unsigned arity)
{
  return new icicle::Hash(icicle::create_poseidon_hash<scalar_t>(arity));
}
}