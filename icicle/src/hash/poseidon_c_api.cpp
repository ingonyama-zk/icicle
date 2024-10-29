#include <memory>
#include "icicle/hash/hash.h"
#include "icicle/hash/poseidon.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

extern "C" {
typedef icicle::Hash* HasherHandle;

eIcicleError CONCAT_EXPAND(FIELD, poseidon_init_constants)(const PoseidonConstantsOptions<scalar_t>* options)
{
  return Poseidon::init_constants<scalar_t>(options);
}

HasherHandle CONCAT_EXPAND(FIELD, create_poseidon_hasher)(
  unsigned arity,
  unsigned default_input_size,
  bool is_domain_tag,
  scalar_t domain_tag_value,
  bool use_all_zeroes_padding)
{
  return new icicle::Hash(icicle::create_poseidon_hash<scalar_t>(
    arity, default_input_size, is_domain_tag, domain_tag_value, use_all_zeroes_padding));
}
}