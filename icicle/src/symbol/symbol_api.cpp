#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "../program/release_pool.h"

using namespace field_config;
using namespace icicle;

typedef Symbol<scalar_t>* SymbolHandle;

extern "C" {
// Symbol functions
// Constructors
SymbolHandle CONCAT_EXPAND(FIELD, create_input_symbol)(int in_idx)
{
  auto symbol_ptr = new Symbol<scalar_t>();
  symbol_ptr->set_as_input(in_idx);
  ReleasePool<Symbol<scalar_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
SymbolHandle CONCAT_EXPAND(FIELD, create_scalar_symbol)(const scalar_t* constant)
{
  auto symbol_ptr = new Symbol<scalar_t>(*constant);
  ReleasePool<Symbol<scalar_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
SymbolHandle CONCAT_EXPAND(FIELD, copy_symbol)(const SymbolHandle other)
{
  auto symbol_ptr = new Symbol<scalar_t>(*other);
  ReleasePool<Symbol<scalar_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}

eIcicleError CONCAT_EXPAND(FIELD, inverse_symbol)(const SymbolHandle input, SymbolHandle* output)
{
  if (!input) { return eIcicleError::INVALID_POINTER; }
  *output = new Symbol<scalar_t>(input->inverse());
  ReleasePool<Symbol<scalar_t>>::instance().insert(*output);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(FIELD, add_symbols)(const SymbolHandle op_a, const SymbolHandle op_b, SymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_t>(op_a->add(*op_b));
  ReleasePool<Symbol<scalar_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(FIELD, multiply_symbols)(const SymbolHandle op_a, const SymbolHandle op_b, SymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_t>(op_a->multiply(*op_b));
  ReleasePool<Symbol<scalar_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(FIELD, sub_symbols)(const SymbolHandle op_a, const SymbolHandle op_b, SymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_t>(op_a->sub(*op_b));
  ReleasePool<Symbol<scalar_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}
}

#ifdef EXT_FIELD
typedef Symbol<extension_t>* ExtensionSymbolHandle;

extern "C" {
// Symbol functions
// Constructors
ExtensionSymbolHandle CONCAT_EXPAND(FIELD, extension_create_input_symbol)(int in_idx)
{
  auto symbol_ptr = new Symbol<extension_t>();
  symbol_ptr->set_as_input(in_idx);
  ReleasePool<Symbol<extension_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
ExtensionSymbolHandle CONCAT_EXPAND(FIELD, extension_create_scalar_symbol)(const extension_t* constant)
{
  auto symbol_ptr = new Symbol<extension_t>(*constant);
  ReleasePool<Symbol<extension_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
ExtensionSymbolHandle CONCAT_EXPAND(FIELD, extension_copy_symbol)(const ExtensionSymbolHandle other)
{
  auto symbol_ptr = new Symbol<extension_t>(*other);
  ReleasePool<Symbol<extension_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}

eIcicleError
CONCAT_EXPAND(FIELD, extension_inverse_symbol)(const ExtensionSymbolHandle input, ExtensionSymbolHandle* output)
{
  if (!input) { return eIcicleError::INVALID_POINTER; }
  *output = new Symbol<extension_t>(input->inverse());
  ReleasePool<Symbol<extension_t>>::instance().insert(*output);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(FIELD, extension_add_symbols)(
  const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b, ExtensionSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<extension_t>(op_a->add(*op_b));
  ReleasePool<Symbol<extension_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(FIELD, extension_multiply_symbols)(
  const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b, ExtensionSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<extension_t>(op_a->multiply(*op_b));
  ReleasePool<Symbol<extension_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(FIELD, extension_sub_symbols)(
  const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b, ExtensionSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<extension_t>(op_a->sub(*op_b));
  ReleasePool<Symbol<extension_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}
}
#endif // EXT_FIELD