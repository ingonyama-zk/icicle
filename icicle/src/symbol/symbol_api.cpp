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
SymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, create_input_symbol)(int in_idx)
{
  auto symbol_ptr = new Symbol<scalar_t>();
  symbol_ptr->set_as_input(in_idx);
  ReleasePool<Symbol<scalar_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
SymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, create_scalar_symbol)(const scalar_t* constant)
{
  auto symbol_ptr = new Symbol<scalar_t>(*constant);
  ReleasePool<Symbol<scalar_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
SymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, copy_symbol)(const SymbolHandle other)
{
  auto symbol_ptr = new Symbol<scalar_t>(*other);
  ReleasePool<Symbol<scalar_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, inverse_symbol)(const SymbolHandle input, SymbolHandle* output)
{
  if (!input) { return eIcicleError::INVALID_POINTER; }
  *output = new Symbol<scalar_t>(input->inverse());
  ReleasePool<Symbol<scalar_t>>::instance().insert(*output);
  return eIcicleError::SUCCESS;
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, add_symbols)(const SymbolHandle op_a, const SymbolHandle op_b, SymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_t>(op_a->add(*op_b));
  ReleasePool<Symbol<scalar_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, multiply_symbols)(const SymbolHandle op_a, const SymbolHandle op_b, SymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_t>(op_a->multiply(*op_b));
  ReleasePool<Symbol<scalar_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, sub_symbols)(const SymbolHandle op_a, const SymbolHandle op_b, SymbolHandle* res)
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
ExtensionSymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_create_input_symbol)(int in_idx)
{
  auto symbol_ptr = new Symbol<extension_t>();
  symbol_ptr->set_as_input(in_idx);
  ReleasePool<Symbol<extension_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
ExtensionSymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_create_scalar_symbol)(const extension_t* constant)
{
  auto symbol_ptr = new Symbol<extension_t>(*constant);
  ReleasePool<Symbol<extension_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}
ExtensionSymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_copy_symbol)(const ExtensionSymbolHandle other)
{
  auto symbol_ptr = new Symbol<extension_t>(*other);
  ReleasePool<Symbol<extension_t>>::instance().insert(symbol_ptr);
  return symbol_ptr;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_inverse_symbol)(
  const ExtensionSymbolHandle input, ExtensionSymbolHandle* output)
{
  if (!input) { return eIcicleError::INVALID_POINTER; }
  *output = new Symbol<extension_t>(input->inverse());
  ReleasePool<Symbol<extension_t>>::instance().insert(*output);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_add_symbols)(
  const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b, ExtensionSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<extension_t>(op_a->add(*op_b));
  ReleasePool<Symbol<extension_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_multiply_symbols)(
  const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b, ExtensionSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<extension_t>(op_a->multiply(*op_b));
  ReleasePool<Symbol<extension_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_sub_symbols)(
  const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b, ExtensionSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<extension_t>(op_a->sub(*op_b));
  ReleasePool<Symbol<extension_t>>::instance().insert(*res);
  return eIcicleError::SUCCESS;
}
}
#endif // EXT_FIELD

#ifdef RING
typedef Symbol<scalar_rns_t>* RnsSymbolHandle;

extern "C" {
// Symbol functions
// Constructors
RnsSymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_create_input_symbol)(int in_idx)
{
  auto symbol_ptr = new Symbol<scalar_rns_t>();
  symbol_ptr->set_as_input(in_idx);
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(symbol_ptr);
  return symbol_ptr;
}
RnsSymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_create_scalar_symbol)(const scalar_rns_t* constant)
{
  auto symbol_ptr = new Symbol<scalar_rns_t>(*constant);
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(symbol_ptr);
  return symbol_ptr;
}
RnsSymbolHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_copy_symbol)(const RnsSymbolHandle other)
{
  auto symbol_ptr = new Symbol<scalar_rns_t>(*other);
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(symbol_ptr);
  return symbol_ptr;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_inverse_symbol)(const RnsSymbolHandle input, RnsSymbolHandle* output)
{
  if (!input) { return eIcicleError::INVALID_POINTER; }
  *output = new Symbol<scalar_rns_t>(input->inverse());
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(*output);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_add_symbols)(
  const RnsSymbolHandle op_a, const RnsSymbolHandle op_b, RnsSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_rns_t>(op_a->add(*op_b));
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_multiply_symbols)(
  const RnsSymbolHandle op_a, const RnsSymbolHandle op_b, RnsSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_rns_t>(op_a->multiply(*op_b));
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(*res);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, rns_sub_symbols)(
  const RnsSymbolHandle op_a, const RnsSymbolHandle op_b, RnsSymbolHandle* res)
{
  if (!op_a || !op_b) { return eIcicleError::INVALID_ARGUMENT; }
  *res = new Symbol<scalar_rns_t>(op_a->sub(*op_b));
  ReleasePool<Symbol<scalar_rns_t>>& pool = ReleasePool<Symbol<scalar_rns_t>>::instance();
  pool.insert(*res);
  return eIcicleError::SUCCESS;
}
}
#endif // RING