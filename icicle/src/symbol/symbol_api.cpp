#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/utils/ffi_manager.h"

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
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}
SymbolHandle CONCAT_EXPAND(FIELD, create_scalar_symbol)(const scalar_t* constant)
{
  auto symbol_ptr = new Symbol<scalar_t>(*constant);
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}
SymbolHandle CONCAT_EXPAND(FIELD, copy_symbol)(const SymbolHandle other)
{
  auto symbol_ptr = new Symbol<scalar_t>(*other);
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

// Destructor
eIcicleError delete_symbol(SymbolHandle symbol)
{
  if (!symbol) { return eIcicleError::INVALID_POINTER; }
  delete symbol;
  return eIcicleError::SUCCESS;
}

void CONCAT_EXPAND(FIELD, set_symbol_as_input)(SymbolHandle this_symbol, int in_index)
{
  this_symbol->set_as_input(in_index);
}

SymbolHandle CONCAT_EXPAND(FIELD, inverse_symbol)(const SymbolHandle input)
{
  auto symbol_ptr = new Symbol<scalar_t>(input->inverse());
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

SymbolHandle CONCAT_EXPAND(FIELD, assign_symbol)(SymbolHandle this_symbol, const SymbolHandle other)
{
  this_symbol->assign(*other);
  return this_symbol;
}

SymbolHandle CONCAT_EXPAND(FIELD, add_symbols)(const SymbolHandle op_a, const SymbolHandle op_b)
{
  auto symbol_ptr = new Symbol<scalar_t>(op_a->add(*op_b));
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

SymbolHandle CONCAT_EXPAND(FIELD, multiply_symbols)(const SymbolHandle op_a, const SymbolHandle op_b)
{
  auto symbol_ptr = new Symbol<scalar_t>(op_a->multiply(*op_b));
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

SymbolHandle CONCAT_EXPAND(FIELD, sub_symbols)(const SymbolHandle op_a, const SymbolHandle op_b)
{
  auto symbol_ptr = new Symbol<scalar_t>(op_a->sub(*op_b));
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
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
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}
ExtensionSymbolHandle CONCAT_EXPAND(FIELD, extension_create_scalar_symbol)(const extension_t* constant)
{
  auto symbol_ptr = new Symbol<extension_t>(*constant);
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}
ExtensionSymbolHandle CONCAT_EXPAND(FIELD, extension_copy_symbol)(const ExtensionSymbolHandle other)
{
  auto symbol_ptr = new Symbol<extension_t>(*other);
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

void CONCAT_EXPAND(FIELD, extension_set_symbol_as_input)(ExtensionSymbolHandle this_symbol, int in_index)
{
  this_symbol->set_as_input(in_index);
}

ExtensionSymbolHandle CONCAT_EXPAND(FIELD, extension_inverse_symbol)(const ExtensionSymbolHandle input)
{
  auto symbol_ptr = new Symbol<extension_t>(input->inverse());
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

ExtensionSymbolHandle
CONCAT_EXPAND(FIELD, extension_assign_symbol)(ExtensionSymbolHandle this_symbol, const ExtensionSymbolHandle other)
{
  this_symbol->assign(*other);
  return this_symbol;
}

ExtensionSymbolHandle
CONCAT_EXPAND(FIELD, extension_add_symbols)(const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b)
{
  auto symbol_ptr = new Symbol<extension_t>(op_a->add(*op_b));
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

ExtensionSymbolHandle
CONCAT_EXPAND(FIELD, extension_multiply_symbols)(const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b)
{
  auto symbol_ptr = new Symbol<extension_t>(op_a->multiply(*op_b));
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}

ExtensionSymbolHandle
CONCAT_EXPAND(FIELD, extension_sub_symbols)(const ExtensionSymbolHandle op_a, const ExtensionSymbolHandle op_b)
{
  auto symbol_ptr = new Symbol<extension_t>(op_a->sub(*op_b));
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.add(symbol_ptr);
  return symbol_ptr;
}
}
#endif // EXT_FIELD