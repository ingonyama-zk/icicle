#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

typedef Symbol<scalar_t>* SymbolHandle;

extern "C" { // TODO remove program from names
  // Symbol functions
  // Constructors
  SymbolHandle CONCAT_EXPAND(FIELD, create_empty_symbol)() { return new Symbol<scalar_t>(); } // TODO rename symbol functions without program and split to a separate file
  SymbolHandle CONCAT_EXPAND(FIELD, create_scalar_symbol)(const scalar_t* constant) {
    return new Symbol<scalar_t>(*constant);
  }
  SymbolHandle CONCAT_EXPAND(FIELD, copy_symbol)(const SymbolHandle other) { 
    return new Symbol<scalar_t>(*other);
  }

  // Destructor
  eIcicleError delete_symbol(SymbolHandle symbol)
  {
    if (!symbol) { return eIcicleError::INVALID_POINTER; }
    delete symbol;
    return eIcicleError::SUCCESS;
  }

  void CONCAT_EXPAND(FIELD, set_symbol_as_input)(SymbolHandle this_symbol, int in_index) {
    this_symbol->set_as_input(in_index);
  }

  SymbolHandle CONCAT_EXPAND(FIELD, inverse_symbol)(const SymbolHandle input) {
    return new Symbol<scalar_t>(input->inverse());
  }

  SymbolHandle CONCAT_EXPAND(FIELD, assign_symbol)(SymbolHandle this_symbol, const SymbolHandle other) {
    this_symbol->assign(other);
    return this_symbol;
  }

  SymbolHandle CONCAT_EXPAND(FIELD, add_symbols)(const SymbolHandle op_a, const SymbolHandle op_b) {
    return new Symbol<scalar_t>(op_a->add(*op_b));
  }

  SymbolHandle CONCAT_EXPAND(FIELD, multiply_symbols)(const SymbolHandle op_a, const SymbolHandle op_b) {
    return new SymbolHandle(op_a->multiply(*op_b));
  }

  SymbolHandle CONCAT_EXPAND(FIELD, sub_symbols)(const SymbolHandle op_a, const SymbolHandle op_b) {
    return new Symbol<scalar_t>(op_a->sub(*op_b));
  }
}