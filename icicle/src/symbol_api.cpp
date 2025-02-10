#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

typedef Symbol<scalar_t>* SymbolHandle;

extern "C" {
  // Symbol functions
  // Constructors
  SymbolHandle CONCAT_EXPAND(FIELD, program_create_empty_symbol)() { return new Symbol<scalar_t>(); } // TODO rename symbol functions without program and split to a separate file
  SymbolHandle CONCAT_EXPAND(FIELD, program_create_scalar_symbol)(const scalar_t* constant) {
    return new Symbol<scalar_t>(*constant);
  }
  SymbolHandle CONCAT_EXPAND(FIELD, program_copy_symbol)(const SymbolHandle other) { 
    return new Symbol<scalar_t>(*other);
  }

  // Destructor
  eIcicleError program_delete_symbol(SymbolHandle symbol)
  {
    if (!symbol) { return eIcicleError::INVALID_POINTER; }
    delete symbol;
    return eIcicleError::SUCCESS;
  }

  void CONCAT_EXPAND(FIELD, program_set_symbol_as_input)(SymbolHandle this_symbol, int in_index) {
    this_symbol->set_as_input(in_index);
  }

  SymbolHandle CONCAT_EXPAND(FIELD, program_inverse_symbol)(const SymbolHandle input) {
    return new Symbol<scalar_t>(input->inverse());
  }

  SymbolHandle CONCAT_EXPAND(FIELD, program_assign_symbol)(SymbolHandle this_symbol, const SymbolHandle other) {
    this_symbol->assign(other);
    return this_symbol;
  }

  SymbolHandle CONCAT_EXPAND(FIELD, program_add_symbols)(const SymbolHandle op_a, const SymbolHandle op_b) {
    return new Symbol<scalar_t>(op_a->add(*op_b));
  }

  SymbolHandle CONCAT_EXPAND(FIELD, program_multiply_symbols)(const SymbolHandle op_a, const SymbolHandle op_b) {
    return new SymbolHandle(op_a->multiply(*op_b));
  }

  SymbolHandle CONCAT_EXPAND(FIELD, program_sub_symbols)(const SymbolHandle op_a, const SymbolHandle op_b) {
    return new Symbol<scalar_t>(op_a->sub(*op_b));
  }
}