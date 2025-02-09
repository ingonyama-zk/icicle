#include "icicle/program/program.h"
#include "icicle/program/returning_value_program.h"
#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

typedef Symbol<scalar_t>* SymbolHandle;
typedef Program<scalar_t>* ProgramHandle;

// Friend function to access the protected default program constructor
template <typename S> Program<S>* create_empty_program() {
  return new Program<S>();
}

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

  // TODO separate files upto here

  // Program functions
  // Constructor
  ProgramHandle CONCAT_EXPAND(FIELD, program_create_empty_program)() { return create_empty_program<scalar_t>(); }
  ProgramHandle CONCAT_EXPAND(FIELD, program_create_predefined_program)(PreDefinedPrograms pre_def) {
    return new ProgramHandle(pre_def);
  }

  // Destructor
  eIcicleError program_delete_program(ProgramHandle program)
  {
    if (!program) { return eIcicleError::INVALID_POINTER; }
    delete program;
    return eIcicleError::SUCCESS;
  }

  // TODO set as input here in program and don't pass it to Rust

  void CONCAT_EXPAND(FIELD, program_generate_program)(
    ProgramHandle program, SymbolHandle* parameters_ptr, int nof_parameters) {
    std::vector<Symbol<scalar_t>> parameters_vec;
    parameters_vec.reserve(nof_parameters);

    for (int i = 0; i < nof_parameters; i++)
    {
      if (parameters_ptr[i] == nullptr)
      {
        throw std::invalid_argument("Null pointer found in parameters");
      }
      parameters_vec.push_back(*parameters_ptr[i]); // TODO replace with span toi avoid copying
    }
    program->generate_program(parameters_vec);
  }

  int program_get_nof_vars(const ProgramHandle prog) { return prog->get_nof_vars(); } // TODO remove

  void program_print_program(const ProgramHandle program) { program->print_program(); } // TODO remove, it's for internal debug
}