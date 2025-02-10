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
}