#include "icicle/program/program.h"
#include "icicle/program/returning_value_program.h"
#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "release_pool.h"

using namespace field_config;
using namespace icicle;

typedef Symbol<scalar_t>* SymbolHandle;
typedef Program<scalar_t>* ProgramHandle;
typedef ReturningValueProgram<scalar_t>* ReturningValueProgramHandle;

extern "C" {
// Program functions
ProgramHandle CONCAT_EXPAND(FIELD, create_predefined_program)(PreDefinedPrograms pre_def)
{
  return new Program<scalar_t>(pre_def);
}

// Destructor
eIcicleError delete_program(ProgramHandle program)
{
  if (!program) { return eIcicleError::INVALID_POINTER; }
  delete program;
  return eIcicleError::SUCCESS;
}

eIcicleError
CONCAT_EXPAND(FIELD, generate_program)(SymbolHandle* parameters_ptr, int nof_parameters, ProgramHandle* program)
{
  *program = create_empty_program<scalar_t>();
  std::vector<Symbol<scalar_t>> parameters_vec;
  parameters_vec.reserve(nof_parameters);

  for (int i = 0; i < nof_parameters; i++) {
    if (parameters_ptr[i] == nullptr) { return eIcicleError::INVALID_ARGUMENT; }
    parameters_vec.push_back(*parameters_ptr[i]);
  }

  (*program)->m_nof_parameters = nof_parameters;
  (*program)->generate_program(parameters_vec);

  return eIcicleError::SUCCESS;
}

ReturningValueProgramHandle CONCAT_EXPAND(FIELD, create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<scalar_t>(pre_def);
}

eIcicleError CONCAT_EXPAND(FIELD, generate_returning_value_program)(
  SymbolHandle* parameters_ptr, int nof_parameters, ReturningValueProgramHandle* returning_program)
{
  ProgramHandle program = *returning_program;
  return CONCAT_EXPAND(FIELD, generate_program)(parameters_ptr, nof_parameters, &program);
}

void CONCAT_EXPAND(FIELD, clear_symbols)()
{
  ReleasePool<Symbol<scalar_t>>& pool = ReleasePool<Symbol<scalar_t>>::instance();
  pool.clear();
}
}

#ifdef EXT_FIELD
typedef Symbol<extension_t>* ExtensionSymbolHandle;
typedef Program<extension_t>* ExtensionProgramHandle;
typedef ReturningValueProgram<extension_t>* ExtensionReturningValueProgramHandle;

extern "C" {
// Program functions
ExtensionProgramHandle CONCAT_EXPAND(FIELD, extension_create_predefined_program)(PreDefinedPrograms pre_def)
{
  return new Program<extension_t>(pre_def);
}

eIcicleError CONCAT_EXPAND(FIELD, extension_generate_program)(
  ExtensionSymbolHandle* parameters_ptr, int nof_parameters, ExtensionProgramHandle* program)
{
  *program = create_empty_program<extension_t>();
  std::vector<Symbol<extension_t>> parameters_vec;
  parameters_vec.reserve(nof_parameters);

  for (int i = 0; i < nof_parameters; i++) {
    if (parameters_ptr[i] == nullptr) { return eIcicleError::INVALID_ARGUMENT; }
    parameters_vec.push_back(*parameters_ptr[i]);
  }
  (*program)->m_nof_parameters = nof_parameters;
  (*program)->generate_program(parameters_vec);

  return eIcicleError::SUCCESS;
}

ExtensionReturningValueProgramHandle
CONCAT_EXPAND(FIELD, extension_create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<extension_t>(pre_def);
}

eIcicleError CONCAT_EXPAND(FIELD, extension_generate_returning_value_program)(
  ExtensionSymbolHandle* parameters_ptr, int nof_parameters, ExtensionReturningValueProgramHandle* returning_program)
{
  ExtensionProgramHandle program = *returning_program;
  return CONCAT_EXPAND(FIELD, extension_generate_program)(parameters_ptr, nof_parameters, &program);
}

void CONCAT_EXPAND(FIELD, extension_clear_symbols)()
{
  ReleasePool<Symbol<extension_t>>& pool = ReleasePool<Symbol<extension_t>>::instance();
  pool.clear();
}
}
#endif // EXT_FIELD
