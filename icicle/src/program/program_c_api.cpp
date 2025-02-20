#include "icicle/program/program.h"
#include "icicle/program/returning_value_program.h"
#include "icicle/program/symbol.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/utils/ffi_manager.h"

using namespace field_config;
using namespace icicle;

typedef Symbol<scalar_t>* SymbolHandle;
typedef Program<scalar_t>* ProgramHandle;
typedef ReturningValueProgram<scalar_t>* ReturningValueProgramHandle;

extern "C" {
// Program functions
// Constructor
ProgramHandle CONCAT_EXPAND(FIELD, create_empty_program)() { return create_empty_program<scalar_t>(); }

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

void CONCAT_EXPAND(FIELD, generate_program)(ProgramHandle program, SymbolHandle* parameters_ptr, int nof_parameters)
{
  std::vector<Symbol<scalar_t>> parameters_vec;
  parameters_vec.reserve(nof_parameters);

  for (int i = 0; i < nof_parameters; i++) {
    if (parameters_ptr[i] == nullptr) { throw std::invalid_argument("Null pointer found in parameters"); }
    parameters_vec.push_back(*parameters_ptr[i]); // TODO replace with span to avoid copying when we switch to cpp20
  }

  program->m_nof_parameters = nof_parameters;
  program->generate_program(parameters_vec);
}

ReturningValueProgramHandle CONCAT_EXPAND(FIELD, create_empty_returning_value_program)()
{
  return create_empty_returning_value_program<scalar_t>();
}

ReturningValueProgramHandle CONCAT_EXPAND(FIELD, create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<scalar_t>(pre_def);
}

void CONCAT_EXPAND(FIELD, generate_returning_value_program)(
  ReturningValueProgramHandle program, SymbolHandle* parameters_ptr, int nof_parameters)
{
  CONCAT_EXPAND(FIELD, generate_program)(program, parameters_ptr, nof_parameters);
}

void CONCAT_EXPAND(FIELD, clear_symbols)()
{
  FfiObjectPool<Symbol<scalar_t>>& pool = FfiObjectPool<Symbol<scalar_t>>::instance();
  pool.clear();
}
}

#ifdef EXT_FIELD
typedef Symbol<extension_t>* ExtensionSymbolHandle;
typedef Program<extension_t>* ExtensionProgramHandle;
typedef ReturningValueProgram<extension_t>* ExtensionReturningValueProgramHandle;

extern "C" {
// Program functions
// Constructor
ExtensionProgramHandle CONCAT_EXPAND(FIELD, extension_create_empty_program)()
{
  return create_empty_program<extension_t>();
}

ExtensionProgramHandle CONCAT_EXPAND(FIELD, extension_create_predefined_program)(PreDefinedPrograms pre_def)
{
  return new Program<extension_t>(pre_def);
}

void CONCAT_EXPAND(FIELD, extension_generate_program)(
  ExtensionProgramHandle program, ExtensionSymbolHandle* parameters_ptr, int nof_parameters)
{
  std::vector<Symbol<extension_t>> parameters_vec;
  parameters_vec.reserve(nof_parameters);

  for (int i = 0; i < nof_parameters; i++) {
    if (parameters_ptr[i] == nullptr) { throw std::invalid_argument("Null pointer found in parameters"); }
    parameters_vec.push_back(*parameters_ptr[i]); // TODO replace with span to avoid copying when we switch to cpp20
  }
  program->m_nof_parameters = nof_parameters;
  program->generate_program(parameters_vec);
}

ExtensionReturningValueProgramHandle CONCAT_EXPAND(FIELD, extension_create_empty_returning_value_program)()
{
  return create_empty_returning_value_program<extension_t>();
}

ExtensionReturningValueProgramHandle
CONCAT_EXPAND(FIELD, extension_create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<extension_t>(pre_def);
}

void CONCAT_EXPAND(FIELD, extension_generate_returning_value_program)(
  ExtensionReturningValueProgramHandle program, ExtensionSymbolHandle* parameters_ptr, int nof_parameters)
{
  CONCAT_EXPAND(FIELD, extension_generate_program)(program, parameters_ptr, nof_parameters);
}

void CONCAT_EXPAND(FIELD, extension_clear_symbols)()
{
  FfiObjectPool<Symbol<extension_t>>& pool = FfiObjectPool<Symbol<extension_t>>::instance();
  pool.clear();
}
}
#endif // EXT_FIELD
