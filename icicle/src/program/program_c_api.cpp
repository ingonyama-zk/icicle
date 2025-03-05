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

template <typename S>
eIcicleError ffi_generate_program(Program<S>* program, Symbol<S>** parameters_ptr, int nof_parameters)
{
  if (program == nullptr) { return eIcicleError::ALLOCATION_FAILED; }

  std::vector<Symbol<S>> parameters_vec;
  parameters_vec.reserve(nof_parameters);
  for (int i = 0; i < nof_parameters; i++) {
    if (parameters_ptr[i] == nullptr) { return eIcicleError::INVALID_ARGUMENT; }
    parameters_vec.push_back(*parameters_ptr[i]);
  }
  program->m_nof_parameters = nof_parameters;
  program->generate_program(parameters_vec);

  ReleasePool<Symbol<S>>::instance().clear();
  return eIcicleError::SUCCESS;
}

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
  return ffi_generate_program(*program, parameters_ptr, nof_parameters);
}

ReturningValueProgramHandle CONCAT_EXPAND(FIELD, create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<scalar_t>(pre_def);
}

eIcicleError CONCAT_EXPAND(FIELD, generate_returning_value_program)(
  SymbolHandle* parameters_ptr, int nof_parameters, ReturningValueProgramHandle* program)
{
  *program = create_empty_returning_value_program<scalar_t>();
  return ffi_generate_program(*program, parameters_ptr, nof_parameters);
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
  return ffi_generate_program(*program, parameters_ptr, nof_parameters);
}

ExtensionReturningValueProgramHandle
CONCAT_EXPAND(FIELD, extension_create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<extension_t>(pre_def);
}

eIcicleError CONCAT_EXPAND(FIELD, extension_generate_returning_value_program)(
  ExtensionSymbolHandle* parameters_ptr, int nof_parameters, ExtensionReturningValueProgramHandle* program)
{
  *program = create_empty_returning_value_program<extension_t>();
  return ffi_generate_program(*program, parameters_ptr, nof_parameters);
}
}
#endif // EXT_FIELD
