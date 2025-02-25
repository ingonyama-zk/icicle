#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/program/returning_value_program.h"

using namespace field_config;
using namespace icicle;

extern "C" {

typedef ReturningValueProgram<scalar_t>* ReturningValueProgramHandle;

ReturningValueProgramHandle CONCAT_EXPAND(FIELD, create_predefined_returning_value_program)(PreDefinedPrograms pre_def)
{
  return new ReturningValueProgram<scalar_t>(pre_def);
}
}