#include "icicle/config_extension.h"
#include <iostream>

using namespace icicle;
extern "C" {

ConfigExtension* create_config_extension() { return new ConfigExtension(); }

void destroy_config_extension(ConfigExtension* ext) { delete ext; }

void config_extension_set_int(ConfigExtension* ext, const char* key, int value)
{
  if (ext) { ext->set<int>(key, value); }
}

void config_extension_set_bool(ConfigExtension* ext, const char* key, bool value)
{
  if (ext) { ext->set<bool>(key, value); }
}

int config_extension_get_int(const ConfigExtension* ext, const char* key)
{
  if (!ext) { THROW_ICICLE_ERR(eIcicleError::INVALID_DEVICE, "ConfigExtension is null"); }
  return ext->get<int>(key);
}

bool config_extension_get_bool(const ConfigExtension* ext, const char* key)
{
  if (!ext) { THROW_ICICLE_ERR(eIcicleError::INVALID_DEVICE, "ConfigExtension is null"); }
  return ext->get<bool>(key);
}

ConfigExtension* clone_config_extension(const ConfigExtension* ext)
{
  if (!ext) { THROW_ICICLE_ERR(eIcicleError::INVALID_DEVICE, "ConfigExtension is null"); }
  return ext->clone();
}
} // extern "C"