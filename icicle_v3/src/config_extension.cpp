#include "icicle/config_extension.h"
#include <iostream>

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
  if (ext) { return ext->get<int>(key); }
  throw std::runtime_error("ConfigExtension is null");
}

bool config_extension_get_bool(const ConfigExtension* ext, const char* key)
{
  if (ext) { return ext->get<bool>(key); }
  throw std::runtime_error("ConfigExtension is null");
}

ConfigExtension* clone_config_extension(const ConfigExtension* ext)
{
  if (ext) { return ext->clone(); }
  throw std::runtime_error("ConfigExtension is null");
}
} // extern "C"