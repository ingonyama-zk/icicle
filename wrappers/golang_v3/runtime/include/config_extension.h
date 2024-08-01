#include <stdbool.h>

#ifndef _CONFIG_EXTENSION_H
#define _CONFIG_EXTENSION_H

#ifdef __cplusplus
extern "C" {
#endif

// typedef ConfigExtension ConfigExtension;

void* create_config_extension();
void destroy_config_extension(void* ext);
void config_extension_set_int(void* ext, const char* key, int value);
void config_extension_set_bool(void* ext, const char* key, bool value);
int config_extension_get_int(const void* ext, const char* key);
bool config_extension_get_bool(const void* ext, const char* key);
void* clone_config_extension(const void* ext);

#ifdef __cplusplus
}
#endif

#endif
