#include <stdbool.h>

#ifndef _RUNTIME_H
#define _RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Device Device;
typedef struct DeviceProperties DeviceProperties;
typedef struct icicleStreamHandle icicleStreamHandle;

int icicle_load_backend(const char* path, bool is_recursive);
int icicle_set_device(const Device* device);
int icicle_get_active_device(Device* device);
int icicle_is_host_memory(const void* ptr);
int icicle_is_active_device_memory(const void* ptr);
int icicle_get_device_count(int* device_count);
int icicle_malloc(void** ptr, size_t size);
int icicle_malloc_async(void** ptr, size_t size, void* stream);
int icicle_free(void* ptr);
int icicle_free_async(void* ptr, void* stream);
int icicle_get_available_memory(size_t* total, size_t* free);
int icicle_copy_to_host(void* dst, const void* src, size_t size);
int icicle_copy_to_host_async(void* dst, const void* src, size_t size, void* stream);
int icicle_copy_to_device(void* dst, const void* src, size_t size);
int icicle_copy_to_device_async(void* dst, const void* src, size_t size, void* stream);
int icicle_create_stream(void** stream);
int icicle_destroy_stream(void* stream);
int icicle_stream_synchronize(void* stream);
int icicle_device_synchronize();
int icicle_get_device_properties(DeviceProperties* properties);
int icicle_is_device_avialable(const Device* dev);
int icicle_get_registered_devices(char* output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif