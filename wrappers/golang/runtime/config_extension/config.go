package config_extension

// #cgo CFLAGS: -I./include/
// #include "config_extension.h"
import "C"
import (
	"unsafe"
)

type ConfigExtensionHandler = unsafe.Pointer

type ConfigExtension struct {
	handler ConfigExtensionHandler
}

func Create() *ConfigExtension {
	ext := &ConfigExtension{handler: C.create_config_extension()}
	return ext
}

func Delete(ext *ConfigExtension) {
	C.destroy_config_extension(ext.handler)
}

func (ext *ConfigExtension) SetInt(key string, value int) {
	cKey := C.CString(key)
	cValue := C.int(value)
	C.config_extension_set_int(ext.handler, cKey, cValue)
}

func (ext *ConfigExtension) SetBool(key string, value bool) {
	cKey := C.CString(key)
	cValue := C._Bool(value)
	C.config_extension_set_bool(ext.handler, cKey, cValue)
}

func (ext *ConfigExtension) GetInt(key string) int {
	cKey := C.CString(key)
	return int(C.config_extension_get_int(ext.handler, cKey))
}

func (ext *ConfigExtension) GetBool(key string) bool {
	cKey := C.CString(key)
	return C.config_extension_get_bool(ext.handler, cKey) == C._Bool(true)
}

func (ext ConfigExtension) AsUnsafePointer() unsafe.Pointer {
	return ext.handler
}
