#pragma once
#ifndef ERR_H
#define ERR_H

#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line);

#define CHECK_SYNC_DEVICE_ERROR() syncDevice(__FILE__, __LINE__)
void syncDevice(const char* const file, const int line);

#endif
