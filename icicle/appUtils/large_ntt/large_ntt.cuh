#pragma once
#ifndef _LARGE_NTT_H
#define _LARGE_NTT_H

#include <stdint.h>

__global__ void reorder_digits_kernel(uint4* arr, uint4* arr_reordered, uint32_t log_size, bool dit);

void new_ntt(
  uint4* in,
  uint4* out,
  uint4* twiddles,
  uint4* internal_twiddles,
  uint4* basic_twiddles,
  uint32_t log_size,
  bool inv,
  bool dit);

uint4* generate_external_twiddles(
  curve_config::scalar_t basic_root, uint4* twiddles, uint4* basic_twiddles, uint32_t log_size, bool inv);

#endif //_LARGE_NTT_H