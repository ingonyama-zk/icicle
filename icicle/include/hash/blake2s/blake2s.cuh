/*
 * blake2b.cuh CUDA Implementation of BLAKE2B Hashing
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is released into the Public Domain.
 */


 #pragma once
 typedef unsigned char BYTE;
 typedef unsigned int  WORD;
 typedef unsigned long long LONG;
 
 #include <stdlib.h>
 #include <string.h>
 #include <stdio.h>
 #include <stdint.h>
 #include "gpu-utils/device_context.cuh"
 #include "gpu-utils/error_handler.cuh"
 
 #include "hash/hash.cuh"
 using namespace hash;

namespace blake2s{
 extern "C" {
 void mcm_cuda_blake2s_hash_batch(BYTE* key, WORD keylen, BYTE * in, WORD inlen, BYTE * out, WORD n_outbit, WORD n_batch);
 }
}