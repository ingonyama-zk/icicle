#pragma once

namespace icicle::pqc::ml_kem {
#define SHA3_256_RATE  136
#define SHA3_512_RATE  72
#define SHAKE_128_RATE 168
#define SHAKE_256_RATE 136

// SHA3 domain separation and padding constants
#define SHA3_DELIM_BITS   0x06
#define SHA3_DELIM_SUFFIX 0x8000000000000000ULL

// SHAKE domain separation and padding constants
#define SHAKE_DELIM_BITS   0x1F
#define SHAKE_DELIM_SUFFIX 0x8000000000000000ULL

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#define ROTL1(x)     ((x << 1) | __signbit(*reinterpret_cast<double*>(&x)))

#define KECCAK_STATE_SIZE 25
#define KECCAK_ROUNDS     24
#define MASK              0xffffffff

#define MAX_HASHES_PER_WRAP 6 // 32 / 5 = 6
#define BEST_STATE_SIZE     25

  __constant__ const uint64_t RC[24] = {0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
                                        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
                                        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
                                        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
                                        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
                                        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};

  __constant__ uint8_t destination[32] = {
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4,
    // padding to 32
    25, 26, 27, 28, 29, 30, 31};

  __constant__ uint8_t rot_amount[32] = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
    // padding to 32
    0, 0, 0, 0, 0, 0, 0};
} // namespace icicle::pqc::ml_kem
