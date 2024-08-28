#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "icicle/hash.h"

namespace icicle {

#define SHA3_ASSERT(x)
#define SHA3_TRACE(format, ...)
#define SHA3_TRACE_BUF(format, buf, l)

/*
 * This flag is used to configure "pure" Keccak, as opposed to NIST SHA3.
 */
#define SHA3_USE_KECCAK_FLAG 0x80000000
#define SHA3_CW(x)           ((x) & (~SHA3_USE_KECCAK_FLAG))

#if defined(_MSC_VER)
#define SHA3_CONST(x) x
#else
#define SHA3_CONST(x) x##L
#endif

#ifndef SHA3_ROTL64
#define SHA3_ROTL64(x, y) (((x) << (y)) | ((x) >> ((sizeof(uint64_t) * 8) - (y))))
#endif

/* 'Words' here refers to uint64_t */
#define SHA3_KECCAK_SPONGE_WORDS (((1600) / 8) / sizeof(uint64_t))

  enum SHA3_FLAGS { SHA3_FLAGS_SHA3 = 0, SHA3_FLAGS_KECCAK = 1 };

  enum SHA3_RETURN { SHA3_RETURN_OK = 0, SHA3_RETURN_BAD_PARAMS = 1 };
  typedef enum SHA3_RETURN sha3_return_t;

  class Keccak : public Hash
  {
  public:
    // Constructor
    explicit Keccak(int total_input_limbs, int total_output_limbs, enum SHA3_FLAGS flag)
        : Hash(total_input_limbs, total_output_limbs), sha_flag(flag)
    {
    }
    const size_t bit_size = this->m_total_output_limbs * sizeof(limb_t) * 8;
    const enum SHA3_FLAGS sha_flag;

    eIcicleError run_single_hash(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      const HashConfig& config,
      const limb_t* secondary_input_limbs = nullptr) const override;
    eIcicleError run_multiple_hash(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config,
      const limb_t* secondary_input_limbs = nullptr) const override;

  private:
    static const uint64_t keccakf_rndc[24];
    static const unsigned keccakf_rotc[24];
    static const unsigned keccakf_piln[24];

    // SHA-3 context structure
    struct sha3_context {
      uint64_t saved; // the portion of the input message that we didn't consume yet
      union {         // Keccak's state
        uint64_t s[SHA3_KECCAK_SPONGE_WORDS];
        uint8_t sb[SHA3_KECCAK_SPONGE_WORDS * 8];
      } u;
      unsigned byteIndex;     // 0..7--the next byte after the set one
      unsigned wordIndex;     // 0..24--the next word to integrate input
      unsigned capacityWords; // the double size of the hash output in words
    };

    // Functions
    sha3_return_t sha3_init(sha3_context* ctx, unsigned bit_size) const;
    void sha3_init256(sha3_context* ctx) const;
    void sha3_init384(sha3_context* ctx) const;
    void sha3_init512(sha3_context* ctx) const;
    SHA3_FLAGS sha3_set_flags(sha3_context* ctx, SHA3_FLAGS flags) const;
    void sha3_update(sha3_context* ctx, const void* bufIn, size_t len) const;
    const void* sha3_finalize(sha3_context* ctx) const;
    static void keccakf(uint64_t s[25]);

    // Single-call hashing
    sha3_return_t sha3_hash_buffer(
      unsigned bit_size, // 256, 384, 512
      SHA3_FLAGS flags,  // SHA3_FLAGS_NONE or SHA3_FLAGS_KECCAK
      const void* in,
      unsigned inBytes,
      void* out,
      unsigned outBytes // up to bit_size/8; truncation OK
    ) const;
  };

  class Keccak256 : public Keccak
  {
  public:
    Keccak256(int total_input_limbs) : Keccak(total_input_limbs, 256 / (8 * sizeof(limb_t)), SHA3_FLAGS_KECCAK) {}
  };

  class Keccak512 : public Keccak
  {
  public:
    Keccak512(int total_input_limbs) : Keccak(total_input_limbs, 512 / (8 * sizeof(limb_t)), SHA3_FLAGS_KECCAK) {}
  };

  class Sha3_256 : public Keccak
  {
  public:
    Sha3_256(int total_input_limbs) : Keccak(total_input_limbs, 256 / (8 * sizeof(limb_t)), SHA3_FLAGS_SHA3) {}
  };

  class Sha3_512 : public Keccak
  {
  public:
    Sha3_512(int total_input_limbs) : Keccak(total_input_limbs, 512 / (8 * sizeof(limb_t)), SHA3_FLAGS_SHA3) {}
  };

} // namespace icicle
