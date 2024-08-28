
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "icicle/hash.h"

using namespace icicle;
namespace blake2s_cpu {


#if defined(_MSC_VER)
#define BLAKE2_PACKED(x) __pragma(pack(push, 1)) x __pragma(pack(pop))
#else
#define BLAKE2_PACKED(x) x __attribute__((packed))
#endif

#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L)
#if defined(_MSC_VER)
#define BLAKE2_INLINE __inline
#elif defined(__GNUC__)
#define BLAKE2_INLINE __inline__
#else
#define BLAKE2_INLINE
#endif
#else
#define BLAKE2_INLINE inline
#endif

  class Blake2s : public Hash
  {
  public:
    explicit Blake2s(int total_input_limbs) : Hash(total_input_limbs, BLAKE2S_OUTBYTES / sizeof(limb_t)) {}

    eIcicleError
    run_single_hash(const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config, const limb_t* secondary_input_limbs = nullptr) const override;
    eIcicleError run_multiple_hash(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config,
      const limb_t* secondary_input_limbs = nullptr) const override;

  private:
    enum blake2s_constant {
      BLAKE2S_BLOCKBYTES = 64, // input length in bytes
      BLAKE2S_OUTBYTES = 32,
      BLAKE2S_KEYBYTES = 32,
      BLAKE2S_SALTBYTES = 8,
      BLAKE2S_PERSONALBYTES = 8
    };

    BLAKE2_PACKED(struct blake2s_param {
      uint8_t digest_length;                   /* 1 */
      uint8_t key_length;                      /* 2 */
      uint8_t fanout;                          /* 3 */
      uint8_t depth;                           /* 4 */
      uint32_t leaf_length;                    /* 8 */
      uint32_t node_offset;                    /* 12 */
      uint16_t xof_length;                     /* 14 */
      uint8_t node_depth;                      /* 15 */
      uint8_t inner_length;                    /* 16 */
      uint8_t salt[BLAKE2S_SALTBYTES];         /* 24 */
      uint8_t personal[BLAKE2S_PERSONALBYTES]; /* 32 */
    });

    struct blake2s_state {
      uint32_t h[8];
      uint32_t t[2];
      uint32_t f[2];
      uint8_t buf[BLAKE2S_BLOCKBYTES];
      size_t buflen;
      size_t outlen;
      uint8_t last_node;
    };

    static const uint32_t blake2s_IV[8];
    static const uint8_t blake2s_sigma[10][16];

    void blake2s_set_lastnode(blake2s_state* S) const;
    int blake2s_is_lastblock(const blake2s_state* S) const;
    void blake2s_set_lastblock(blake2s_state* S) const;
    void blake2s_increment_counter(blake2s_state* S, const uint32_t inc) const;
    void blake2s_init0(blake2s_state* S) const;
    void blake2s_compress(blake2s_state* S, const uint8_t in[BLAKE2S_BLOCKBYTES]) const;

    /* Streaming API */
    int blake2s_init(blake2s_state* S, size_t outlen) const;
    int blake2s_init_key(blake2s_state* S, size_t outlen, const void* key, size_t keylen) const;
    int blake2s_init_param(blake2s_state* S, const blake2s_param* P) const;
    int blake2s_update(blake2s_state* S, const void* in, size_t inlen) const;
    int blake2s_final(blake2s_state* S, void* out, size_t outlen) const;

    /* Simple API */
    int blake2s(void* out, size_t outlen, const void* in, size_t inlen, const void* key, size_t keylen) const;

    /* Padded structs result in a compile-time error */
    enum { BLAKE2_DUMMY_1 = 1 / (int)(sizeof(blake2s_param) == BLAKE2S_OUTBYTES) };

    // Helper functions
    uint32_t load32(const void* src) const;
    uint64_t load64(const void* src) const;
    uint16_t load16(const void* src) const;
    void store16(void* dst, uint16_t w) const;
    void store32(void* dst, uint32_t w) const;
    void store64(void* dst, uint64_t w) const;
    uint64_t load48(const void* src) const;
    void store48(void* dst, uint64_t w) const;
    uint32_t rotr32(uint32_t w, unsigned c) const;
    uint64_t rotr64(uint64_t w, unsigned c) const;
    void secure_zero_memory(void* v, size_t n) const;
  };
} // namespace blake2s_cpu