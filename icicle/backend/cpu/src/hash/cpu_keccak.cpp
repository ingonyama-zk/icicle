/*
   This file includes code from the SHA3IUF project.

   SHA3IUF is licensed under the MIT License.
   You may obtain a copy of the MIT License at:

     https://opensource.org/licenses/MIT

   Original source:
     https://github.com/brainhub/SHA3IUF
*/

#include "icicle/backend/hash/keccak_backend.h"
#include "icicle/utils/modifiers.h"

namespace icicle {

// Configuration flags for Keccak and SHA-3
#define SHA3_USE_KECCAK_FLAG (0x80000000)
#define SHA3_CW(x)           ((x) & (~SHA3_USE_KECCAK_FLAG))
#ifndef SHA3_ROTL64
  #define SHA3_ROTL64(x, y) (((x) << (y)) | ((x) >> ((sizeof(uint64_t) * 8) - (y))))
#endif

// Define sponge words for Keccak
#define SHA3_KECCAK_SPONGE_WORDS (((1600) / 8) / sizeof(uint64_t))

  class KeccakBackendCPU : public HashBackend
  {
  public:
    KeccakBackendCPU(uint64_t input_chunk_size, uint64_t output_size, bool is_keccak, const char* name)
        : HashBackend{name, output_size, input_chunk_size}, m_is_keccak{is_keccak}
    {
    }

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      const auto digest_size_in_bytes = output_size();
      const auto single_input_size = get_single_chunk_size(
        size); // if size==0 using default input chunk size. This is useful for Merkle-Tree constructions

      // TODO (future): use tasks manager to parallel across threads. Add option to config-extension to set #threads
      // with default=0. for now we don't do it and let the merkle-tree define the parallelizm so hashing a large batch
      // outside a merkle-tree context is not as fast as it could be.
      // Note that for batch=1 this has not effect.
      for (unsigned batch_idx = 0; batch_idx < config.batch; ++batch_idx) {
        eIcicleError err = sha3_hash_buffer(
          8 * digest_size_in_bytes /*=bitsize*/, m_is_keccak, input + batch_idx * single_input_size, single_input_size,
          output + batch_idx * digest_size_in_bytes);

        if (err != eIcicleError::SUCCESS) { return err; }
      }
      return eIcicleError::SUCCESS;
    }

  private:
    static const uint64_t s_keccakf_rndc[24];
    static const unsigned s_keccakf_rotc[24];
    static const unsigned s_keccakf_piln[24];
    const bool m_is_keccak;

    // SHA-3 context structure
    struct sha3_context {
      uint64_t saved = 0; // the portion of the input message that we didn't consume yet
      union {             // Keccaks's state
        uint64_t s[SHA3_KECCAK_SPONGE_WORDS];
        uint8_t sb[SHA3_KECCAK_SPONGE_WORDS * 8];
      } u = {0};
      unsigned byteIndex = 0;     // 0..7--the next byte after the set one
      unsigned wordIndex = 0;     // 0..24--the next word to integrate input
      unsigned capacityWords = 0; // the double size of the hash output in words
    };

    eIcicleError sha3_init(sha3_context* ctx, unsigned bit_size, bool is_keccak) const;
    void sha3_update(sha3_context* ctx, const void* bufIn, size_t len) const;
    const void* sha3_finalize(sha3_context* ctx) const;
    static void keccakf(uint64_t s[25]);

    eIcicleError sha3_hash_buffer(unsigned bit_size, bool is_keccak, const void* in, unsigned inBytes, void* out) const;
  };

  const uint64_t KeccakBackendCPU::s_keccakf_rndc[24] = {
    LONG_CONST_SUFFIX(0x0000000000000001UL), LONG_CONST_SUFFIX(0x0000000000008082UL),
    LONG_CONST_SUFFIX(0x800000000000808aUL), LONG_CONST_SUFFIX(0x8000000080008000UL),
    LONG_CONST_SUFFIX(0x000000000000808bUL), LONG_CONST_SUFFIX(0x0000000080000001UL),
    LONG_CONST_SUFFIX(0x8000000080008081UL), LONG_CONST_SUFFIX(0x8000000000008009UL),
    LONG_CONST_SUFFIX(0x000000000000008aUL), LONG_CONST_SUFFIX(0x0000000000000088UL),
    LONG_CONST_SUFFIX(0x0000000080008009UL), LONG_CONST_SUFFIX(0x000000008000000aUL),
    LONG_CONST_SUFFIX(0x000000008000808bUL), LONG_CONST_SUFFIX(0x800000000000008bUL),
    LONG_CONST_SUFFIX(0x8000000000008089UL), LONG_CONST_SUFFIX(0x8000000000008003UL),
    LONG_CONST_SUFFIX(0x8000000000008002UL), LONG_CONST_SUFFIX(0x8000000000000080UL),
    LONG_CONST_SUFFIX(0x000000000000800aUL), LONG_CONST_SUFFIX(0x800000008000000aUL),
    LONG_CONST_SUFFIX(0x8000000080008081UL), LONG_CONST_SUFFIX(0x8000000000008080UL),
    LONG_CONST_SUFFIX(0x0000000080000001UL), LONG_CONST_SUFFIX(0x8000000080008008UL)};

  const unsigned KeccakBackendCPU::s_keccakf_rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                                         27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};

  const unsigned KeccakBackendCPU::s_keccakf_piln[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                                         15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

  // Keccak round function
  void KeccakBackendCPU::keccakf(uint64_t s[25])
  {
    int i, j, round;
    uint64_t t, bc[5];
#define KECCAK_ROUNDS 24

    for (round = 0; round < KECCAK_ROUNDS; round++) {
      /* Theta */
      for (i = 0; i < 5; i++)
        bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];

      for (i = 0; i < 5; i++) {
        t = bc[(i + 4) % 5] ^ SHA3_ROTL64(bc[(i + 1) % 5], 1);
        for (j = 0; j < 25; j += 5)
          s[j + i] ^= t;
      }

      /* Rho Pi */
      t = s[1];
      for (i = 0; i < 24; i++) {
        j = s_keccakf_piln[i];
        bc[0] = s[j];
        s[j] = SHA3_ROTL64(t, s_keccakf_rotc[i]);
        t = bc[0];
      }

      /* Chi */
      for (j = 0; j < 25; j += 5) {
        for (i = 0; i < 5; i++)
          bc[i] = s[j + i];
        for (i = 0; i < 5; i++)
          s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
      }

      /* Iota */
      s[0] ^= s_keccakf_rndc[round];
    }
  }

  eIcicleError KeccakBackendCPU::sha3_init(sha3_context* ctx, unsigned bit_size, bool is_keccak) const
  {
    if (bit_size != 256 && bit_size != 384 && bit_size != 512) {
      ICICLE_LOG_ERROR << "Invalid bitsize for Keccak. Supported bitsize in [256, 384, 512]";
      return eIcicleError::INVALID_ARGUMENT;
    }
    ctx->capacityWords = 2 * bit_size / (8 * sizeof(uint64_t));
    if (is_keccak) { ctx->capacityWords |= SHA3_USE_KECCAK_FLAG; }
    return eIcicleError::SUCCESS;
  }

  void KeccakBackendCPU::sha3_update(sha3_context* priv, void const* bufIn, size_t len) const
  {
    sha3_context* ctx = (sha3_context*)priv;

    /* 0...7 -- how much is needed to have a word */
    unsigned old_tail = (8 - ctx->byteIndex) & 7;
    size_t words;
    unsigned tail;
    size_t i;

    const uint8_t* buf = (const uint8_t*)bufIn;

    ICICLE_ASSERT(ctx->byteIndex < 8);
    ICICLE_ASSERT(ctx->wordIndex < sizeof(ctx->u.s) / sizeof(ctx->u.s[0]));

    if (len < old_tail) { /* have no complete word or haven't started
                           * the word yet */
      /* endian-independent code follows: */
      while (len--)
        ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);
      ICICLE_ASSERT(ctx->byteIndex < 8);
      return;
    }

    if (old_tail) { /* will have one word to process */
      /* endian-independent code follows: */
      len -= old_tail;
      while (old_tail--)
        ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);

      /* now ready to add saved to the sponge */
      ctx->u.s[ctx->wordIndex] ^= ctx->saved;
      ICICLE_ASSERT(ctx->byteIndex == 8);
      ctx->byteIndex = 0;
      ctx->saved = 0;
      if (++ctx->wordIndex == (SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords))) {
        keccakf(ctx->u.s);
        ctx->wordIndex = 0;
      }
    }

    /* now work in full words directly from input */

    ICICLE_ASSERT(ctx->byteIndex == 0);

    words = len / sizeof(uint64_t);
    tail = len - words * sizeof(uint64_t);

    for (i = 0; i < words; i++, buf += sizeof(uint64_t)) {
      const uint64_t t = (uint64_t)(buf[0]) | ((uint64_t)(buf[1]) << 8 * 1) | ((uint64_t)(buf[2]) << 8 * 2) |
                         ((uint64_t)(buf[3]) << 8 * 3) | ((uint64_t)(buf[4]) << 8 * 4) | ((uint64_t)(buf[5]) << 8 * 5) |
                         ((uint64_t)(buf[6]) << 8 * 6) | ((uint64_t)(buf[7]) << 8 * 7);
#if defined(__x86_64__) || defined(__i386__)
      ICICLE_ASSERT(memcmp(&t, buf, 8) == 0);
#endif
      ctx->u.s[ctx->wordIndex] ^= t;
      if (++ctx->wordIndex == (SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords))) {
        keccakf(ctx->u.s);
        ctx->wordIndex = 0;
      }
    }

    /* finally, save the partial word */
    ICICLE_ASSERT(ctx->byteIndex == 0 && tail < 8);
    while (tail--) {
      ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);
    }
    ICICLE_ASSERT(ctx->byteIndex < 8);
  }

  /* This is simply the 'update' with the padding block.
   * The padding block is 0x01 || 0x00* || 0x80. First 0x01 and last 0x80
   * bytes are always present, but they can be the same byte.
   */
  void const* KeccakBackendCPU::sha3_finalize(sha3_context* ctx) const
  {
    /* Append 2-bit suffix 01, per SHA-3 spec. Instead of 1 for padding we
     * use 1<<2 below. The 0x02 below corresponds to the suffix 01.
     * Overall, we feed 0, then 1, and finally 1 to start padding. Without
     * M || 01, we would simply use 1 to start padding. */

    uint64_t t;

    if (ctx->capacityWords & SHA3_USE_KECCAK_FLAG) {
      /* Keccak version */
      t = (uint64_t)(((uint64_t)1) << (ctx->byteIndex * 8));
    } else {
      /* SHA3 version */
      t = (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << ((ctx->byteIndex) * 8));
    }

    ctx->u.s[ctx->wordIndex] ^= ctx->saved ^ t;
    ctx->u.s[SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords) - 1] ^= LONG_CONST_SUFFIX(0x8000000000000000UL);
    keccakf(ctx->u.s);

    /* Return first bytes of the ctx->s. This conversion is not needed for
     * little-endian platforms e.g. wrap with #if !defined(__BYTE_ORDER__)
     * || !defined(__ORDER_LITTLE_ENDIAN__) || __BYTE_ORDER__!=__ORDER_LITTLE_ENDIAN__
     *    ... the conversion below ...
     * #endif */
    {
      unsigned i;
      for (i = 0; i < SHA3_KECCAK_SPONGE_WORDS; i++) {
        const unsigned t1 = (uint32_t)ctx->u.s[i];
        const unsigned t2 = (uint32_t)((ctx->u.s[i] >> 16) >> 16);
        ctx->u.sb[i * 8 + 0] = (uint8_t)(t1);
        ctx->u.sb[i * 8 + 1] = (uint8_t)(t1 >> 8);
        ctx->u.sb[i * 8 + 2] = (uint8_t)(t1 >> 16);
        ctx->u.sb[i * 8 + 3] = (uint8_t)(t1 >> 24);
        ctx->u.sb[i * 8 + 4] = (uint8_t)(t2);
        ctx->u.sb[i * 8 + 5] = (uint8_t)(t2 >> 8);
        ctx->u.sb[i * 8 + 6] = (uint8_t)(t2 >> 16);
        ctx->u.sb[i * 8 + 7] = (uint8_t)(t2 >> 24);
      }
    }
    return (ctx->u.sb);
  }

  eIcicleError KeccakBackendCPU::sha3_hash_buffer(
    unsigned bit_size, bool is_keccak, const void* in, unsigned inBytes, void* out) const
  {
    sha3_context ctx;
    eIcicleError err = sha3_init(&ctx, bit_size, is_keccak);
    if (err != eIcicleError::SUCCESS) return err;
    sha3_update(&ctx, in, inBytes);
    const void* h = sha3_finalize(&ctx);

    // copy out 32/64B (from 1600B). Can we avoid this copy? seems no
    memcpy(out, h, bit_size >> 3);
    return eIcicleError::SUCCESS;
  }

  /************************ Keccak 256 registration ************************/
  static eIcicleError
  create_keccak_256_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<KeccakBackendCPU>(input_chunk_size, 32, true /*is_keccak*/, "Keccak-256-CPU");
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_256_FACTORY_BACKEND("CPU", create_keccak_256_hash_backend);

  /************************ Keccak 512 registration ************************/
  static eIcicleError
  create_keccak_512_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<KeccakBackendCPU>(input_chunk_size, 64, true /*is_keccak*/, "Keccak-512-CPU");
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_512_FACTORY_BACKEND("CPU", create_keccak_512_hash_backend);

  /************************ SHA3 256 registration ************************/
  static eIcicleError
  create_sha3_256_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<KeccakBackendCPU>(input_chunk_size, 32, false /*is_keccak*/, "SHA3-256-CPU");
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_256_FACTORY_BACKEND("CPU", create_sha3_256_hash_backend);

  /************************ SHA3 512 registration ************************/
  static eIcicleError
  create_sha3_512_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<KeccakBackendCPU>(input_chunk_size, 64, false /*is_keccak*/, "SHA3-512-CPU");
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_512_FACTORY_BACKEND("CPU", create_sha3_512_hash_backend);

} // namespace icicle