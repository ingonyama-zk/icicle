
/* -------------------------------------------------------------------------
 * Works when compiled for either 32-bit or 64-bit targets, optimized for
 * 64 bit.
 *
 * Canonical implementation of Init/Update/Finalize for SHA-3 byte input.
 *
 * SHA3-256, SHA3-384, SHA-512 are implemented. SHA-224 can easily be added.
 *
 * Based on code from http://keccak.noekeon.org/ .
 * ---------------------------------------------------------------------- */

#include "icicle/backend/hash/keccak_backend.h"

namespace icicle {

#define SHA3_ASSERT(x)
#define SHA3_TRACE(format, ...)
#define SHA3_TRACE_BUF(format, buf, l)

/*
 * This flag is used to configure "pure" KeccakBackendCPU, as opposed to NIST SHA3.
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

  class KeccakBackendCPU : public HashBackend
  {
  public:
    KeccakBackendCPU(uint64_t input_chunk_size, uint64_t output_size, SHA3_FLAGS flags, const char* name)
        : HashBackend(name, output_size, input_chunk_size), sha_flag(flags)
    {
    }

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      const auto digest_size_in_bytes = output_size();
      const auto single_input_size = get_single_chunk_size(
        size); // if size==0 using default input chunk size. This is useful for Merkle-Tree constructions
      ICICLE_LOG_DEBUG << "Keccak/sha3 CPU hash() called, batch=" << config.batch
                       << ", single_output_size=" << digest_size_in_bytes << " bytes";

      for (unsigned batch_idx = 0; batch_idx < config.batch; ++batch_idx) {
        int result = sha3_hash_buffer(
          8 * digest_size_in_bytes, this->sha_flag, input + batch_idx * single_input_size, single_input_size,
          output + batch_idx * digest_size_in_bytes, digest_size_in_bytes);
        // TODO better error codes
        if (result != 0) { return eIcicleError::UNKNOWN_ERROR; }
      }
      return eIcicleError::SUCCESS;
    }

  private:
    static const uint64_t keccakf_rndc[24];
    static const unsigned keccakf_rotc[24];
    static const unsigned keccakf_piln[24];
    const enum SHA3_FLAGS sha_flag;

    // SHA-3 context structure
    struct sha3_context {
      uint64_t saved; // the portion of the input message that we didn't consume yet
      union {         // KeccakBackendCPU's state
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
      SHA3_FLAGS flags,  // SHA3_FLAGS_SHA3 or SHA3_FLAGS_KECCAK
      const void* in,
      unsigned inBytes,
      void* out,
      unsigned outBytes // up to bit_size/8; truncation OK
    ) const;
  };

  const uint64_t KeccakBackendCPU::keccakf_rndc[24] = {
    SHA3_CONST(0x0000000000000001UL), SHA3_CONST(0x0000000000008082UL), SHA3_CONST(0x800000000000808aUL),
    SHA3_CONST(0x8000000080008000UL), SHA3_CONST(0x000000000000808bUL), SHA3_CONST(0x0000000080000001UL),
    SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008009UL), SHA3_CONST(0x000000000000008aUL),
    SHA3_CONST(0x0000000000000088UL), SHA3_CONST(0x0000000080008009UL), SHA3_CONST(0x000000008000000aUL),
    SHA3_CONST(0x000000008000808bUL), SHA3_CONST(0x800000000000008bUL), SHA3_CONST(0x8000000000008089UL),
    SHA3_CONST(0x8000000000008003UL), SHA3_CONST(0x8000000000008002UL), SHA3_CONST(0x8000000000000080UL),
    SHA3_CONST(0x000000000000800aUL), SHA3_CONST(0x800000008000000aUL), SHA3_CONST(0x8000000080008081UL),
    SHA3_CONST(0x8000000000008080UL), SHA3_CONST(0x0000000080000001UL), SHA3_CONST(0x8000000080008008UL)};

  const unsigned KeccakBackendCPU::keccakf_rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                                       27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};

  const unsigned KeccakBackendCPU::keccakf_piln[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                                       15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

  /* generally called after SHA3_KECCAK_SPONGE_WORDS-ctx->capacityWords words
   * are XORed into the state s
   */
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
        j = keccakf_piln[i];
        bc[0] = s[j];
        s[j] = SHA3_ROTL64(t, keccakf_rotc[i]);
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
      s[0] ^= keccakf_rndc[round];
    }
  }

  /* *************************** Public Interface ************************ */

  /* For Init or Reset call these: */
  sha3_return_t KeccakBackendCPU::sha3_init(sha3_context* priv, unsigned bit_size) const
  {
    sha3_context* ctx = (sha3_context*)priv;
    if (bit_size != 256 && bit_size != 384 && bit_size != 512) return SHA3_RETURN_BAD_PARAMS;
    memset(ctx, 0, sizeof(*ctx));
    ctx->capacityWords = 2 * bit_size / (8 * sizeof(uint64_t));
    return SHA3_RETURN_OK;
  }

  void KeccakBackendCPU::sha3_init256(sha3_context* priv) const { KeccakBackendCPU::sha3_init(priv, 256); }

  void KeccakBackendCPU::sha3_init384(sha3_context* priv) const { KeccakBackendCPU::sha3_init(priv, 384); }

  void KeccakBackendCPU::sha3_init512(sha3_context* priv) const { KeccakBackendCPU::sha3_init(priv, 512); }

  enum SHA3_FLAGS KeccakBackendCPU::sha3_set_flags(sha3_context* priv, enum SHA3_FLAGS flags) const
  {
    sha3_context* ctx = (sha3_context*)priv;
    flags = (enum SHA3_FLAGS)(flags & SHA3_FLAGS_KECCAK);
    ctx->capacityWords |= (flags == SHA3_FLAGS_KECCAK ? SHA3_USE_KECCAK_FLAG : 0);
    return flags;
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

    SHA3_TRACE_BUF("called to update with:", buf, len);

    SHA3_ASSERT(ctx->byteIndex < 8);
    SHA3_ASSERT(ctx->wordIndex < sizeof(ctx->u.s) / sizeof(ctx->u.s[0]));

    if (len < old_tail) { /* have no complete word or haven't started
                           * the word yet */
      SHA3_TRACE("because %d<%d, store it and return", (unsigned)len, (unsigned)old_tail);
      /* endian-independent code follows: */
      while (len--)
        ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);
      SHA3_ASSERT(ctx->byteIndex < 8);
      return;
    }

    if (old_tail) { /* will have one word to process */
      SHA3_TRACE("completing one word with %d bytes", (unsigned)old_tail);
      /* endian-independent code follows: */
      len -= old_tail;
      while (old_tail--)
        ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);

      /* now ready to add saved to the sponge */
      ctx->u.s[ctx->wordIndex] ^= ctx->saved;
      SHA3_ASSERT(ctx->byteIndex == 8);
      ctx->byteIndex = 0;
      ctx->saved = 0;
      if (++ctx->wordIndex == (SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords))) {
        keccakf(ctx->u.s);
        ctx->wordIndex = 0;
      }
    }

    /* now work in full words directly from input */

    SHA3_ASSERT(ctx->byteIndex == 0);

    words = len / sizeof(uint64_t);
    tail = len - words * sizeof(uint64_t);

    SHA3_TRACE("have %d full words to process", (unsigned)words);

    for (i = 0; i < words; i++, buf += sizeof(uint64_t)) {
      const uint64_t t = (uint64_t)(buf[0]) | ((uint64_t)(buf[1]) << 8 * 1) | ((uint64_t)(buf[2]) << 8 * 2) |
                         ((uint64_t)(buf[3]) << 8 * 3) | ((uint64_t)(buf[4]) << 8 * 4) | ((uint64_t)(buf[5]) << 8 * 5) |
                         ((uint64_t)(buf[6]) << 8 * 6) | ((uint64_t)(buf[7]) << 8 * 7);
#if defined(__x86_64__) || defined(__i386__)
      SHA3_ASSERT(memcmp(&t, buf, 8) == 0);
#endif
      ctx->u.s[ctx->wordIndex] ^= t;
      if (++ctx->wordIndex == (SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords))) {
        keccakf(ctx->u.s);
        ctx->wordIndex = 0;
      }
    }

    SHA3_TRACE("have %d bytes left to process, save them", (unsigned)tail);

    /* finally, save the partial word */
    SHA3_ASSERT(ctx->byteIndex == 0 && tail < 8);
    while (tail--) {
      SHA3_TRACE("Store byte %02x '%c'", *buf, *buf);
      ctx->saved |= (uint64_t)(*(buf++)) << ((ctx->byteIndex++) * 8);
    }
    SHA3_ASSERT(ctx->byteIndex < 8);
    SHA3_TRACE("Have saved=0x%016" PRIx64 " at the end", ctx->saved);
  }

  /* This is simply the 'update' with the padding block.
   * The padding block is 0x01 || 0x00* || 0x80. First 0x01 and last 0x80
   * bytes are always present, but they can be the same byte.
   */
  void const* KeccakBackendCPU::sha3_finalize(sha3_context* priv) const
  {
    sha3_context* ctx = (sha3_context*)priv;

    SHA3_TRACE("called with %d bytes in the buffer", ctx->byteIndex);

    /* Append 2-bit suffix 01, per SHA-3 spec. Instead of 1 for padding we
     * use 1<<2 below. The 0x02 below corresponds to the suffix 01.
     * Overall, we feed 0, then 1, and finally 1 to start padding. Without
     * M || 01, we would simply use 1 to start padding. */

    uint64_t t;

    if (ctx->capacityWords & SHA3_USE_KECCAK_FLAG) {
      /* KeccakBackendCPU version */
      t = (uint64_t)(((uint64_t)1) << (ctx->byteIndex * 8));
    } else {
      /* SHA3 version */
      t = (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << ((ctx->byteIndex) * 8));
    }

    ctx->u.s[ctx->wordIndex] ^= ctx->saved ^ t;

    ctx->u.s[SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(ctx->capacityWords) - 1] ^= SHA3_CONST(0x8000000000000000UL);
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

    SHA3_TRACE_BUF("Hash: (first 32 bytes)", ctx->u.sb, 256 / 8);

    return (ctx->u.sb);
  }

  sha3_return_t KeccakBackendCPU::sha3_hash_buffer(
    unsigned bit_size, enum SHA3_FLAGS flags, const void* in, unsigned inBytes, void* out, unsigned outBytes) const
  {
    sha3_return_t err;
    sha3_context c;

    err = sha3_init(&c, bit_size);
    if (err != SHA3_RETURN_OK) return err;
    if (sha3_set_flags(&c, flags) != flags) { return SHA3_RETURN_BAD_PARAMS; }
    sha3_update(&c, in, inBytes);
    const void* h = sha3_finalize(&c);

    if (outBytes > bit_size / 8) outBytes = bit_size / 8;
    memcpy(out, h, outBytes);
    return SHA3_RETURN_OK;
  }

  /*==========================================================================*/

  // const int KECCAK_256_RATE = 136;
  // const int KECCAK_256_DIGEST = 4;
  // const int KECCAK_512_RATE = 72;
  // const int KECCAK_512_DIGEST = 8;
  // const int KECCAK_STATE_SIZE = 25;
  // const int KECCAK_PADDING_CONST = 1;
  // const int SHA3_PADDING_CONST = 6;

  class Keccak256Backend : public KeccakBackendCPU
  {
  public:
    Keccak256Backend(int input_chunk_size)
        : KeccakBackendCPU(input_chunk_size, 32 /*bytes*/, SHA3_FLAGS_KECCAK, "Keccak-256-CPU")
    {
    }
  };

  class Keccak512Backend : public KeccakBackendCPU
  {
  public:
    Keccak512Backend(int input_chunk_size)
        : KeccakBackendCPU(input_chunk_size, 64 /*bytes*/, SHA3_FLAGS_KECCAK, "Keccak-512-CPU")
    {
    }
  };

  class Sha3_256Backend : public KeccakBackendCPU
  {
  public:
    Sha3_256Backend(int input_chunk_size)
        : KeccakBackendCPU(input_chunk_size, 32 /*bytes*/, SHA3_FLAGS_SHA3, "SHA3-256-CPU")
    {
    }
  };

  class Sha3_512Backend : public KeccakBackendCPU
  {
  public:
    Sha3_512Backend(int input_chunk_size)
        : KeccakBackendCPU(input_chunk_size, 64 /*bytes*/, SHA3_FLAGS_SHA3, "SHA3-512-CPU")
    {
    }
  };

  /************************ Keccak 256 registration ************************/
  eIcicleError
  create_keccak_256_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Keccak256Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_256_FACTORY_BACKEND("CPU", create_keccak_256_hash_backend);

  /************************ Keccak 512 registration ************************/
  eIcicleError
  create_keccak_512_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Keccak512Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_KECCAK_512_FACTORY_BACKEND("CPU", create_keccak_512_hash_backend);

  /************************ SHA3 256 registration ************************/
  eIcicleError
  create_sha3_256_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Sha3_256Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_256_FACTORY_BACKEND("CPU", create_sha3_256_hash_backend);

  /************************ SHA3 512 registration ************************/
  eIcicleError
  create_sha3_512_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Sha3_512Backend>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_SHA3_512_FACTORY_BACKEND("CPU", create_sha3_512_hash_backend);

} // namespace icicle