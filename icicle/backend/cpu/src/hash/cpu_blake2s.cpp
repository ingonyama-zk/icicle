/*
   This file includes code from the Microsoft SEAL project.

   Microsoft SEAL is licensed under the MIT License.
   You may obtain a copy of the MIT License at:

     https://opensource.org/licenses/MIT

   Original source:
     https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/blake2.h
*/

#include "icicle/backend/hash/blake2s_backend.h"
#include "icicle/utils/modifiers.h"

namespace icicle {

  class Blake2sBackendCPU : public HashBackend
  {
  public:
    Blake2sBackendCPU(uint64_t input_chunk_size) : HashBackend("Blake2s-CPU", BLAKE2S_OUTBYTES, input_chunk_size) {}

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      const auto digest_size_in_bytes = output_size();
      const auto single_input_size = get_single_chunk_size(
        size); // if size==0 using default input chunk size. This is useful for Merkle-Tree constructions

      ICICLE_LOG_DEBUG << "Blake2s CPU hash() called, batch=" << config.batch
                       << ", single_output_size=" << output_size();

      // TODO (future): use tasks manager to parallel across threads. Add option to config-extension to set #threads
      // with default=0. for now we don't do it and let the merkle-tree define the parallelizm so hashing a large batch
      // outside a merkle-tree context is not as fast as it could be.
      // Note that for batch=1 this has not effect.
      for (unsigned batch_idx = 0; batch_idx < config.batch; ++batch_idx) {
        int result = blake2s(
          output + batch_idx * digest_size_in_bytes, digest_size_in_bytes, input + batch_idx * single_input_size,
          single_input_size,
          nullptr, // No key used
          0        // Key length is 0
        );

        if (result != 0) { return eIcicleError::UNKNOWN_ERROR; } // TODO Yuval error codes
      }
      return eIcicleError::SUCCESS;
    }

  private:
    static constexpr unsigned int BLAKE2S_BLOCKBYTES = 64; // input length in bytes
    static constexpr unsigned int BLAKE2S_OUTBYTES = 32;
    static constexpr unsigned int BLAKE2S_KEYBYTES = 32;
    static constexpr unsigned int BLAKE2S_SALTBYTES = 8;
    static constexpr unsigned int BLAKE2S_PERSONALBYTES = 8;

    PACKED(struct blake2s_param {
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

  const uint32_t Blake2sBackendCPU::blake2s_IV[8] = {0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
                                                     0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL};

  const uint8_t Blake2sBackendCPU::blake2s_sigma[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
  };

  void Blake2sBackendCPU::blake2s_set_lastnode(blake2s_state* S) const { S->f[1] = (uint32_t)-1; }

  /* Some helper functions, not necessarily useful */
  int Blake2sBackendCPU::blake2s_is_lastblock(const blake2s_state* S) const { return S->f[0] != 0; }

  void Blake2sBackendCPU::blake2s_set_lastblock(blake2s_state* S) const
  {
    if (S->last_node) blake2s_set_lastnode(S);

    S->f[0] = (uint32_t)-1;
  }

  void Blake2sBackendCPU::blake2s_increment_counter(blake2s_state* S, const uint32_t inc) const
  {
    S->t[0] += inc;
    S->t[1] += (S->t[0] < inc);
  }

  void Blake2sBackendCPU::blake2s_init0(blake2s_state* S) const
  {
    size_t i;
    memset(S, 0, sizeof(blake2s_state));

    for (i = 0; i < 8; ++i)
      S->h[i] = blake2s_IV[i];
  }

  /* init2 xors IV with input parameter block */
  int Blake2sBackendCPU::blake2s_init_param(blake2s_state* S, const blake2s_param* P) const
  {
    const unsigned char* p = (const unsigned char*)(P);
    size_t i;

    blake2s_init0(S);

    /* IV XOR ParamBlock */
    for (i = 0; i < 8; ++i)
      S->h[i] ^= load32(&p[i * 4]);

    S->outlen = P->digest_length;
    return 0;
  }

  /* Sequential blake2s initialization */
  int Blake2sBackendCPU::blake2s_init(blake2s_state* S, size_t outlen) const
  {
    blake2s_param P[1];

    /* Move interval verification here? */
    if ((!outlen) || (outlen > BLAKE2S_OUTBYTES)) return -1;

    P->digest_length = (uint8_t)outlen;
    P->key_length = 0;
    P->fanout = 1;
    P->depth = 1;
    store32(&P->leaf_length, 0);
    store32(&P->node_offset, 0);
    store16(&P->xof_length, 0);
    P->node_depth = 0;
    P->inner_length = 0;
    /* memset(P->reserved, 0, sizeof(P->reserved) ); */
    memset(P->salt, 0, sizeof(P->salt));
    memset(P->personal, 0, sizeof(P->personal));
    return blake2s_init_param(S, P);
  }

  int Blake2sBackendCPU::blake2s_init_key(blake2s_state* S, size_t outlen, const void* key, size_t keylen) const
  {
    blake2s_param P[1];

    if ((!outlen) || (outlen > BLAKE2S_OUTBYTES)) return -1;

    if (!key || !keylen || keylen > BLAKE2S_KEYBYTES) return -1;

    P->digest_length = (uint8_t)outlen;
    P->key_length = (uint8_t)keylen;
    P->fanout = 1;
    P->depth = 1;
    store32(&P->leaf_length, 0);
    store32(&P->node_offset, 0);
    store16(&P->xof_length, 0);
    P->node_depth = 0;
    P->inner_length = 0;
    /* memset(P->reserved, 0, sizeof(P->reserved) ); */
    memset(P->salt, 0, sizeof(P->salt));
    memset(P->personal, 0, sizeof(P->personal));

    if (blake2s_init_param(S, P) < 0) return -1;

    {
      uint8_t block[BLAKE2S_BLOCKBYTES];
      memset(block, 0, BLAKE2S_BLOCKBYTES);
      memcpy(block, key, keylen);
      blake2s_update(S, block, BLAKE2S_BLOCKBYTES);
      secure_zero_memory(block, BLAKE2S_BLOCKBYTES); /* Burn the key from stack */
    }
    return 0;
  }

#define G(r, i, a, b, c, d)                                                                                            \
  do {                                                                                                                 \
    a = a + b + m[blake2s_sigma[r][2 * i + 0]];                                                                        \
    d = rotr32(d ^ a, 16);                                                                                             \
    c = c + d;                                                                                                         \
    b = rotr32(b ^ c, 12);                                                                                             \
    a = a + b + m[blake2s_sigma[r][2 * i + 1]];                                                                        \
    d = rotr32(d ^ a, 8);                                                                                              \
    c = c + d;                                                                                                         \
    b = rotr32(b ^ c, 7);                                                                                              \
  } while (0)

#define ROUND(r)                                                                                                       \
  do {                                                                                                                 \
    G(r, 0, v[0], v[4], v[8], v[12]);                                                                                  \
    G(r, 1, v[1], v[5], v[9], v[13]);                                                                                  \
    G(r, 2, v[2], v[6], v[10], v[14]);                                                                                 \
    G(r, 3, v[3], v[7], v[11], v[15]);                                                                                 \
    G(r, 4, v[0], v[5], v[10], v[15]);                                                                                 \
    G(r, 5, v[1], v[6], v[11], v[12]);                                                                                 \
    G(r, 6, v[2], v[7], v[8], v[13]);                                                                                  \
    G(r, 7, v[3], v[4], v[9], v[14]);                                                                                  \
  } while (0)

  void Blake2sBackendCPU::blake2s_compress(blake2s_state* S, const uint8_t in[BLAKE2S_BLOCKBYTES]) const
  {
    uint32_t m[16];
    uint32_t v[16];
    size_t i;

    for (i = 0; i < 16; ++i) {
      m[i] = load32(in + i * sizeof(m[i]));
    }

    for (i = 0; i < 8; ++i) {
      v[i] = S->h[i];
    }

    v[8] = blake2s_IV[0];
    v[9] = blake2s_IV[1];
    v[10] = blake2s_IV[2];
    v[11] = blake2s_IV[3];
    v[12] = S->t[0] ^ blake2s_IV[4];
    v[13] = S->t[1] ^ blake2s_IV[5];
    v[14] = S->f[0] ^ blake2s_IV[6];
    v[15] = S->f[1] ^ blake2s_IV[7];

    ROUND(0);
    ROUND(1);
    ROUND(2);
    ROUND(3);
    ROUND(4);
    ROUND(5);
    ROUND(6);
    ROUND(7);
    ROUND(8);
    ROUND(9);

    for (i = 0; i < 8; ++i) {
      S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
    }
  }

#undef G
#undef ROUND

  int Blake2sBackendCPU::blake2s_update(blake2s_state* S, const void* pin, size_t inlen) const
  {
    const unsigned char* in = (const unsigned char*)pin;
    if (inlen > 0) {
      size_t left = S->buflen;
      size_t fill = BLAKE2S_BLOCKBYTES - left;
      if (inlen > fill) {
        S->buflen = 0;
        memcpy(S->buf + left, in, fill); /* Fill buffer */
        blake2s_increment_counter(S, BLAKE2S_BLOCKBYTES);
        blake2s_compress(S, S->buf); /* Compress */
        in += fill;
        inlen -= fill;
        while (inlen > BLAKE2S_BLOCKBYTES) {
          blake2s_increment_counter(S, BLAKE2S_BLOCKBYTES);
          blake2s_compress(S, in);
          in += BLAKE2S_BLOCKBYTES;
          inlen -= BLAKE2S_BLOCKBYTES;
        }
      }
      memcpy(S->buf + S->buflen, in, inlen);
      S->buflen += inlen;
    }
    return 0;
  }

  int Blake2sBackendCPU::blake2s_final(blake2s_state* S, void* out, size_t outlen) const
  {
    uint8_t buffer[BLAKE2S_OUTBYTES] = {0};
    size_t i;

    if (out == NULL || outlen < S->outlen) return -1;

    if (blake2s_is_lastblock(S)) return -1;

    blake2s_increment_counter(S, (uint32_t)S->buflen);
    blake2s_set_lastblock(S);
    memset(S->buf + S->buflen, 0, BLAKE2S_BLOCKBYTES - S->buflen); /* Padding */
    blake2s_compress(S, S->buf);

    for (i = 0; i < 8; ++i) /* Output full hash to temp buffer */
      store32(buffer + sizeof(S->h[i]) * i, S->h[i]);

    memcpy(out, buffer, outlen);
    secure_zero_memory(buffer, sizeof(buffer));
    return 0;
  }

  int Blake2sBackendCPU::blake2s(
    void* out, size_t outlen, const void* in, size_t inlen, const void* key, size_t keylen) const
  {
    blake2s_state S[1];

    /* Verify parameters */
    if (NULL == in && inlen > 0) return -1;

    if (NULL == out) return -1;

    if (NULL == key && keylen > 0) return -1;

    if (!outlen || outlen > BLAKE2S_OUTBYTES) return -1;

    if (keylen > BLAKE2S_KEYBYTES) return -1;

    if (keylen > 0) {
      if (blake2s_init_key(S, outlen, key, keylen) < 0) return -1;
    } else {
      if (blake2s_init(S, outlen) < 0) return -1;
    }

    blake2s_update(S, (const uint8_t*)in, inlen);
    blake2s_final(S, out, outlen);
    return 0;
  }

  /* Helper functions */

  inline uint32_t Blake2sBackendCPU::load32(const void* src) const
  {
#if defined(NATIVE_LITTLE_ENDIAN)
    uint32_t w;
    memcpy(&w, src, sizeof w);
    return w;
#else
    const uint8_t* p = (const uint8_t*)src;
    return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) | ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
#endif
  }

  inline uint64_t Blake2sBackendCPU::load64(const void* src) const
  {
#if defined(NATIVE_LITTLE_ENDIAN)
    uint64_t w;
    memcpy(&w, src, sizeof w);
    return w;
#else
    const uint8_t* p = (const uint8_t*)src;
    return ((uint64_t)(p[0]) << 0) | ((uint64_t)(p[1]) << 8) | ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
           ((uint64_t)(p[4]) << 32) | ((uint64_t)(p[5]) << 40) | ((uint64_t)(p[6]) << 48) | ((uint64_t)(p[7]) << 56);
#endif
  }

  inline uint16_t Blake2sBackendCPU::load16(const void* src) const
  {
#if defined(NATIVE_LITTLE_ENDIAN)
    uint16_t w;
    memcpy(&w, src, sizeof w);
    return w;
#else
    const uint8_t* p = (const uint8_t*)src;
    return (uint16_t)(((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8));
#endif
  }

  inline void Blake2sBackendCPU::store16(void* dst, uint16_t w) const
  {
#if defined(NATIVE_LITTLE_ENDIAN)
    memcpy(dst, &w, sizeof w);
#else
    uint8_t* p = (uint8_t*)dst;
    *p++ = (uint8_t)w;
    w >>= 8;
    *p++ = (uint8_t)w;
#endif
  }

  inline void Blake2sBackendCPU::store32(void* dst, uint32_t w) const
  {
#if defined(NATIVE_LITTLE_ENDIAN)
    memcpy(dst, &w, sizeof w);
#else
    uint8_t* p = (uint8_t*)dst;
    p[0] = (uint8_t)(w >> 0);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
#endif
  }

  inline void Blake2sBackendCPU::store64(void* dst, uint64_t w) const
  {
#if defined(NATIVE_LITTLE_ENDIAN)
    memcpy(dst, &w, sizeof w);
#else
    uint8_t* p = (uint8_t*)dst;
    p[0] = (uint8_t)(w >> 0);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
    p[4] = (uint8_t)(w >> 32);
    p[5] = (uint8_t)(w >> 40);
    p[6] = (uint8_t)(w >> 48);
    p[7] = (uint8_t)(w >> 56);
#endif
  }

  inline uint64_t Blake2sBackendCPU::load48(const void* src) const
  {
    const uint8_t* p = (const uint8_t*)src;
    return ((uint64_t)(p[0]) << 0) | ((uint64_t)(p[1]) << 8) | ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
           ((uint64_t)(p[4]) << 32) | ((uint64_t)(p[5]) << 40);
  }

  inline void Blake2sBackendCPU::store48(void* dst, uint64_t w) const
  {
    uint8_t* p = (uint8_t*)dst;
    p[0] = (uint8_t)(w >> 0);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
    p[4] = (uint8_t)(w >> 32);
    p[5] = (uint8_t)(w >> 40);
  }

  inline uint32_t Blake2sBackendCPU::rotr32(const uint32_t w, const unsigned c) const
  {
    return (w >> c) | (w << (32 - c));
  }

  inline uint64_t Blake2sBackendCPU::rotr64(const uint64_t w, const unsigned c) const
  {
    return (w >> c) | (w << (64 - c));
  }

  /* prevents compiler optimizing out memset() */
  inline void Blake2sBackendCPU::secure_zero_memory(void* v, size_t n) const
  {
    static void* (*const volatile memset_v)(void*, int, size_t) = &memset;
    memset_v(v, 0, n);
  }

  /************************ Blake2s registration ************************/
  eIcicleError
  create_blake2s_hash_backend(const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend)
  {
    backend = std::make_shared<Blake2sBackendCPU>(input_chunk_size);
    return eIcicleError::SUCCESS;
  }

  REGISTER_BLAKE2S_FACTORY_BACKEND("CPU", create_blake2s_hash_backend);

} // namespace icicle