#include <assert.h>
extern "C" {
#include "blake2s.cuh"
}
// #include <chrono>
// #define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
// #define END_TIMER(timer, msg) \
//   printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

#define BLAKE2S_ROUNDS 10
#define BLAKE2S_BLOCK_LENGTH 64
#define BLAKE2S_CHAIN_SIZE 8
#define BLAKE2S_CHAIN_LENGTH (BLAKE2S_CHAIN_SIZE * sizeof(uint32_t))
#define BLAKE2S_STATE_SIZE 16
#define BLAKE2S_STATE_LENGTH (BLAKE2S_STATE_SIZE * sizeof(uint32_t))

typedef struct {
    WORD digestlen;
    BYTE key[32];
    WORD keylen;
    BYTE buff[BLAKE2S_BLOCK_LENGTH];
    uint32_t chain[BLAKE2S_CHAIN_SIZE];
    uint32_t state[BLAKE2S_STATE_SIZE];
    WORD pos;
    uint32_t t0;
    uint32_t t1;
    uint32_t f0;
} cuda_blake2s_ctx_t;

typedef cuda_blake2s_ctx_t CUDA_BLAKE2S_CTX;

__constant__ CUDA_BLAKE2S_CTX c_CTX;

__constant__ uint32_t BLAKE2S_IVS[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

const uint32_t CPU_BLAKE2S_IVS[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

void cpu_blake2s_init(cuda_blake2s_ctx_t *ctx, BYTE *key, WORD keylen, WORD digestbitlen) {
    memset(ctx, 0, sizeof(cuda_blake2s_ctx_t));
    if (keylen > 0) {
        memcpy(ctx->buff, key, keylen);
        memcpy(ctx->key, key, keylen);
    }
    ctx->keylen = keylen;
    ctx->digestlen = digestbitlen >> 3;
    ctx->pos = 0;
    ctx->t0 = 0;
    ctx->t1 = 0;
    ctx->f0 = 0;
    ctx->chain[0] = CPU_BLAKE2S_IVS[0] ^ (ctx->digestlen | (ctx->keylen << 8) | 0x1010000);
    ctx->chain[1] = CPU_BLAKE2S_IVS[1];
    ctx->chain[2] = CPU_BLAKE2S_IVS[2];
    ctx->chain[3] = CPU_BLAKE2S_IVS[3];
    ctx->chain[4] = CPU_BLAKE2S_IVS[4];
    ctx->chain[5] = CPU_BLAKE2S_IVS[5];
    ctx->chain[6] = CPU_BLAKE2S_IVS[6];
    ctx->chain[7] = CPU_BLAKE2S_IVS[7];

    ctx->pos = (keylen > 0) ? BLAKE2S_BLOCK_LENGTH : 0;
}

__constant__ uint8_t BLAKE2S_SIGMA[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 }
};

__device__ uint32_t cuda_blake2s_leuint32(BYTE *in) {
    uint32_t a;
    memcpy(&a, in, 4);
    return a;
}

__device__ uint32_t cuda_blake2s_ROTR32(uint32_t a, uint8_t b) {
    return (a >> b) | (a << (32 - b));
}

__device__ void cuda_blake2s_G(cuda_blake2s_ctx_t *ctx, uint32_t m1, uint32_t m2, int32_t a, int32_t b, int32_t c, int32_t d) {
    ctx->state[a] = ctx->state[a] + ctx->state[b] + m1;
    ctx->state[d] = cuda_blake2s_ROTR32(ctx->state[d] ^ ctx->state[a], 16);
    ctx->state[c] = ctx->state[c] + ctx->state[d];
    ctx->state[b] = cuda_blake2s_ROTR32(ctx->state[b] ^ ctx->state[c], 12);
    ctx->state[a] = ctx->state[a] + ctx->state[b] + m2;
    ctx->state[d] = cuda_blake2s_ROTR32(ctx->state[d] ^ ctx->state[a], 8);
    ctx->state[c] = ctx->state[c] + ctx->state[d];
    ctx->state[b] = cuda_blake2s_ROTR32(ctx->state[b] ^ ctx->state[c], 7);
}

__device__ __forceinline__ void cuda_blake2s_init_state(cuda_blake2s_ctx_t *ctx) {
    memcpy(ctx->state, ctx->chain, BLAKE2S_CHAIN_LENGTH);
    // ctx->state[8] = ctx->t0;
    // ctx->state[9] = ctx->t1;
    // ctx->state[10] = ctx->f0;
    // ctx->state[11] = BLAKE2S_IVS[4];
    ctx->state[8] = BLAKE2S_IVS[0];
    ctx->state[9] = BLAKE2S_IVS[1];
    ctx->state[10] = BLAKE2S_IVS[2];
    ctx->state[11] = BLAKE2S_IVS[3];
    ctx->state[12] = ctx->t0 ^ BLAKE2S_IVS[4];
    ctx->state[13] = ctx->t1 ^ BLAKE2S_IVS[5];
    ctx->state[14] = ctx->f0 ^ BLAKE2S_IVS[6];
    ctx->state[15] = BLAKE2S_IVS[7];
    // ctx->state[12] = BLAKE2S_IVS[5];
    // ctx->state[13] = BLAKE2S_IVS[6];
    // ctx->state[14] = BLAKE2S_IVS[7];
}

__device__ __forceinline__ void cuda_blake2s_compress(cuda_blake2s_ctx_t *ctx, BYTE *in, WORD inoffset) {
    cuda_blake2s_init_state(ctx);
    uint32_t m[16] = { 0 };
    for (int j = 0; j < 16; j++)
        m[j] = cuda_blake2s_leuint32(in + inoffset + (j << 2));

    for (int round = 0; round < BLAKE2S_ROUNDS; round++) {
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][0]], m[BLAKE2S_SIGMA[round][1]], 0, 4, 8, 12);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][2]], m[BLAKE2S_SIGMA[round][3]], 1, 5, 9, 13);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][4]], m[BLAKE2S_SIGMA[round][5]], 2, 6, 10, 14);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][6]], m[BLAKE2S_SIGMA[round][7]], 3, 7, 11, 15);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][8]], m[BLAKE2S_SIGMA[round][9]], 0, 5, 10, 15);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][10]], m[BLAKE2S_SIGMA[round][11]], 1, 6, 11, 12);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][12]], m[BLAKE2S_SIGMA[round][13]], 2, 7, 8, 13);
        cuda_blake2s_G(ctx, m[BLAKE2S_SIGMA[round][14]], m[BLAKE2S_SIGMA[round][15]], 3, 4, 9, 14);
    }

    for (int offset = 0; offset < BLAKE2S_CHAIN_SIZE; offset++)
        ctx->chain[offset] = ctx->chain[offset] ^ ctx->state[offset] ^ ctx->state[offset + 8];
}

__device__ void cuda_blake2s_init(cuda_blake2s_ctx_t *ctx, BYTE *key, WORD keylen, WORD digestbitlen) {
    memset(ctx, 0, sizeof(cuda_blake2s_ctx_t));
    ctx->keylen = keylen;
    ctx->digestlen = digestbitlen >> 3;
    ctx->pos = 0;
    ctx->t0 = 0;
    ctx->t1 = 0;
    ctx->f0 = 0;
    ctx->chain[0] = BLAKE2S_IVS[0] ^ (ctx->digestlen | (ctx->keylen << 8) | 0x1010000);
    ctx->chain[1] = BLAKE2S_IVS[1];
    ctx->chain[2] = BLAKE2S_IVS[2];
    ctx->chain[3] = BLAKE2S_IVS[3];
    ctx->chain[4] = BLAKE2S_IVS[4];
    ctx->chain[5] = BLAKE2S_IVS[5];
    ctx->chain[6] = BLAKE2S_IVS[6];
    ctx->chain[7] = BLAKE2S_IVS[7];
    
    if (keylen > 0) {
        memcpy(ctx->buff, key, keylen);
        memcpy(ctx->key, key, keylen);
    }
    ctx->pos = (keylen > 0) ? BLAKE2S_BLOCK_LENGTH : 0;
}

__device__ void cuda_blake2s_update(cuda_blake2s_ctx_t *ctx, BYTE *in, LONG inlen) {
    if (inlen == 0)
        return;

    WORD start = 0;
    int64_t in_index = 0, block_index = 0;

    if (ctx->pos) {
        start = BLAKE2S_BLOCK_LENGTH - ctx->pos;
        if (start < inlen) {
            memcpy(ctx->buff + ctx->pos, in, start);
            ctx->t0 += BLAKE2S_BLOCK_LENGTH;

            if (ctx->t0 == 0) ctx->t1++;

            cuda_blake2s_compress(ctx, ctx->buff, 0);
            ctx->pos = 0;
            memset(ctx->buff, 0, BLAKE2S_BLOCK_LENGTH);
        } else {
            memcpy(ctx->buff + ctx->pos, in, inlen);
            ctx->pos += inlen;
            return;
        }
    }

    block_index = inlen - BLAKE2S_BLOCK_LENGTH;
    for (in_index = start; in_index < block_index; in_index += BLAKE2S_BLOCK_LENGTH) {
        ctx->t0 += BLAKE2S_BLOCK_LENGTH;
        if (ctx->t0 == 0)
            ctx->t1++;

        cuda_blake2s_compress(ctx, in, in_index);
    }

    memcpy(ctx->buff, in + in_index, inlen - in_index);
    ctx->pos += inlen - in_index;
}

__device__ void cuda_blake2s_final(cuda_blake2s_ctx_t *ctx, BYTE *out) {
    ctx->f0 = 0xFFFFFFFFUL;
    ctx->t0 += ctx->pos;
    if (ctx->pos > 0 && ctx->t0 == 0)
        ctx->t1++;

    cuda_blake2s_compress(ctx, ctx->buff, 0);
    memset(ctx->buff, 0, BLAKE2S_BLOCK_LENGTH);
    memset(ctx->state, 0, BLAKE2S_STATE_LENGTH);

    int i4 = 0;
    for (int i = 0; i < BLAKE2S_CHAIN_SIZE && ((i4 = i * 4) < ctx->digestlen); i++) {
        BYTE *BYTEs = (BYTE*)(&ctx->chain[i]);
        if (i4 < ctx->digestlen - 4)
            memcpy(out + i4, BYTEs, 4);
        else
            memcpy(out + i4, BYTEs, ctx->digestlen - i4);
    }
}

__global__ void kernel_blake2s_hash(BYTE *indata, WORD inlen, BYTE *outdata, WORD n_batch, WORD BLAKE2S_BLOCK_SIZE) {
    WORD thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= n_batch) {
        return;
    }
    BYTE *in = indata + thread * inlen;
    BYTE *out = outdata + thread * BLAKE2S_BLOCK_SIZE;
    CUDA_BLAKE2S_CTX ctx = c_CTX;
    cuda_blake2s_update(&ctx, in, inlen);
    cuda_blake2s_final(&ctx, out);
}

extern "C" {
void mcm_cuda_blake2s_hash_batch(BYTE *key, WORD keylen, BYTE *in, WORD inlen, BYTE *out, WORD n_outbit, WORD n_batch) {
    BYTE *cuda_indata;
    BYTE *cuda_outdata;
    const WORD BLAKE2S_BLOCK_SIZE = (n_outbit >> 3);
    cudaMalloc(&cuda_indata, inlen * n_batch);
    cudaMalloc(&cuda_outdata, BLAKE2S_BLOCK_SIZE * n_batch);

    CUDA_BLAKE2S_CTX ctx;
    assert(keylen <= 32);
    cpu_blake2s_init(&ctx, key, keylen, n_outbit);

    cudaMemcpy(cuda_indata, in, inlen * n_batch, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_CTX, &ctx, sizeof(CUDA_BLAKE2S_CTX), 0, cudaMemcpyHostToDevice);

    WORD thread = 256;
    WORD block = (n_batch + thread - 1) / thread;
    kernel_blake2s_hash<<<block, thread>>>(cuda_indata, inlen, cuda_outdata, n_batch, BLAKE2S_BLOCK_SIZE);
    cudaMemcpy(out, cuda_outdata, BLAKE2S_BLOCK_SIZE * n_batch, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error cuda blake2s hash: %s \n", cudaGetErrorString(error));
    }
    cudaFree(cuda_indata);
    cudaFree(cuda_outdata);
}
}
