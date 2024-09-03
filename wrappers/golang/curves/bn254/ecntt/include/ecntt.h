#ifndef _BN254_ECNTT_H
#define _BN254_ECNTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NTTConfig NTTConfig;
typedef struct projective_t projective_t;

int bn254_ecntt(const projective_t* input, int size, int dir, NTTConfig* config, projective_t* output);

#ifdef __cplusplus
}
#endif

#endif