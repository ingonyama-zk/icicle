#include <stdbool.h>
#include <cuda.h>
// c_api.h

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BN254_projective_t BN254_projective_t;

BN254_projective_t* create_projective();
void delete_projective(BN254_projective_t* p);
bool eq_bn254(BN254_projective_t *point1, BN254_projective_t *point2, size_t device_id);

#ifdef __cplusplus
}
#endif
