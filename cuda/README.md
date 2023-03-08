# ICICLE - Ingonyama CUDA Library

## Zero Knowledge on GPU
Cuda implementation of general functions widely used in ZKP
- MSM
- NTT
- ECNTT
- Hash functions // TODO

### Supported primitives:
- Fields
  - Scalars
  - Points
    - Projective: {x, y, z}
    - Affine: {x, y}
- Curves
  - [BLS12-381](./curves/bls12_381.cuh)
  - BN254     // TODO
  - BLS12-377 // TODO


### Usage
1. Define or select a curve for your application use [curve_template.cuh](./curves/curve_template.cuh) to define a curve
2. Include the curve in [curve_config.cuh](./curves/curve_config.cuh)
3. Now you can build and use the ICICLE appUtils