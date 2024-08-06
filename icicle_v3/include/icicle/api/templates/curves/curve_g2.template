extern "C" bool ${CURVE}_g2_eq(${CURVE}::g2_projective_t* point1, ${CURVE}::g2_projective_t* point2);

extern "C" void ${CURVE}_g2_to_affine(${CURVE}::g2_projective_t* point, ${CURVE}::g2_affine_t* point_out);

extern "C" void ${CURVE}_g2_generate_projective_points(${CURVE}::g2_projective_t* points, int size);

extern "C" void ${CURVE}_g2_generate_affine_points(${CURVE}::g2_affine_t* points, int size);

extern "C" eIcicleError ${CURVE}_g2_affine_convert_montgomery(
  const ${CURVE}::g2_affine_t* input, size_t n, bool is_into, const VecOpsConfig& config, ${CURVE}::g2_affine_t* output);

extern "C" eIcicleError ${CURVE}_g2_projective_convert_montgomery(
  const ${CURVE}::g2_projective_t* input, size_t n, bool is_into, const VecOpsConfig& config, ${CURVE}::g2_projective_t* output);  