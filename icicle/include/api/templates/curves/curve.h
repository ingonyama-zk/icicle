extern "C" bool ${CURVE} _eq(
  $ { CURVE } ::projective_t* point1, $ { CURVE } ::projective_t* point2);

extern "C" void ${CURVE} _to_affine(
  $ { CURVE } ::projective_t* point, $ { CURVE } ::affine_t* point_out);

extern "C" void ${CURVE} _generate_projective_points(
  $ { CURVE } ::projective_t* points, int size);

extern "C" void ${CURVE} _generate_affine_points(
  $ { CURVE } ::affine_t* points, int size);

extern "C" cudaError_t ${CURVE} _affine_convert_montgomery(
  $ { CURVE } ::affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

extern "C" cudaError_t ${CURVE} _projective_convert_montgomery(
  $ { CURVE } ::projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);