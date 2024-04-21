extern "C" void ${FIELD}_generate_scalars(${FIELD}::scalar_t* scalars, int size);

extern "C" cudaError_t ${FIELD}_scalar_convert_montgomery(
  ${FIELD}::scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);