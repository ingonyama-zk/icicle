extern "C" void ${FIELD}GenerateScalars(${FIELD}::scalar_t* scalars, int size);

extern "C" cudaError_t ${FIELD}ScalarConvertMontgomery(
  ${FIELD}::scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);