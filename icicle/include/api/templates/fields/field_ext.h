extern "C" void ${FIELD}ExtensionGenerateScalars(${FIELD}::extension_t* scalars, int size);

extern "C" cudaError_t ${FIELD}ExtensionScalarConvertMontgomery(
  ${FIELD}::extension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);