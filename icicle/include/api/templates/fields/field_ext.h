extern "C" void ${FIELD} _extension_generate_scalars(
  $ { FIELD } ::extension_t* scalars, int size);

extern "C" cudaError_t ${FIELD} _extension_scalar_convert_montgomery(
  $ { FIELD } ::extension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);