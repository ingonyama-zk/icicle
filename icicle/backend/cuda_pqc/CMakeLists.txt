cmake_minimum_required(VERSION 3.18)

# TODO: share cmake code with the closed CUDA backend ? worth it?
include(cmake/Common.cmake)
find_cuda_compiler()

project(icicle_cuda_pqc_backend LANGUAGES CUDA CXX)

set_env()
set_gpu_env()

find_package(CUDAToolkit REQUIRED)

# Build CUDA backend as a static lib
add_library(icicle_backend_cuda_pqc STATIC src/cuda_pqc_device_api.cu src/ml_kem/cuda_pqc_ml_kem.cu)
target_link_libraries(icicle_backend_cuda_pqc PRIVATE CUDA::cudart)
target_include_directories(icicle_backend_cuda_pqc PRIVATE include)

# Create an interface library that properly handles static linking
add_library(icicle_backend_cuda_pqc_interface INTERFACE)

# When using static linking, we need to use whole-archive to ensure registration works
if(ICICLE_STATIC_LINK)
  # On Linux/GCC/Clang, use --whole-archive
  if(UNIX AND NOT APPLE)
    target_link_libraries(icicle_backend_cuda_pqc_interface INTERFACE 
      "-Wl,--whole-archive" 
      icicle_backend_cuda_pqc 
      "-Wl,--no-whole-archive"
    )
  # On macOS, use -force_load
  elseif(APPLE)
    target_link_libraries(icicle_backend_cuda_pqc_interface INTERFACE 
      "-Wl,-force_load" 
      "$<TARGET_FILE:icicle_backend_cuda_pqc>"
    )
  # On Windows, use /WHOLEARCHIVE
  elseif(WIN32)
    message(WARNING "EXPERIMENTAL: Windows platform is not supported yet, this feature wasn't tested yet")
    target_link_libraries(icicle_backend_cuda_pqc_interface INTERFACE 
      "/WHOLEARCHIVE:icicle_backend_cuda_pqc"
    )
  endif()
else()
  # For shared libraries, just link normally
  target_link_libraries(icicle_backend_cuda_pqc_interface INTERFACE icicle_backend_cuda_pqc)
endif()

# Install targets
install(TARGETS icicle_backend_cuda_pqc icicle_backend_cuda_pqc_interface
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")

# Install headers
install(DIRECTORY include DESTINATION "${CMAKE_INSTALL_PREFIX}/include/")

# Build tests if requested
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()