if (CUDA_BACKEND)
  string(TOLOWER "${CUDA_BACKEND}" CUDA_BACKEND_LOWER)
  if (CUDA_BACKEND_LOWER STREQUAL "local")
    # CUDA backend is local, no need to pull
    message(STATUS "Adding CUDA backend from local path: icicle/backend/cuda")
    add_subdirectory(backend/cuda)

    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend")
  else()
    set(CUDA_BACKEND_URL "git@github.com:ingonyama-zk/icicle-cuda-backend.git")

    include(FetchContent)
    message(STATUS "Fetching cuda backend from ${CUDA_BACKEND_URL}:${CUDA_BACKEND}")
    FetchContent_Declare(
      cuda_backend
      GIT_REPOSITORY ${CUDA_BACKEND_URL}
      GIT_TAG ${CUDA_BACKEND}
    )
    FetchContent_MakeAvailable(cuda_backend)
    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/_deps/cuda_backend-build")
    endif()
endif()

if (METAL_BACKEND)
  string(TOLOWER "${METAL_BACKEND}" METAL_BACKEND_LOWER)
  if (METAL_BACKEND_LOWER STREQUAL "local")
    # METAL backend is local, no need to pull
    message(STATUS "Adding Metal backend from local path: icicle/backend/metal")
    add_subdirectory(backend/metal)

    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend")
  else()
    set(METAL_BACKEND_URL "git@github.com:ingonyama-zk/icicle-metal-backend.git")

    include(FetchContent)
    message(STATUS "Fetching cuda backend from ${METAL_BACKEND_URL}:${METAL_BACKEND}")
    FetchContent_Declare(
      metal_backend
      GIT_REPOSITORY ${METAL_BACKEND_URL}
      GIT_TAG ${METAL_BACKEND}
    )
    FetchContent_MakeAvailable(metal_backend)
    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/_deps/metal_backend-build")
    endif()
endif()

if (VULKAN_BACKEND)
  string(TOLOWER "${VULKAN_BACKEND}" VULKAN_BACKEND_LOWER)
  if (VULKAN_BACKEND_LOWER STREQUAL "local")
    # VULKAN backend is local, no need to pull
    message(STATUS "Adding Vulkan backend from local path: icicle/backend/vulkan")
    add_subdirectory(backend/vulkan)

    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend")
  else()
    set(VULKAN_BACKEND_URL "git@github.com:ingonyama-zk/icicle-vulkan-backend.git")

    include(FetchContent)
    message(STATUS "Fetching cuda backend from ${VULKAN_BACKEND_URL}:${VULKAN_BACKEND}")
    FetchContent_Declare(
      vulkan_backend
      GIT_REPOSITORY ${VULKAN_BACKEND_URL}
      GIT_TAG ${VULKAN_BACKEND}
    )
    FetchContent_MakeAvailable(vulkan_backend)
    # Set the compile definition for the backend build directory
    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/_deps/vulkan-backend-build")
    endif()
endif()