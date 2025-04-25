# Option to cross-compile for Android
option(BUILD_FOR_ANDROID "Cross-compile for Android" OFF)

if (BUILD_FOR_ANDROID)
    message(STATUS "Configuring for Android...")

    # Check for NDK in the environment variable
    if (NOT DEFINED ENV{ANDROID_NDK} AND NOT DEFINED ANDROID_NDK)
        message(FATAL_ERROR "ANDROID_NDK is not defined. Please set the environment variable or pass -DANDROID_NDK=<path>")
    endif()

    # Use the CMake option if specified; otherwise, use the environment variable
    if (DEFINED ANDROID_NDK)
        set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
    elseif (DEFINED ENV{ANDROID_NDK})
        set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
    endif()

    # Debugging message for NDK path
    message(STATUS "Using Android NDK: ${CMAKE_ANDROID_NDK}")

    # Set toolchain and other options
    set(ANDROID_MIN_API 24) # Minimum API (24 is for android 7.0 and later)
    set(CMAKE_SYSTEM_NAME Android CACHE STRING "Target system name for cross-compilation")
    set(ANDROID_ABI arm64-v8a CACHE STRING "Default Android ABI")
    set(ANDROID_PLATFORM "android-${ANDROID_MIN_API}" CACHE STRING "Android API level")
    set(CMAKE_ANDROID_ARCH_ABI "${ANDROID_ABI}" CACHE STRING "Target ABI for Android")
    set(CMAKE_ANDROID_STL_TYPE c++_shared CACHE STRING "Android STL type")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_ANDROID_NDK}/build/cmake/android.toolchain.cmake" CACHE FILEPATH "Path to the Android toolchain file")
    list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/${ANDROID_MIN_API}")

    message(STATUS "Using ANDROID_MIN_API: ${ANDROID_MIN_API}")
    message(STATUS "Using ANDROID_ABI: ${ANDROID_ABI}")

endif()

# Options for iOS cross-compilation
option(BUILD_FOR_IOS "Cross-compile for iOS" OFF)
option(IOS_SIMULATOR "Build for iOS simulator" OFF)
option(IOS_DEVICE "Build for iOS device" OFF)

if (BUILD_FOR_IOS)
    message(STATUS "Configuring for iOS...")
    
    # Force static library building for iOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    
    # Set iOS specific variables
    set(CMAKE_SYSTEM_NAME iOS CACHE STRING "Target system name for iOS")
    
    if (IOS_SIMULATOR)
        message(STATUS "Configuring for iOS Simulator...")
        set(CMAKE_OSX_SYSROOT iphonesimulator CACHE STRING "iOS simulator SDK")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "iOS architectures")
        set(CMAKE_C_COMPILER_TARGET arm64-apple-ios14.0-simulator)
        set(CMAKE_CXX_COMPILER_TARGET arm64-apple-ios14.0-simulator)
    else()
        message(STATUS "Configuring for iOS Device...")
        set(CMAKE_OSX_SYSROOT iphoneos CACHE STRING "iOS device SDK")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "iOS architectures")
        set(CMAKE_C_COMPILER_TARGET arm64-apple-ios14.0)
        set(CMAKE_CXX_COMPILER_TARGET arm64-apple-ios14.0)
    endif()
    
    # Common iOS settings
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "iOS deployment target")
    
    # Set compiler flags for iOS
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fembed-bitcode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fembed-bitcode")
    
    # Set iOS-specific build flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fPIC")
    
    # Set iOS SDK path based on target
    if (IOS_SIMULATOR)
        execute_process(
            COMMAND xcrun --sdk iphonesimulator --show-sdk-path
            OUTPUT_VARIABLE IOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        execute_process(
            COMMAND xcrun --sdk iphoneos --show-sdk-path
            OUTPUT_VARIABLE IOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    set(CMAKE_OSX_SYSROOT ${IOS_SDK_PATH} CACHE PATH "iOS SDK path")
    
    # Set up the toolchain
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    
    # Set toolchain variables
    set(CMAKE_SYSTEM_PROCESSOR arm64)
    set(CMAKE_SYSTEM_VERSION 14.0)
    
    message(STATUS "Using iOS SDK: ${CMAKE_OSX_SYSROOT}")
    message(STATUS "Using iOS architectures: ${CMAKE_OSX_ARCHITECTURES}")
    message(STATUS "Using iOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    message(STATUS "Using iOS toolchain: ${CMAKE_C_COMPILER_TARGET}")
    message(STATUS "Building static libraries for iOS")
endif()

# Platform specific libraries and compiler
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(LOG_LIB log REQUIRED)  # Android log library
    set(PLATFORM_LIBS ${LOG_LIB})
elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(PLATFORM_LIBS "")
else()
  message(STATUS "Configuring for native platform...")
  # Select the C++ compiler
  find_program(CLANG_COMPILER clang++)
  find_program(CLANG_C_COMPILER clang)

  if(CLANG_COMPILER AND CLANG_C_COMPILER)
      set(CMAKE_CXX_COMPILER ${CLANG_COMPILER} CACHE STRING "Clang++ compiler" FORCE)
      set(CMAKE_C_COMPILER ${CLANG_C_COMPILER} CACHE STRING "Clang compiler" FORCE)
  else()
      message(WARNING "ICICLE CPU works best with clang++ and clang. Defaulting to ${CLANG_COMPILER}")
  endif()

  set(PLATFORM_LIBS pthread dl)
endif()

link_libraries(${PLATFORM_LIBS})

# Find the ccache program
find_program(CCACHE_PROGRAM ccache)
# If ccache is found, use it as the compiler launcher
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")

    # Use ccache for C and C++ compilers
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
else()
    message(STATUS "ccache not found")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

# Print the selected build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")