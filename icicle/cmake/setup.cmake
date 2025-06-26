# Option to cross-compile for Android
option(BUILD_FOR_ANDROID "Cross-compile for Android" OFF)

if (BUILD_FOR_ANDROID)
    message(STATUS "Configuring for Android...")

    # Use the CMake option if specified; otherwise, use the environment variable
    if (DEFINED ANDROID_NDK_HOME)
        set(CMAKE_ANDROID_NDK ${ANDROID_NDK_HOME})
    elseif (DEFINED ENV{ANDROID_NDK_HOME})
        set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_HOME})
    else()
        message(FATAL_ERROR "ANDROID_NDK_HOME is not defined. Please set the environment variable or pass -DANDROID_NDK_HOME=<path>")
    endif()
    message(STATUS "Using Android NDK: ${CMAKE_ANDROID_NDK}")

    # Set toolchain and other options
    set(CMAKE_SYSTEM_NAME Android CACHE STRING "Target system name for cross-compilation")
    
    if(DEFINED ANDROID_MIN_API)
        set(ANDROID_MIN_API ${ANDROID_MIN_API} CACHE STRING "Android API level")
    elseif (DEFINED ENV{ANDROID_MIN_API})
        set(ANDROID_MIN_API $ENV{ANDROID_MIN_API} CACHE STRING "Android API level")
    else()
        set(ANDROID_MIN_API 24 CACHE STRING "Android API level")
    endif()
    message(STATUS "Using Android API level: ${ANDROID_MIN_API}")
    
    if(DEFINED ANDROID_ABI)
        set(ANDROID_ABI ${ANDROID_ABI} CACHE STRING "Android ABI")
    elseif (DEFINED ENV{ANDROID_ABI})
        set(ANDROID_ABI $ENV{ANDROID_ABI} CACHE STRING "Android ABI")
    else()
        set(ANDROID_ABI arm64-v8a CACHE STRING "Default Android ABI")
    endif()
    message(STATUS "Using Android ABI: ${ANDROID_ABI}")

    if(DEFINED ANDROID_PLATFORM)
        set(ANDROID_PLATFORM ${ANDROID_PLATFORM} CACHE STRING "Android API level")
    elseif (DEFINED ENV{ANDROID_PLATFORM})
        set(ANDROID_PLATFORM $ENV{ANDROID_PLATFORM} CACHE STRING "Android API level")
    else()
        set(ANDROID_PLATFORM "android-${ANDROID_MIN_API}" CACHE STRING "Android API level")
    endif()
    message(STATUS "Using Android platform: ${ANDROID_PLATFORM}")

    if(DEFINED CMAKE_ANDROID_STL_TYPE)
        set(CMAKE_ANDROID_STL_TYPE ${CMAKE_ANDROID_STL_TYPE} CACHE STRING "Android STL type")
    elseif (DEFINED ENV{CMAKE_ANDROID_STL_TYPE})
        set(CMAKE_ANDROID_STL_TYPE $ENV{CMAKE_ANDROID_STL_TYPE} CACHE STRING "Android STL type")
    else()
        set(CMAKE_ANDROID_STL_TYPE c++_shared CACHE STRING "Android STL type")
    endif()
    message(STATUS "Using Android STL type: ${CMAKE_ANDROID_STL_TYPE}")

    if(DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE} CACHE FILEPATH "Path to the Android toolchain file")
    elseif (DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
        set(CMAKE_TOOLCHAIN_FILE $ENV{CMAKE_TOOLCHAIN_FILE} CACHE FILEPATH "Path to the Android toolchain file")
    else()
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_ANDROID_NDK}/build/cmake/android.toolchain.cmake" CACHE FILEPATH "Path to the Android toolchain file")
    endif()
    message(STATUS "Using Android toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

    if(DEFINED ADDITIONAL_SYSTEM_LIBRARY_PATHS)
        list(APPEND CMAKE_SYSTEM_LIBRARY_PATH ${ADDITIONAL_SYSTEM_LIBRARY_PATHS})
    elseif (DEFINED ENV{ADDITIONAL_SYSTEM_LIBRARY_PATHS})
        list(APPEND CMAKE_SYSTEM_LIBRARY_PATH $ENV{ADDITIONAL_SYSTEM_LIBRARY_PATHS})
    endif()
    message(STATUS "Using additional system library paths: ${CMAKE_SYSTEM_LIBRARY_PATH}")

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