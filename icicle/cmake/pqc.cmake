

function(setup_pqc_target)
  add_library(icicle_pqc SHARED)
  target_sources(icicle_pqc PRIVATE 
   src/pqc/ml_kem/ml_kem.cpp
   src/pqc/ml_kem/ml_kem_c_api.cpp
  )
  
  target_link_libraries(icicle_pqc PUBLIC icicle_device)

  install(TARGETS icicle_pqc
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()

function(setup_pqc_package_target)
  # Create the unified PQC package as an INTERFACE library
  add_library(icicle_pqc_package INTERFACE)
  
  # Set up an alias for easier reference
  add_library(icicle::icicle_pqc_package ALIAS icicle_pqc_package)
  
  # Link all the required components to the package
  target_link_libraries(icicle_pqc_package INTERFACE
    icicle_device
    icicle_pqc
  )
  
  # Set package properties
  set_target_properties(icicle_pqc_package PROPERTIES
    VERSION "1.0.0"
    SOVERSION "1"
    EXPORT_NAME "icicle_pqc_package"
  )
  
  # Export the package and its dependencies for installation
  install(TARGETS icicle_pqc_package icicle_device icicle_pqc
    EXPORT icicle_pqc_package_targets
    INCLUDES DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  )
  
  # Install the export set
  install(EXPORT icicle_pqc_package_targets
    FILE icicle_pqc_package_targets.cmake
    NAMESPACE icicle::
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/icicle_pqc_package"
  )
  
  # Generate and install the config file
  include(CMakePackageConfigHelpers)
  
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/icicle_pqc_package_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/icicle_pqc_packageConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/icicle_pqc_package"
    PATH_VARS CMAKE_INSTALL_PREFIX
  )
  
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/icicle_pqc_packageConfigVersion.cmake"
    VERSION "1.0.0"
    COMPATIBILITY SameMajorVersion
  )
  
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/icicle_pqc_packageConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/icicle_pqc_packageConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/icicle_pqc_package"
  )
  
  # Install headers
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/pqc"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include/icicle"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
  
  # Install specific utils headers required by PQC
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/utils/log.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/utils/utils.h"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include/icicle/utils"
  )
  
  # Install core headers required by PQC
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/device.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/device_api.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/runtime.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/errors.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/memory_tracker.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/icicle/config_extension.h"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include/icicle"
  )
  
  message(STATUS "ICICLE PQC Package target configured successfully")
endfunction()

