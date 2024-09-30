include(cmake/fields_and_curves.cmake)
include(cmake/target_editor.cmake)

function(extract_curve_names CURVE_NAMES_OUT)
  set(CURVE_NAMES "")

  foreach (ITEM ${ICICLE_CURVES})
    string(REPLACE ":" ";" ITEM_SPLIT ${ITEM})
    list(GET ITEM_SPLIT 1 CURVE_NAME)
    list(APPEND CURVE_NAMES ${CURVE_NAME})
  endforeach()

  # Output the list of curve names
  set(${CURVE_NAMES_OUT} "${CURVE_NAMES}" PARENT_SCOPE)
endfunction()

function(check_curve CURVE CURVE_INDEX_OUT FEATURES_STRING_OUT)
  set(IS_CURVE_SUPPORTED FALSE)
  foreach (ITEM ${ICICLE_CURVES})
    string(REPLACE ":" ";" ITEM_SPLIT ${ITEM})
    list(GET ITEM_SPLIT 0 CURVE_INDEX)
    list(GET ITEM_SPLIT 1 CURVE_NAME)
    list(GET ITEM_SPLIT 2 FEATURES_STRING)

    if (CURVE STREQUAL CURVE_NAME)
      set(IS_CURVE_SUPPORTED TRUE)
      message(STATUS "building CURVE_NAME=${CURVE_NAME} ; CURVE_INDEX=${CURVE_INDEX} ; SUPPORTED_FEATURES=${FEATURES_STRING}")
      # Output the CURVE_INDEX and FEATURES_STRING
      set(${CURVE_INDEX_OUT} "${CURVE_INDEX}" PARENT_SCOPE)
      set(${FEATURES_STRING_OUT} "${FEATURES_STRING}" PARENT_SCOPE)
      break()
    endif ()
  endforeach()

  if (NOT IS_CURVE_SUPPORTED)
    set(ALL_CURVES "")
    extract_curve_names(ALL_CURVES)
    message(FATAL_ERROR "The value of CURVE variable: ${CURVE} is not supported: choose from [${ALL_CURVES}]")
  endif ()
endfunction()


function(setup_curve_target CURVE CURVE_INDEX FEATURES_STRING)
  # the scalar field of the curve is built to a field library (like babybear is built)
  setup_field_target(${CURVE} ${CURVE_INDEX} ${FEATURES_STRING})

  add_library(icicle_curve SHARED)

  # Split FEATURES_STRING into a list using "," as the separator
  string(REPLACE "," ";" FEATURES_LIST ${FEATURES_STRING})

  # customize the curve lib to choose what to include
  handle_curve(icicle_curve) # basic curve and field methods, including vec ops
  # Handle features
  handle_msm(icicle_curve "${FEATURES_LIST}")
  handle_g2(icicle_curve "${FEATURES_LIST}")
  handle_ecntt(icicle_curve "${FEATURES_LIST}")
  # Add additional feature handling calls here

  set_target_properties(icicle_curve PROPERTIES OUTPUT_NAME "icicle_curve_${CURVE}")
  if(NOT ANDROID)
    target_link_libraries(icicle_curve PUBLIC icicle_device icicle_field pthread)
  else()
    # Android doesn't need pthread, it's already included in the system
    target_link_libraries(icicle_curve PUBLIC icicle_device icicle_field)
  endif()

  # Ensure CURVE is defined in the cache for backends to see
  set(CURVE "${CURVE}" CACHE STRING "")
  target_compile_definitions(icicle_curve PUBLIC CURVE=${CURVE} CURVE_ID=${CURVE_INDEX})

  install(TARGETS icicle_curve
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()