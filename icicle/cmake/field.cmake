include(cmake/fields_and_curves.cmake)
include(cmake/target_editor.cmake)

function(extract_field_names FIELD_NAMES_OUT)
  set(FIELD_NAMES "")

  foreach (ITEM ${ICICLE_FIELDS})
    string(REPLACE ":" ";" ITEM_SPLIT ${ITEM})
    list(GET ITEM_SPLIT 1 FIELD_NAME)
    list(APPEND FIELD_NAMES ${FIELD_NAME})
  endforeach()

  # Output the list of field names
  set(${FIELD_NAMES_OUT} "${FIELD_NAMES}" PARENT_SCOPE)
endfunction()

function(check_field FIELD FIELD_INDEX_OUT FEATURES_STRING_OUT)
  set(IS_FIELD_SUPPORTED FALSE)
  foreach (ITEM ${ICICLE_FIELDS})
    string(REPLACE ":" ";" ITEM_SPLIT ${ITEM})
    list(GET ITEM_SPLIT 0 FIELD_INDEX)
    list(GET ITEM_SPLIT 1 FIELD_NAME)
    list(GET ITEM_SPLIT 2 FEATURES_STRING)

    if (FIELD STREQUAL FIELD_NAME)
      set(IS_FIELD_SUPPORTED TRUE)
      message(STATUS "building FIELD_NAME=${FIELD_NAME} ; FIELD_INDEX=${FIELD_INDEX} ; SUPPORTED_FEATURES=${FEATURES_STRING}")
      # Output the FIELD_INDEX and FEATURES_STRING
      set(${FIELD_INDEX_OUT} "${FIELD_INDEX}" PARENT_SCOPE)
      set(${FEATURES_STRING_OUT} "${FEATURES_STRING}" PARENT_SCOPE)
      break()
    endif ()
  endforeach()

  if (NOT IS_FIELD_SUPPORTED)
    set(ALL_FIELDS "")
    extract_field_names(ALL_FIELDS)
    message(FATAL_ERROR "The value of FIELD variable: ${FIELD} is not supported: choose from [${ALL_FIELDS}]")
  endif ()
endfunction()

function(setup_field_target FIELD FIELD_INDEX FEATURES_STRING)
  add_library(icicle_field SHARED)

  # Split FEATURES_STRING into a list using "," as the separator
  string(REPLACE "," ";" FEATURES_LIST ${FEATURES_STRING})

  # customize the field lib to choose what to include
  handle_field(icicle_field) # basic field methods, including vec ops
  # Handle features
  handle_ntt(icicle_field "${FEATURES_LIST}")
  handle_ext_field(icicle_field "${FEATURES_LIST}")
  handle_poseidon(icicle_field "${FEATURES_LIST}")
  handle_poseidon2(icicle_field "${FEATURES_LIST}")
  handle_sumcheck(icicle_field "${FEATURES_LIST}")
  # Add additional feature handling calls here

  set_target_properties(icicle_field PROPERTIES OUTPUT_NAME "icicle_field_${FIELD}")
  target_link_libraries(icicle_field PUBLIC icicle_device pthread OpenMP::OpenMP_CXX)

  # Ensure FIELD is defined in the cache for backends to see
  set(FIELD "${FIELD}" CACHE STRING "")
  add_compile_definitions(FIELD=${FIELD} FIELD_ID=${FIELD_INDEX})

  install(TARGETS icicle_field
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()

