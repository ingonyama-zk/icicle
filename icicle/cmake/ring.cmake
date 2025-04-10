include(cmake/features.cmake)
include(cmake/target_editor.cmake)

function(extract_ring_names RING_NAMES_OUT)
  set(RING_NAMES "")

  foreach (ITEM ${ICICLE_RINGS})
    string(REPLACE ":" ";" ITEM_SPLIT ${ITEM})
    list(GET ITEM_SPLIT 1 RING_NAME)
    list(APPEND RING_NAMES ${RING_NAME})
  endforeach()

  # Output the list of RING names
  set(${RING_NAMES_OUT} "${RING_NAMES}" PARENT_SCOPE)
endfunction()

function(check_ring RING RING_INDEX_OUT FEATURES_STRING_OUT)
  set(IS_RING_SUPPORTED FALSE)
  foreach (ITEM ${ICICLE_RINGS})
    string(REPLACE ":" ";" ITEM_SPLIT ${ITEM})
    list(GET ITEM_SPLIT 0 RING_INDEX)
    list(GET ITEM_SPLIT 1 RING_NAME)
    list(GET ITEM_SPLIT 2 FEATURES_STRING)

    if (RING STREQUAL RING_NAME)
      set(IS_RING_SUPPORTED TRUE)
      message(STATUS "building RING_NAME=${RING_NAME} ; RING_INDEX=${RING_INDEX} ; SUPPORTED_FEATURES=${FEATURES_STRING}")
      # Output the RING_INDEX and FEATURES_STRING
      set(${RING_INDEX_OUT} "${RING_INDEX}" PARENT_SCOPE)
      set(${FEATURES_STRING_OUT} "${FEATURES_STRING}" PARENT_SCOPE)
      break()
    endif ()
  endforeach()

  if (NOT IS_RING_SUPPORTED)
    set(ALL_RINGS "")
    extract_ring_names(ALL_RINGS)
    message(FATAL_ERROR "The value of RING variable: ${RING} is not supported: choose from [${ALL_RINGS}]")
  endif ()
endfunction()

function(setup_ring_target RING RING_INDEX FEATURES_STRING)
  add_library(icicle_ring SHARED)

  # Split FEATURES_STRING into a list using "," as the separator
  string(REPLACE "," ";" FEATURES_LIST ${FEATURES_STRING})

  # customize the RING lib to choose what to include
  handle_ring(icicle_ring) # basic RING methods, including vec ops
  # Handle features
  handle_ntt(icicle_ring "${FEATURES_LIST}")
  # Add additional feature handling calls here

  set_target_properties(icicle_ring PROPERTIES OUTPUT_NAME "icicle_ring_${RING}")
  target_link_libraries(icicle_ring PUBLIC icicle_device)

  # Ensure RING is defined in the cache for backends to see
  set(RING "${RING}" CACHE STRING "")
  add_compile_definitions(RING=${RING} RING_ID=${RING_INDEX})

  install(TARGETS icicle_ring
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()

