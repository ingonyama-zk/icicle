function(check_field)
  # set(SUPPORTED_FIELDS babybear;stark252)
  set(SUPPORTED_FIELDS babybear;stark252;m31;goldilocks)# m31 is not implemented yet 


  set(IS_FIELD_SUPPORTED FALSE)
  set(I 1000)
  foreach (SUPPORTED_FIELD ${SUPPORTED_FIELDS})
    math(EXPR I "${I} + 1")
    if (FIELD STREQUAL SUPPORTED_FIELD)
      add_compile_definitions(FIELD_ID=${I})
      set(IS_FIELD_SUPPORTED TRUE)
    endif ()
  endforeach()

  if (NOT IS_FIELD_SUPPORTED)
    message( FATAL_ERROR "The value of FIELD variable: ${FIELD} is not one of the supported fields: ${SUPPORTED_FIELDS}" )
  endif ()
endfunction()

function(setup_field_target)
    add_library(icicle_field SHARED
      src/fields/ffi_extern.cpp
      src/vec_ops.cpp
      src/matrix_ops.cpp
    )
    # handle APIs that are for some curves only
    add_ntt_sources_or_disable()
    set_target_properties(icicle_field PROPERTIES OUTPUT_NAME "icicle_field_${FIELD}")
    target_link_libraries(icicle_field PUBLIC icicle_device)

    # Make sure FIELD is defined in the cache for backends to see
    set(FIELD "${FIELD}" CACHE STRING "")
    target_compile_definitions(icicle_field PUBLIC FIELD=${FIELD})
    if (EXT_FIELD)
      set(EXT_FIELD "${EXT_FIELD}" CACHE STRING "")
      target_compile_definitions(icicle_field PUBLIC EXT_FIELD=${EXT_FIELD})
    endif()

    install(TARGETS icicle_field
      RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
      LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
      ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()

function(add_ntt_sources_or_disable)
  set(SUPPORTED_FIELDS_WITHOUT_NTT grumpkin)

  if (NOT FIELD IN_LIST SUPPORTED_FIELDS_WITHOUT_NTT)
    add_compile_definitions(NTT_ENABLED)
    target_sources(icicle_field PRIVATE  
      src/ntt.cpp 
      src/polynomials/polynomials.cpp 
      src/polynomials/polynomials_c_api.cpp
      src/polynomials/polynomials_abstract_factory.cpp
    )
  else()
    set(NTT OFF CACHE BOOL "NTT not available for field" FORCE)
  endif()

endfunction()