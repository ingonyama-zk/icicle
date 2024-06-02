function(check_field)
  set(SUPPORTED_FIELDS babybear;stark252)

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
      src/vec_ops.cpp
      src/matrix_ops.cpp
      src/ntt.cpp
    )
    set_target_properties(icicle_field PROPERTIES OUTPUT_NAME "icicle_field_${FIELD}")

    # Make sure FIELD is defined in the cache for backends to see
    set(FIELD "${FIELD}" CACHE STRING "")
    target_compile_definitions(icicle_field PUBLIC FIELD=${FIELD})
    if (EXT_FIELD)
      set(EXT_FIELD "${EXT_FIELD}" CACHE STRING "")      
      target_compile_definitions(icicle_field PUBLIC EXT_FIELD=${EXT_FIELD})
    endif()
endfunction()