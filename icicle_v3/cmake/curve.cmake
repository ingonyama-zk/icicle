function(check_curve)
  set(SUPPORTED_CURVES bn254;bls12_381;bls12_377;bw6_761;grumpkin)

  set(IS_CURVE_SUPPORTED FALSE)
  set(I 0)
  foreach (SUPPORTED_CURVE ${SUPPORTED_CURVES})
    math(EXPR I "${I} + 1")
    if (CURVE STREQUAL SUPPORTED_CURVE)
      add_compile_definitions(FIELD_ID=${I})
      add_compile_definitions(CURVE_ID=${I})
      set(IS_CURVE_SUPPORTED TRUE)
    endif ()
  endforeach()

  if (NOT IS_CURVE_SUPPORTED)
    message( FATAL_ERROR "The value of CURVE variable: ${CURVE} is not one of the supported curves: ${SUPPORTED_CURVES}" )
  endif ()
endfunction()

function(setup_curve_target)
  set(FIELD ${CURVE})
  setup_field_target()

  add_library(icicle_curve SHARED
    src/msm.cpp
    src/curves/ffi_extern.cpp
    src/curves/montgomery_conversion.cpp
  )
  target_link_libraries(icicle_curve PUBLIC icicle_device) # for thread local device
  set_target_properties(icicle_curve PROPERTIES OUTPUT_NAME "icicle_curve_${CURVE}")

  # Make sure CURVE is defined in the cache for backends to see
  set(CURVE "${CURVE}" CACHE STRING "")
  target_compile_definitions(icicle_curve PUBLIC CURVE=${CURVE})
  if (G2)
    set(G2 "${G2}" CACHE STRING "")
    target_compile_definitions(icicle_curve PUBLIC G2=${G2})
  endif()
  if (ECNTT)
    target_sources(icicle_curve PRIVATE src/ecntt.cpp)
    set(ECNTT "${ECNTT}" CACHE STRING "")
    target_compile_definitions(icicle_curve PUBLIC ECNTT=${ECNTT})
  endif()
endfunction()