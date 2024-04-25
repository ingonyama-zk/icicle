function(check_field) set(SUPPORTED_FIELDS babybear; stark252)

  set(IS_FIELD_SUPPORTED FALSE) set(I 1000) foreach (SUPPORTED_FIELD ${SUPPORTED_FIELDS})
    math(EXPR I "${I} + 1") if (FIELD STREQUAL SUPPORTED_FIELD)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFIELD_ID=${I}" PARENT_SCOPE) set(IS_FIELD_SUPPORTED TRUE) endif()
        endforeach()

          if (NOT IS_FIELD_SUPPORTED)
            message(FATAL_ERROR
                    "The value of FIELD variable: ${FIELD} is not one of the supported fields: ${SUPPORTED_FIELDS}")
              endif() endfunction()
