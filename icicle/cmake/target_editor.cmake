# The following functions, each adds a function to a target.
# In addition, some can check feature is enabled for target, given FEATURE_LIST.

function(handle_field TARGET)
  target_sources(${TARGET} PRIVATE
      src/fields/ffi_extern.cpp
      src/vec_ops.cpp
      src/matrix_ops.cpp
      src/program/program_c_api.cpp
      src/symbol/symbol_api.cpp
  )
endfunction()

function(handle_curve TARGET)
  target_sources(${TARGET} PRIVATE
    src/curves/ffi_extern.cpp
    src/curves/montgomery_conversion.cpp
  )
endfunction()

function(handle_ring TARGET)
  target_sources(${TARGET} PRIVATE
    src/fields/ffi_extern.cpp
    src/vec_ops.cpp
    src/rings/rns_vec_ops.cpp
    src/rings/polyring_vec_ops.cpp
    src/rings/random_sampling.cpp
    src/matrix_ops.cpp
    src/program/program_c_api.cpp
    src/symbol/symbol_api.cpp
    src/balanced_decomposition.cpp
    src/norm.cpp
    src/jl_projection.cpp)
endfunction()

function(handle_ntt TARGET FEATURE_LIST)
  if(NTT AND "NTT" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC NTT=${NTT})
    target_sources(${TARGET} PRIVATE
      src/ntt.cpp
      src/polynomials/polynomials.cpp
      src/polynomials/polynomials_c_api.cpp
      src/polynomials/polynomials_abstract_factory.cpp
    )
    set(NTT ON CACHE BOOL "Enable NTT feature" FORCE)
  else()
    set(NTT OFF CACHE BOOL "NTT not available for this field" FORCE)
  endif()
endfunction()

function(handle_ext_field TARGET FEATURE_LIST)
  if(EXT_FIELD AND "EXT_FIELD" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC EXT_FIELD=${EXT_FIELD})
    set(EXT_FIELD ON CACHE BOOL "Enable EXT_FIELD feature" FORCE)
  else()
    set(EXT_FIELD OFF CACHE BOOL "EXT_FIELD not available for this field" FORCE)
  endif()
endfunction()

function(handle_msm TARGET FEATURE_LIST)
  if(MSM AND "MSM" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC MSM=${MSM})
    target_sources(${TARGET} PRIVATE src/msm.cpp)
    set(MSM ON CACHE BOOL "Enable MSM feature" FORCE)
  else()
    set(MSM OFF CACHE BOOL "MSM not available for this curve" FORCE)
  endif()
endfunction()

function(handle_g2 TARGET FEATURE_LIST)
  if(G2 AND "G2" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC G2_ENABLED=${G2})
    set(G2 "G2" CACHE BOOL "Enable G2 feature" FORCE)
  else()
    set(G2 OFF CACHE BOOL "G2 not available for this curve" FORCE)
  endif()
endfunction()

function(handle_ecntt TARGET FEATURE_LIST)
  if(ECNTT AND "ECNTT" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC ECNTT=${ECNTT})
    target_sources(${TARGET} PRIVATE src/ecntt.cpp)
    set(ECNTT ON CACHE BOOL "Enable ECNTT feature" FORCE)
  else()
    set(ECNTT OFF CACHE BOOL "ECNTT not available for this curve" FORCE)
  endif()
endfunction()

function(handle_poseidon TARGET FEATURE_LIST)
  if(POSEIDON AND "POSEIDON" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC POSEIDON=${POSEIDON})
    target_sources(${TARGET} PRIVATE src/hash/poseidon.cpp src/hash/poseidon_c_api.cpp)
    set(POSEIDON ON CACHE BOOL "Enable POSEIDON feature" FORCE)
  else()
    set(POSEIDON OFF CACHE BOOL "POSEIDON not available for this field" FORCE)
  endif()
endfunction()

function(handle_poseidon2 TARGET FEATURE_LIST)
  if(POSEIDON2 AND "POSEIDON2" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC POSEIDON2=${POSEIDON2})
    target_sources(${TARGET} PRIVATE src/hash/poseidon2.cpp src/hash/poseidon2_c_api.cpp)
    set(POSEIDON2 ON CACHE BOOL "Enable POSEIDON2 feature" FORCE)
  else()
    set(POSEIDON2 OFF CACHE BOOL "POSEIDON2 not available for this field" FORCE)
  endif()
endfunction()

function(handle_sumcheck TARGET FEATURE_LIST)
  if(SUMCHECK AND "SUMCHECK" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC SUMCHECK=${SUMCHECK})
    target_sources(${TARGET} PRIVATE src/sumcheck/sumcheck.cpp src/sumcheck/sumcheck_c_api.cpp src/program/program_c_api.cpp)
    set(SUMCHECK ON CACHE BOOL "Enable SUMCHECK feature" FORCE)
  else()
    set(SUMCHECK OFF CACHE BOOL "SUMCHECK not available for this field" FORCE)
  endif()
endfunction()

function(handle_pairing TARGET FEATURE_LIST)
  if(G2 AND "PAIRING" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC PAIRING=1)
    target_sources(${TARGET} PRIVATE src/pairing.cpp)
  endif()
endfunction()

function(handle_fri TARGET FEATURE_LIST)
  if(FRI AND "FRI" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC FRI=${FRI})
    target_sources(${TARGET} PRIVATE src/fri/fri.cpp src/fri/fri_c_api.cpp)
    target_link_libraries(${TARGET} PRIVATE icicle_hash)
    set(FRI ON CACHE BOOL "Enable FRI feature" FORCE)
  else()
    set(FRI OFF CACHE BOOL "FRI not available for this field" FORCE)
  endif()
endfunction()
