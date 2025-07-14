

function(setup_pqc_target)
  if(ICICLE_STATIC_LINK)
    add_library(icicle_pqc STATIC)
  else()
    add_library(icicle_pqc SHARED)
  endif()
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

