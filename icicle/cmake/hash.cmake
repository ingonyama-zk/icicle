

function(setup_hash_target)
  add_library(icicle_hash SHARED)
  target_sources(icicle_hash PRIVATE 
    src/hash/keccak.cpp
    src/hash/blake2s.cpp
    src/hash/merkle_tree.cpp
    src/hash/hash_c_api.cpp
    src/hash/merkle_c_api.cpp
  )
  
  target_link_libraries(icicle_hash PUBLIC icicle_device)
  if(ANDROID)
    target_link_libraries(icicle_hash PRIVATE ${log-lib})
  endif()

  install(TARGETS icicle_hash
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()

