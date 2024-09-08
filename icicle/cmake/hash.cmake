

function(setup_hash_target)
  add_library(icicle_hash SHARED)
  target_sources(icicle_hash PRIVATE 
    src/hash/keccak.cpp
    src/hash/blake2s.cpp
    src/hash/merkle_tree.cpp
  )
  
  target_link_libraries(icicle_hash PUBLIC icicle_device)

  install(TARGETS icicle_hash
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")
endfunction()

