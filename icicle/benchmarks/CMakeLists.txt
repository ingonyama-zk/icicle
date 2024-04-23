
add_executable(benches benches.cu)
target_link_libraries(benches benchmark::benchmark)
target_include_directories(benches PUBLIC ${CMAKE_SOURCE_DIR}/include/)
find_package(benchmark REQUIRED)
