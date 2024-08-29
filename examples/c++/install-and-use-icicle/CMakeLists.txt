# Add the executable
add_executable(example example.cpp)
# Link the libraries
target_link_libraries(example icicle_device icicle_field_bn254 icicle_curve_bn254)

# OPTIONAL (if not installed in default location)

# The following is setting compile and runtime paths for headers and libs assuming
#   - headers in /custom/path/icicle/include
#   - libs in/custom/path/icicle/lib

# Include directories
target_include_directories(example PUBLIC /custom/path/icicle/include)
# Library directories
target_link_directories(example PUBLIC /custom/path/icicle/lib/)
# Set the RPATH so linker finds icicle libs at runtime
set_target_properties(example PROPERTIES
                      BUILD_RPATH /custom/path/icicle/lib/
                      INSTALL_RPATH /custom/path/icicle/lib/)