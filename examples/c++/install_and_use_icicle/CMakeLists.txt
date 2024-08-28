# Add the executable
add_executable(example example.cpp)
# Link the libraries
target_link_libraries(example icicle_device icicle_field_bn254 icicle_curve_bn254)

# OPTIONAL (if not installed in default location)

# The following is setting compile and runtime paths for headers and libs assuming
#   - headers in /opt/icicle/include
#   - libs in /opt/icicle/lib

# # Include directories
# include_directories(/opt/icicle/include)
# # Library directories
# link_directories(/opt/icicle/lib/)
# Set the RPATH so linker finds icicle libs at runtime
# set_target_properties(example PROPERTIES
#     BUILD_RPATH /opt/icicle/lib/
#     INSTALL_RPATH /opt/icicle/lib/)