
#!/bin/bash

# TODO update for V3

# # Exit immediately on error
# set -e

# mkdir -p build/example
# mkdir -p build/icicle

# # Configure and build Icicle
# cmake -S ../../../icicle/ -B build/icicle -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254
# cmake --build build/icicle

# # Configure and build the example application
# cmake -S . -B build/example
# cmake --build build/example