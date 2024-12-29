#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

get_llvm_tag() {
    clang_version=$(clang --version | grep -oE "([0-9]+\.[0-9]+\.[0-9]+)" | head -1)
    major_version=$(echo "$clang_version" | cut -d. -f1)
    case "$major_version" in
        16) echo "llvmorg-16.0.0" ;;
        17) echo "llvmorg-17.0.0" ;;
        18) echo "llvmorg-18.0.0" ;;
        15) echo "llvmorg-15.0.0" ;;
        14) echo "llvmorg-14.0.0" ;;
        13) echo "llvmorg-13.0.0" ;;
        *) 
            echo "Unknown clang version $clang_version. Defaulting to llvmorg-16.0.0."
            echo "llvmorg-16.0.0"
            ;;
    esac
}

# Default parameters
LLVM_TAG=$(get_llvm_tag)
INSTALL_PATH=${1:-"$PWD/openmp_install"}

# Check if OpenMP is already installed
if [ -f "$INSTALL_PATH/lib/libomp.a" ] && [ -d "$INSTALL_PATH/include" ]; then
    echo "OpenMP is already installed in $INSTALL_PATH. Skipping build."
    exit 0
fi

echo "Detected Clang version. Using LLVM tag: $LLVM_TAG"

# Clone the llvm-project with sparse checkout for OpenMP
echo "Cloning llvm-project repository with tag: $LLVM_TAG"
git clone --depth 1 --branch "$LLVM_TAG" --filter=blob:none --sparse https://github.com/llvm/llvm-project.git
pushd llvm-project

# Enable sparse checkout and fetch only the required directories
echo "Configuring sparse checkout for OpenMP and ExtendPath..."
git sparse-checkout init --cone
git sparse-checkout set openmp extendpath

popd

# Implement ExtendPath functionality for standalone OpenMP
echo "Injecting ExtendPath into OpenMP CMake..."
cat > llvm-project/openmp/runtime/src/ExtendPath << 'EOF'
function(extend_path result base relative_path)
    # Combine base directory and relative path
    set(${result} "${base}/${relative_path}" PARENT_SCOPE)
endfunction()
EOF

# Create a build directory for OpenMP
mkdir -p llvm-project/openmp/build
pushd llvm-project/openmp/build

# Configure OpenMP build with CMake
echo "Configuring OpenMP build with CMake..."
cmake .. \
    -DOPENMP_STANDALONE_BUILD=TRUE \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DLIBOMP_USE_ADAPTIVE_LOCKS=OFF \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"

# Build and install OpenMP
echo "Building OpenMP..."
make -j$(nproc)
echo "Installing OpenMP to $INSTALL_PATH..."
make install
popd

# Cleanup
rm -rf llvm-project

echo "OpenMP build and installation complete! Installed at: $INSTALL_PATH"