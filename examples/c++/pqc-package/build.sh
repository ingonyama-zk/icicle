#!/bin/bash

# Check if NVIDIA drivers are available
check_nvidia_drivers() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA drivers found (nvidia-smi available)"
        return 0
    elif [ -f /proc/driver/nvidia/version ]; then
        echo "✅ NVIDIA drivers found (/proc/driver/nvidia/version exists)"
        return 0
    elif [ -d /usr/lib/nvidia ] || [ -d /usr/lib/x86_64-linux-gnu/nvidia ]; then
        echo "✅ NVIDIA drivers found (driver libraries detected)"
        return 0
    else
        echo "❌ NVIDIA drivers not found. This example requires NVIDIA GPU support."
        echo "Please install NVIDIA drivers and ensure your GPU is properly configured."
        return 1
    fi
}

# Function to check if ICICLE PQC is installed
check_icicle_pqc() {
    if [ -z "$ICICLE_PQC_INSTALL_DIR" ]; then
        echo "❌ ICICLE_PQC_INSTALL_DIR environment variable not set."
        echo "Please set it to your ICICLE PQC installation directory:"
        echo "  export ICICLE_PQC_INSTALL_DIR=/path/to/icicle/pqc/install"
        return 1
    fi
    
    if [ ! -d "$ICICLE_PQC_INSTALL_DIR" ]; then
        echo "❌ ICICLE PQC installation directory not found: $ICICLE_PQC_INSTALL_DIR"
        echo "Please ensure ICICLE PQC is properly installed."
        return 1
    fi
    
    echo "✅ ICICLE PQC installation found at: $ICICLE_PQC_INSTALL_DIR"
    return 0
}

# Main build function
build_example() {
    echo "🔨 Building ICICLE PQC example..."
    
    # Create build directory
    mkdir -p build
    
    # Configure and build with CMake
    if cmake -S . -B build -DICICLE_PQC_INSTALL_DIR="$ICICLE_PQC_INSTALL_DIR"; then
        echo "✅ CMake configuration successful"
    else
        echo "❌ CMake configuration failed"
        return 1
    fi
    
    if cmake --build build; then
        echo "✅ Build successful"
        echo ""
        echo "🚀 To run the example:"
        echo "  ./build/pqc_example [batch_size]"
        echo ""
        echo "Examples:"
        echo "  ./build/pqc_example        # Run with batch size 1"
        echo "  ./build/pqc_example 10     # Run with batch size 10"
        return 0
    else
        echo "❌ Build failed"
        return 1
    fi
}

# Main script execution
echo "ICICLE PQC Example Build Script"
echo "==============================="
echo ""

# Check for NVIDIA drivers
if ! check_nvidia_drivers; then
    exit 1
fi

echo ""

# Check for ICICLE PQC installation
if ! check_icicle_pqc; then
    exit 1
fi

echo ""

# Build the example
if ! build_example; then
    exit 1
fi

echo ""
echo "🎉 Build completed successfully!" 