#!/bin/bash

echo "ICICLE PQC Example Run Script"
echo "============================="
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [ICICLE_PQC_INSTALL_DIR] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  ICICLE_PQC_INSTALL_DIR  Path to ICICLE PQC installation (optional if env var is set)"
    echo "  batch_size              Batch size for the example (optional, defaults to 1)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use existing ICICLE_PQC_INSTALL_DIR env var"
    echo "  $0 /path/to/icicle/pqc/install      # Set install dir and use default batch size"
    echo "  $0 /path/to/icicle/pqc/install 10   # Set install dir and batch size"
    echo "  $0 \"\" 10                            # Use env var for install dir, set batch size to 10"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Handle ICICLE_PQC_INSTALL_DIR argument
if [ $# -ge 1 ] && [ -n "$1" ]; then
    export ICICLE_PQC_INSTALL_DIR="$1"
    echo "📍 Using ICICLE PQC installation directory: $ICICLE_PQC_INSTALL_DIR"
    # Shift arguments so remaining args are passed to the example
    shift
elif [ -z "$ICICLE_PQC_INSTALL_DIR" ]; then
    echo "❌ No ICICLE_PQC_INSTALL_DIR provided and environment variable not set."
    echo ""
    show_usage
    exit 1
else
    echo "📍 Using ICICLE PQC installation directory from env var: $ICICLE_PQC_INSTALL_DIR"
fi

echo ""

# Build the example first
echo "🔨 Building example..."
if ! ./build.sh; then
    echo "❌ Build failed. Cannot run example."
    exit 1
fi

echo ""
echo "🚀 Running example..."

# Check if the executable exists
if [ ! -f "./build/pqc_example" ]; then
    echo "❌ Example executable not found at ./build/pqc_example"
    exit 1
fi

# Run the example with any remaining arguments
if [ $# -eq 0 ]; then
    echo "Running with default batch size (1)..."
    ./build/pqc_example
else
    echo "Running with batch size: $1"
    ./build/pqc_example "$@"
fi

echo ""
echo "🎉 Example completed!" 