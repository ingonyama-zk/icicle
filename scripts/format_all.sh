#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
show_help() {
    echo "Usage: $0 [DIRECTORY] [--check]"
    echo
    echo "Options:"
    echo "  DIRECTORY       The directory to process (default: current directory)"
    echo "  --check         Only check formatting, do not modify files"
    exit 1
}

# Default values
DIRECTORY="."
CHECK_ONLY=false

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            DIRECTORY="$arg"
            shift
            ;;
    esac
done

# Validate the directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: $DIRECTORY is not a valid directory."
    exit 1
fi

# Create a temporary file to track failures
FAILED_FILES=$(mktemp)

# Initialize failure flag in the file
echo 0 > "$FAILED_FILES"

# Find and process files, excluding `wrappers` directory
find "$DIRECTORY" \
    \( -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.metal' -o -name '*.metalh' \) \
    -not -path "$DIRECTORY/wrappers/*" | while read -r file; do
    if [ "$CHECK_ONLY" = true ]; then
        # Check if the file is properly formatted
        if ! clang-format -output-replacements-xml "$file" | grep -q "<replacement "; then
            echo "✅ $file is properly formatted."
        else
            echo "❌ $file needs formatting."
            echo 1 > "$FAILED_FILES"  # Mark as failed
        fi
    else
        # Format the file
        echo "Formatting $file"
        clang-format -i "$file"
    fi
done

# Read the failure flag from the file
FAILED=$(cat "$FAILED_FILES")
rm "$FAILED_FILES"  # Clean up the temporary file

# If any file failed, exit with a non-zero status
if [ "$FAILED" -ne 0 ]; then
    echo
    echo "Some files need formatting. Please run the script without '--check' to format them."
    exit 1
fi

echo "All files processed successfully."