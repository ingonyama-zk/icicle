#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
show_help() {
    echo "Usage: $0 [DIRECTORY] [--check] [--exclude REGEX]"
    echo
    echo "Options:"
    echo "  DIRECTORY       The directory to process (default: current directory)"
    echo "  --check         Only check formatting, do not modify files"
    echo "  --exclude       Regex pattern for directories/files to exclude"
    echo "  -h, --help      Show this help message"
    exit 1
}

# Default values
DIRECTORY="."
CHECK_ONLY=false
EXCLUDE_REGEX=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --exclude)
            EXCLUDE_REGEX="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            DIRECTORY="$1"
            shift
            ;;
    esac
done

# Exclude Objective-C headers and Metal-specific files by default
EXCLUDE_REGEX+="|Metal/Metal\.h|.*\.m$|.*\.mm$"

# Validate the directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: $DIRECTORY is not a valid directory."
    exit 1
fi

# Create a temporary file to track failures
FAILED_FILES=$(mktemp)
echo 0 > "$FAILED_FILES" # Initialize failure flag

# Find files and apply exclusions using grep
FILES=$(find "$DIRECTORY" \
    \( -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.metal' -o -name '*.metalh' \) \
    -type f | grep -vE "$EXCLUDE_REGEX")

# Process files
echo "$FILES" | while read -r file; do
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

# Read the failure flag from the temporary file
FAILED=$(cat "$FAILED_FILES")
rm "$FAILED_FILES" # Clean up

# Exit with non-zero status if any file needs formatting
if [ "$FAILED" -ne 0 ]; then
    echo
    echo "Some files need formatting. Please run the script without '--check' to format them."
    exit 1
fi

echo "All files processed successfully."