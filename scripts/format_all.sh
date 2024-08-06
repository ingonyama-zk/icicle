#!/bin/bash

# Check if directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY="$1"

# Find and format all C, C++, header, and other relevant files
find "$DIRECTORY" -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' | while read -r file; do
    echo "Formatting $file"
    clang-format -i "$file"
done

echo "All files formatted."
