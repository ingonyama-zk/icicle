#!/bin/bash

# Check if NVIDIA drivers are available
check_nvidia_drivers() {
    if command -v nvidia-smi &> /dev/null; then
        return 0
    elif [ -f /proc/driver/nvidia/version ]; then
        return 0
    elif [ -d /usr/lib/nvidia ] || [ -d /usr/lib/x86_64-linux-gnu/nvidia ]; then
        return 0
    else
        return 1
    fi
}

# Check for NVIDIA drivers
if ! check_nvidia_drivers; then
    echo "❌ NVIDIA drivers not found. This PQC example requires NVIDIA GPU support."
    echo "Please install NVIDIA drivers and ensure your GPU is properly configured."
    exit 1
fi

echo "✅ NVIDIA drivers detected."
echo "Check out the README file. You will need to install ICICLE PQC and follow the build instructions." 