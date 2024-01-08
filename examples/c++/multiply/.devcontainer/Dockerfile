# Make sure NVIDIA Container Toolkit is installed on your host

# Use NVIDIA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    nsight-systems-12.2 \
    cmake \
    protobuf-compiler \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Icicle from a GitHub repository
RUN git clone https://github.com/ingonyama-zk/icicle.git  /icicle

# Set the working directory in the container
WORKDIR /icicle-example

# Specify the default command for the container
CMD ["/bin/bash"]
