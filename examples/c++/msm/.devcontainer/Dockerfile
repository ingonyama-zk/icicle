# Make sure NVIDIA Container Toolkit is installed on your host

# Use the specified base image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    build-essential \
    git \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Icicle from a GitHub repository
RUN git clone https://github.com/ingonyama-zk/icicle.git  /opt/icicle

# Set the working directory in the container
WORKDIR /icicle-example

# Specify the default command for the container
CMD ["/bin/bash"]



