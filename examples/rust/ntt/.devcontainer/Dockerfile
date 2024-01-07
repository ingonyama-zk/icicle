# Use the specified base image
#FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    protobuf-compiler \
    curl \
    build-essential \
    git \
    llvm \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory in the container
WORKDIR /icicle-example

# Copy the content of the local directory to the working directory
COPY . .

# Specify the default command for the container
CMD ["/bin/bash"]