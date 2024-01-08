# Make sure NVIDIA Container Toolkit is installed on your host

# Use the specified base image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    protobuf-compiler \
    curl \
    build-essential \
    git \
    libboost-all-dev \
    python3-pip \    
    && rm -rf /var/lib/apt/lists/*

# Install Rust
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#ENV PATH="/root/.cargo/bin:${PATH}"

# Install Golang
#ENV GOLANG_VERSION 1.21.1
#RUN curl -L https://golang.org/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz | tar -xz -C /usr/local
#ENV PATH="/usr/local/go/bin:${PATH}"

# Clone Icicle from a GitHub repository
RUN git clone https://github.com/ingonyama-zk/icicle.git  /icicle

# Install Python dependencies
RUN pip install poseidon-hash


# Set the working directory in the container
WORKDIR /icicle-example

# Specify the default command for the container
CMD ["/bin/bash"]



