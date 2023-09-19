# Use the specified base image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    protobuf-compiler \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Golang
ENV GOLANG_VERSION 1.21.1
RUN curl -L https://golang.org/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz | tar -xz -C /usr/local
ENV PATH="/usr/local/go/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy the content of the local directory to the working directory
COPY . .

# Specify the default command for the container
CMD ["/bin/bash"]
