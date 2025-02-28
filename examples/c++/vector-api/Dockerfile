# syntax=docker/dockerfile:1

# Use an official Ubuntu base image. 
FROM --platform=linux/amd64 ubuntu:24.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    make \
    clang \
    software-properties-common \
    build-essential \
    git \
    ninja-build \
    python3.8 \
    xxd




# Install cmake, the version in apt is too old 
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
-o /tmp/cmake-install.sh \
&& chmod u+x /tmp/cmake-install.sh \
&& mkdir /opt/cmake-3.24.1 \
&& /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
&& rm /tmp/cmake-install.sh \
&& ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# Download, build and Install MIcrosoft's DirectXShaderCompiler ('dxc')

RUN curl -L https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2502/linux_dxc_2025_02_20.x86_64.tar.gz -o dxc.tar.gz
RUN tar -xzf dxc.tar.gz
ENV PATH="$PATH:linux_dxc_2025_02_20.x86_64/bin/dxc"

# Set the Android NDK version and download URL
ARG NDK_VERSION=28
ARG NDK_URL=https://dl.google.com/android/repository/android-ndk-r${NDK_VERSION}-linux.zip
ENV ANDROID_NDK=/opt/android-ndk-r28/
ENV NDK_DIR=/opt/android-ndk-r28/
ENV CMAKE_ANDROID_NDK=/opt/android-ndk-r28/

# Download and extract the Android NDK
RUN curl -L ${NDK_URL} -o android-ndk.zip
RUN unzip android-ndk -d /opt/ 




# # Set environment variables for NDK
# ENV ANDROID_NDK_HOME=${NDK_DIR}
# ENV PATH="${ANDROID_NDK_HOME}:${PATH}"

# # Verify installation
#RUN ${NDK_DIR}/ndk-build --version

# Set the working directory (optional)
WORKDIR /workspace

# Command to keep the container running (for development)
CMD ["bash"]
