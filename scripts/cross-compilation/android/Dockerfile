# Use Ubuntu LTS as base
FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    make \
    clang \
    wget \
    unzip \
    software-properties-common \
    git \
    ninja-build \
    python3.8 \
    adb \
    xxd \
    build-essential \
    pkg-config 

# Install cmake, the version in apt is too old 
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
-o /tmp/cmake-install.sh \
&& chmod u+x /tmp/cmake-install.sh \
&& mkdir /opt/cmake-3.24.1 \
&& /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
&& rm /tmp/cmake-install.sh \
&& ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Download and extract Android NDK
ARG NDK_VERSION=28
ARG NDK_URL=https://dl.google.com/android/repository/android-ndk-r${NDK_VERSION}-linux.zip
ENV ANDROID_NDK=/opt/android-ndk-r${NDK_VERSION}/
ENV ANDROID_NDK_HOME=/opt/android-ndk-r${NDK_VERSION}/
ENV NDK_DIR=/opt/android-ndk-r${NDK_VERSION}/
ENV CMAKE_ANDROID_NDK=/opt/android-ndk-r${NDK_VERSION}/
ENV RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined"

RUN wget -q ${NDK_URL} -O android-ndk.zip \
    && unzip -q android-ndk.zip -d /opt/ \
    && rm android-ndk.zip

# Add Android targets (ARM64, ARMv7, x86_64)
RUN rustup target add \
    aarch64-linux-android \
    armv7-linux-androideabi \
    x86_64-linux-android

# Configure Cargo for cross-compilation
RUN mkdir -p /root/.cargo
COPY <<EOF /root/.cargo/config.toml
[target.aarch64-linux-android]
linker = "${ANDROID_NDK_HOME}toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang"
ar = "${ANDROID_NDK_HOME}toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-ar"

[target.armv7-linux-androideabi]
linker = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi21-clang"
ar = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"

[target.x86_64-linux-android]
linker = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android21-clang"
ar = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
EOF

# Set working directory for builds
WORKDIR /app
VOLUME /app

# Use bash shell
ENTRYPOINT ["/bin/bash", "-c"]