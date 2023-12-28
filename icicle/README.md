# ICICLE CUDA

## Running tests

```sh
mkdir -p build;
cmake -DBUILD_TESTS=ON -DCURVE=<support_curve> -S . -B build;
cmake --build build;
./build/runner --gtest_brief=1; 
```

## Prerequisites on Ubuntu

Before proceeding, make sure the following software installed:

1. CMake at least version 3.18, which can be downloaded from [cmake.org](https://cmake.org/files/)
   It is recommended to have the latest version installed.
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu) version 12.0 or newer.
3. GCC - version 9 or newer recommended.

## Troubleshooting

In case you encounter problems during the build, please follow the points below to troubleshoot:

### 1 - Check CMake log files

If there are issues with the CMake configuration, please check the logs which are located in the `./build/CMakeFiles` directory. Depending on the version of CMake, the log file may have a different name. For example, for CMake version 3.20, one of log files is called `CMakeConfigureLog.yaml`.

### 2 - Check for conflicting GCC versions

Make sure that there are no conflicting versions of GCC installed. You can use the following commands to install and switch between different versions:

```sh
sudo update-alternatives --install /usr/bin/gcc gcc /home/linuxbrew/.linuxbrew/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /home/linuxbrew/.linuxbrew/bin/g++-12 12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```

Then you can select with the following command

```sh
sudo update-alternatives --config gcc
```

### 3 - Check for conflicting binaries in PATH

Make sure that there are no conflicting binaries in the PATH environment variable. For example, if `/home/linuxbrew/.linuxbrew/bin` precedes `/usr/bin/` in the PATH, it will override the `update-alternatives` settings.

### 4 - Add nvvm path to PATH

If you encounter the error `cicc not found`, make sure to add the nvvm path to PATH. For example, for CUDA version 12.1, the nvvm path is `/usr/local/cuda-12.1/nvvm/bin`.

### 5 - Add CUDA libraries to the project

If you encounter the error `Failed to extract nvcc implicit link line`, add the following code to the CMakeLists.txt file after enabling CUDA:

```c
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
   enable_language(CUDA)
   find_package(CUDAToolkit)
   target_link_libraries(project CUDA::cudart)
   target_link_libraries(project CUDA::cuda_driver)
else()
   message(STATUS "No CUDA compiler found")
endif()
```

### 6 - Fix update alternatives

If the `update-alternatives` settings are broken, you can try to fix them with the following command:

`yes '' | update-alternatives --force --all`

### 7 - ..bin/crt/link.stub: No such file or directory

If you encounter the error, check if the `$CUDA_HOME/bin/crt/link.stub` file is available.

Otherwise create a symlink. For example, if the CUDA toolkit is installed with apt-get to the default path, you can create a symlink with the following command:

`ln -sf /usr/local/cuda-12.1/bin/crt/link.stub /usr/lib/nvidia-cuda-toolkit/bin/crt/link.stub`

Alternatively, you can replace the old CUDA root with a symlink to the new CUDA installation with the following command:

`ln -sf /usr/local/cuda-12.1/ /usr/lib/nvidia-cuda-toolkit/`
