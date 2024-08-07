# ICICLE CUDA

## Running tests

```sh
mkdir -p build;
cmake -DBUILD_TESTS=ON -DCURVE=<supported_curve> -S . -B build;
cmake --build build;
./build/tests/runner --gtest_brief=1
```

The command above will build ICICLE Core and run the ctest.

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

### 8 - Error while loading shared libraries

`cmake: error while loading shared libraries: libssl.so.10: cannot open shared object file: No such file or directory`

Make sure `libssl` is installed.

```sh
sudo apt-get update
sudo apt-get install libssl1.0.0 libssl-dev
```

### 9 - PIC and Linking against shared libraries

Note that currently - ICICLE is static library with [PIC](https://en.wikipedia.org/wiki/Position-independent_code) enabled by default. You can disable it by setting either `ICICLE_PIC` environment variable to `OFF` or passing `-DICICLE_PIC=OFF` to CMake.

## Running with Nix

If you have Nix or NixOs installed on your machine, you can create a development shell to load all build dependencies and set the required environmental variables.

From the ```/icicle/icicle```  directory run the following command.

```sh
nix-shell --pure cuda-shell.nix
```

This will install everything you need to build and run ICICLE Core.

