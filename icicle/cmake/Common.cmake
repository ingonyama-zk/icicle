function(set_env)
    set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    set(CMAKE_CUDA_STANDARD 17 PARENT_SCOPE)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE PARENT_SCOPE)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE PARENT_SCOPE)

    if("$ENV{ICICLE_PIC}" STREQUAL "OFF" OR ICICLE_PIC STREQUAL "OFF")
        message(WARNING "Note that PIC (position-independent code) is disabled.")
    else()
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

function(set_gpu_env)
    # add the target cuda architectures
    # each additional architecture increases the compilation time and output file size
    if(${CMAKE_VERSION} VERSION_LESS "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH} PARENT_SCOPE)
    else()
    find_program(_nvidia_smi "nvidia-smi")

    if(_nvidia_smi)
        set(DETECT_GPU_COUNT_NVIDIA_SMI 0)

        # execute nvidia-smi -L to get a short list of GPUs available
        exec_program(${_nvidia_smi_path} ARGS -L
        OUTPUT_VARIABLE _nvidia_smi_out
        RETURN_VALUE _nvidia_smi_ret)

        # process the stdout of nvidia-smi
        if(_nvidia_smi_ret EQUAL 0)
        # convert string with newlines to list of strings
        string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")

        foreach(_line ${_nvidia_smi_out})
            if(_line MATCHES "^GPU [0-9]+:")
            math(EXPR DETECT_GPU_COUNT_NVIDIA_SMI "${DETECT_GPU_COUNT_NVIDIA_SMI}+1")

            # the UUID is not very useful for the user, remove it
            string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")

            if(NOT _gpu_info STREQUAL "")
                list(APPEND DETECT_GPU_INFO "${_gpu_info}")
            endif()
            endif()
        endforeach()

        check_num_gpu_info(${DETECT_GPU_COUNT_NVIDIA_SMI} DETECT_GPU_INFO)
        set(DETECT_GPU_COUNT ${DETECT_GPU_COUNT_NVIDIA_SMI})
        endif()
    endif()

    # ##
    if(DETECT_GPU_COUNT GREATER 0)
        set(CMAKE_CUDA_ARCHITECTURES native PARENT_SCOPE) # do native
    else()
        # no GPUs found, like on Github CI runners
        set(CMAKE_CUDA_ARCHITECTURES 50 PARENT_SCOPE) # some safe value
    endif()
    endif()

    # Check CUDA version and, if possible, enable multi-threaded compilation 
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.2")
        message(STATUS "Using multi-threaded CUDA compilation.")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --split-compile 0" PARENT_SCOPE)
    else()
        message(STATUS "Can't use multi-threaded CUDA compilation.")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr" PARENT_SCOPE)
    set(CMAKE_CUDA_FLAGS_RELEASE "" PARENT_SCOPE)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -lineinfo" PARENT_SCOPE)
endfunction()