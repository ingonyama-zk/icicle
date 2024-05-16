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
    if(DEFINED CUDA_ARCH) # user defined arch takes priority
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH} PARENT_SCOPE)
    elseif(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0") # otherwise, use native to detect GPU arch
        set(CMAKE_CUDA_ARCHITECTURES native PARENT_SCOPE)
    else()
        find_program(_nvidia_smi "nvidia-smi")

        if(_nvidia_smi)
            execute_process(
                COMMAND ${_nvidia_smi} --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE GPU_COMPUTE_CAPABILITIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            # Process the output to form the CUDA architectures string
            string(REPLACE "\n" ";" GPU_COMPUTE_CAPABILITIES_LIST "${GPU_COMPUTE_CAPABILITIES}")

            set(CUDA_ARCHITECTURES "")
            foreach(CAPABILITY ${GPU_COMPUTE_CAPABILITIES_LIST})
                # Remove the dot in compute capability to match CMake format
                string(REPLACE "." "" CAPABILITY "${CAPABILITY}")
                if(CUDA_ARCHITECTURES)
                    set(CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES};${CAPABILITY}")
                else()
                    set(CUDA_ARCHITECTURES "${CAPABILITY}")
                endif()
            endforeach()

            message("Setting CMAKE_CUDA_ARCHITECTURES to: ${CUDA_ARCHITECTURES}")        
            set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}" PARENT_SCOPE)                        
        else()
            # no GPUs found, like on Github CI runners
            message("Setting CMAKE_CUDA_ARCHITECTURES to: 50") 
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
