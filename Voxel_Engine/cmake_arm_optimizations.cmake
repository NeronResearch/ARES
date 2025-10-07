# ARM/Raspberry Pi specific compiler optimizations
# Add this to your existing CMakeLists.txt

# Detect ARM architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(ARM_ARCHITECTURE TRUE)
    message(STATUS "ARM architecture detected - enabling ARM optimizations")
else()
    set(ARM_ARCHITECTURE FALSE)
endif()

# ARM-specific compiler flags
if(ARM_ARCHITECTURE)
    # Raspberry Pi 4 optimizations (Cortex-A72)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crc -mtune=cortex-a72")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard")
    # Raspberry Pi 3 optimizations (Cortex-A53)  
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mtune=cortex-a53")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4 -mfloat-abi=hard")
    endif()
    
    # ARM-specific optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fprefetch-loop-arrays")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fno-signed-zeros")
    
    # Cache and memory optimizations for ARM
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -mtune=native")
    
    # NEON SIMD support
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARM_NEON")
    
    message(STATUS "ARM compiler flags: ${CMAKE_CXX_FLAGS}")
    
else()
    # x86/x64 optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fprefetch-loop-arrays")
endif()

# OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(your_target_name OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found - parallel processing disabled")
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC specific optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fwhole-program")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use -fprofile-correction")
    
    if(ARM_ARCHITECTURE)
        # ARM-specific GCC flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -marm -munaligned-access")
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang specific optimizations  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fuse-ld=lld")
    
    if(ARM_ARCHITECTURE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mllvm -arm-global-merge")
    endif()
endif()

# Link-time optimizations
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Debug vs Release configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -s")
    
    # Release-specific optimizations for ARM
    if(ARM_ARCHITECTURE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
    endif()
endif()

message(STATUS "Final compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Final linker flags: ${CMAKE_EXE_LINKER_FLAGS}")