cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 3.31)

# Set language to C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add kokkos sources
add_subdirectory(extern/kokkos)

# If building with CUDA
if (Kokkos_ENABLE_CUDA)
    # Add nvcc flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler fno-lto")
endif()

# Link libraries
link_libraries(Kokkos::kokkos)

# Add pybind11 sources
add_subdirectory(extern/pybind11)

# Add linking to Python
pybind11_add_module(ppv src/main.cpp)

# Copy file "test.py"
configure_file(src/test.py test.py COPYONLY)