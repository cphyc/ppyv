cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 3.31)

# Set language to C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add kokkos sources
add_subdirectory(kokkos)

# # Add linking to Python
# find_package(Python3 COMPONENTS Development NumPy)

# Add sources in src/
add_executable(off_axis src/main.cpp)

# Link libraries
target_link_libraries(off_axis Kokkos::kokkos)
