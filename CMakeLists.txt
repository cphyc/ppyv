# Set CMP0048 to NEW
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 3.31)

project(ppv VERSION 0.0.1)

# Set language to C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# If ccache is found, use it
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

# Compile with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add kokkos sources
add_subdirectory(extern/kokkos)

# Link libraries
link_libraries(Kokkos::kokkos)

# Add pybind11 sources
add_subdirectory(extern/pybind11)

# Add linking to Python
pybind11_add_module(ppyv src/main.cpp)

# Copy file "test.py"
configure_file(src/test.py test.py COPYONLY)
