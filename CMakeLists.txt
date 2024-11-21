# Set CMP0048 to NEW
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_minimum_required(VERSION 3.31)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

# Set language to C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If ccache is found, use it
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    # Inform user
    message(STATUS "ccache found, using it")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

# Compile with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add kokkos sources
add_subdirectory(extern/kokkos)

# Link libraries
link_libraries(Kokkos::kokkos)

# Add Python bindings
add_subdirectory(extern/pybind11)

# Add library
set(PYBIND11_NEWPYTHON ON)
pybind11_add_module(ppyv ppyv/main.cpp)

# Pass the version to the target
target_compile_definitions(ppyv PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Instal
install(TARGETS ppyv LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
